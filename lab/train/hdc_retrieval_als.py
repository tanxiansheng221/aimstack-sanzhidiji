# -*- coding: utf-8 -*-
# §23-v7: 语义级 ALS —— 对话态检索闭式解 (hDC 上文态 → 回复态, 零 SGD)
#   任务: 对话上文(≤64字) → 检索真实下一句 (1 真 + 99 随机负例, Recall@1/@10)
#   h = Σ_i w_i p_i ⊙ E[c_i]  (上文, K1=64, 右对齐新近权重 0.5→2.0)
#   g = Σ_j p2_j ⊙ E[d_j]     (回复, K2=32)
#   ĝ = h M   (M: D×D 闭式岭回归)
#   ALS 交替: (a) 固定 E 解 M; (b) 固定 M 解 E (逐字 D×D 正规方程, 批量)
#     H_c = G_mat ⊙ S_c + diag_rep + λe I,  G_mat = M Mᵀ
#     S_c = Σ_i n_{c,i} w_i² p_i p_iᵀ ;  diag_rep[d] = Σ_j n'_c,j p2_j[d]²
#     rhs_c = Σ_i w_i p_i ⊙ (M @ Gsum_ctx[c,i]) + Σ_j p2_j ⊙ (Gsum_rep[c,j] − Ĝsum[c,j])
import torch, csv, sys, collections, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
dev = 'cuda'

rows = list(csv.reader(open('/workspace/train_split.csv', encoding='utf-8-sig')))
hdr = rows[0]; ui = hdr.index('Utterance'); di = hdr.index('Dialogue_ID')
dialogs = collections.OrderedDict()
for r in rows[1:]:
    dialogs.setdefault(r[di], []).append(r[ui])

cc = collections.Counter(''.join(''.join(us) for us in dialogs.values()))
VOC = 3000
vocab = [c for c, _ in cc.most_common(VOC)]
vmap = {c: i for i, c in enumerate(vocab)}
UNK = len(vocab)          # 3000
PAD = UNK + 1             # 3001 (E 行恒零)

nprng = np.random.default_rng(7)
keys = list(dialogs.keys())
perm = nprng.permutation(len(keys))
n_tr = int(len(keys)*0.85)
tr_keys = set(keys[i] for i in perm[:n_tr])

pairs = []                # (ctx_str, rep_str, is_train)
for k in keys:
    us = [u for u in dialogs[k] if u.strip()]
    for t in range(1, len(us)):
        ctx = ''.join(us[:t])[-64:]
        rep = us[t][:32]
        if ctx and rep:
            pairs.append((ctx, rep, k in tr_keys))
tr = [(a, b) for a, b, istr in pairs if istr]
te = [(a, b) for a, b, istr in pairs if not istr]
print('对话 %d, 句对 训练 %d 测试 %d' % (len(keys), len(tr), len(te)))

K1, K2, D = 64, 32, 1024
E = torch.zeros(VOC+2, D, device=dev)
E[:UNK] = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (UNK, D)).astype(np.float32)).to(dev)
PV1 = torch.from_numpy(nprng.normal(0, 1, (K1, D)).astype(np.float32)).to(dev)
PV1 = PV1/PV1.norm(dim=1, keepdim=True)
PV2 = torch.from_numpy(nprng.normal(0, 1, (K2, D)).astype(np.float32)).to(dev)
PV2 = PV2/PV2.norm(dim=1, keepdim=True)
BW1 = torch.tensor([0.5 + 1.5*t/(K1-1) for t in range(K1)], device=dev)  # t=0 最新

def ids_of(s, L):
    ids = [vmap.get(c, UNK) for c in s][-L:]
    return [PAD]*(L-len(ids)) + ids

def enc_h(ctxs):
    n = len(ctxs)
    Wi = torch.tensor([ids_of(c, K1) for c in ctxs], dtype=torch.int64, device=dev)
    h = torch.zeros(n, D, device=dev)
    for i in range(K1):
        t = K1-1-i
        h += (E[Wi[:, i]]*(BW1[t]*PV1[i]))
    return h, Wi

def enc_g(reps):
    n = len(reps)
    Wi = torch.tensor([ids_of(r, K2) for r in reps], dtype=torch.int64, device=dev)
    g = torch.zeros(n, D, device=dev)
    for j in range(K2):
        g += E[Wi[:, j]]*PV2[j]
    return g, Wi

def enc_dataset(data):
    H = torch.zeros(len(data), D, device=dev)
    G = torch.zeros(len(data), D, device=dev)
    B = 20000
    for s in range(0, len(data), B):
        h, _ = enc_h([x[0] for x in data[s:s+B]])
        g, _ = enc_g([x[1] for x in data[s:s+B]])
        H[s:s+B] = h; G[s:s+B] = g
    return H, G

# 测试候选 (固定: 每题 1 真 + 99 随机, 真回复恒在槽 0)
rng = np.random.default_rng(11)
P = len(te)
cand = np.zeros((P, 100), dtype=np.int64)
for i in range(P):
    neg = rng.choice(P-1, 99, replace=False)
    neg = np.where(neg >= i, neg+1, neg)
    cand[i, 0] = i
    cand[i, 1:] = neg
cand_t = torch.from_numpy(cand).to(dev)

def eval_score(H, G, M):
    Hn = H/H.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Gn = G/G.norm(dim=1, keepdim=True).clamp_min(1e-9)
    r1 = r10 = 0
    B = 4096
    for s in range(0, P, B):
        e = min(s+B, P)
        gh = (Hn[s:e]@M).clamp(-8, 8)
        gh = gh/gh.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sc = torch.einsum('bd,bcd->bc', gh, Gn[cand_t[s:e]])
        r1 += int((sc.argmax(1) == 0).sum())
        r10 += int((sc.topk(10, 1).indices == 0).any(1).sum())
    return r1/P, r10/P

# 词面重叠基线 (子样本 3000 题)
sub = rng.choice(P, 3000, replace=False)
Gte_np = [set(te[i][1]) for i in range(P)]
lex1 = lex10 = 0
for i in sub:
    cs = set(te[i][0])
    sc = [len(cs & Gte_np[c]) for c in cand[i]]
    order = np.argsort(sc)[::-1]
    lex1 += int(order[0] == 0)
    lex10 += int(0 in order[:10])
print('词面重叠基线: R@1 %.3f R@10 %.3f' % (lex1/3000, lex10/3000))

lam, lam_e, ETA = 1e-2, 10.0, 0.5
I_D = torch.eye(D, device=dev)
arD = torch.arange(D, device=dev)

# NCI 预聚合
NCI_CTX = torch.zeros(VOC+2, K1, device=dev)
NCI_REP = torch.zeros(VOC+2, K2, device=dev)
CTX_T = torch.tensor([ids_of(x[0], K1) for x in tr], dtype=torch.int64, device=dev)
REP_T = torch.tensor([ids_of(x[1], K2) for x in tr], dtype=torch.int64, device=dev)
for i in range(K1):
    NCI_CTX[:, i] = torch.bincount(CTX_T[:, i], minlength=VOC+2).to(torch.float32)
for j in range(K2):
    NCI_REP[:, j] = torch.bincount(REP_T[:, j], minlength=VOC+2).to(torch.float32)
NCI_CTX[PAD] = 0; NCI_REP[PAD] = 0
WPV = BW1[:, None]*PV1                                    # (K1,D)

M = torch.eye(D, device=dev)
print('=== §23-v7: 对话态检索 ALS (零 SGD) ===')
for rnd in range(1, 4):
    t0 = time.time()
    H, G = enc_dataset(tr)
    HH = H.T@H; HG = H.T@G
    M = torch.linalg.solve(HH/len(tr) + lam*I_D, HG/len(tr))
    Hte, Gte = enc_dataset(te)
    a1, a10 = eval_score(Hte, Gte, M)
    print('轮%d M-解 (%.0fs): R@1 %.3f R@10 %.3f' % (rnd, time.time()-t0, a1, a10))
    open('/workspace/lm_result.txt', 'a').write('RESULT-v7 rnd%d R1 %.4f R10 %.4f\n' % (rnd, a1, a10))
    # ---- e-解 ----
    t1 = time.time()
    Gmat = M@M.T                                          # (D,D)
    Gsum_ctx = torch.zeros((VOC+2)*K1, D, device=dev)
    Gsum_rep = torch.zeros((VOC+2)*K2, D, device=dev)
    GH = (H@M)                                            # ĝ 全量
    for j in range(K2):
        Gsum_rep.index_add_(0, REP_T[:, j]*K2 + j, G)
        Gsum_rep.index_add_(0, REP_T[:, j]*K2 + j, -GH)
    for i in range(K1):
        Gsum_ctx.index_add_(0, CTX_T[:, i]*K1 + i, G)
    rhs_ctx = torch.einsum('ab,cib->cia', M, Gsum_ctx.reshape(VOC+2, K1, D))
    rhs = ((BW1[None, :, None]*PV1[None, :, :])*rhs_ctx).sum(1) \
        + torch.einsum('jd,cjd->cd', PV2, Gsum_rep.reshape(VOC+2, K2, D))
    Eb = E.clone()
    B = 250
    for s in range(0, VOC+2, B):
        S = torch.einsum('ck,kd,ke->cde', NCI_CTX[s:s+B], WPV, WPV)
        Hs = Gmat[None]*S
        Hs[:, arD, arD] += torch.einsum('cj,jd->cd', NCI_REP[s:s+B], PV2**2) + lam_e
        Eb[s:s+B] = torch.linalg.solve(Hs, rhs[s:s+B, :, None])[..., 0]
    E += ETA*(Eb - E)
    E[PAD] = 0
    print('  e-解 (%.0fs) 完成 (η=%.1f λe=%.0f)' % (time.time()-t1, ETA, lam_e))

Hte, Gte = enc_dataset(te)
a1, a10 = eval_score(Hte, Gte, M)
print('\nv7 最终: R@1 %.3f R@10 %.3f' % (a1, a10))
open('/workspace/lm_result.txt', 'a').write('RESULT-v7 final R1 %.4f R10 %.4f\n' % (a1, a10))
torch.save({'M': M.cpu(), 'E': E.cpu()}, '/workspace/als3_lm.pt')
