# -*- coding: utf-8 -*-
# §23-v8: 语义级 ALS 修正版 —— 去绑定语义袋 (v7 教训: 独立位置绑定摧毁词面相关性)
#   h = Σ_i w_i e_{c_i}  (上文 64 字, 右对齐新近权 0.5→2.0, PAD 行恒零)
#   g = Σ_j e_{d_j}      (回复 32 字)
#   ĝ = h M (D×D 闭式岭回归); 检索 = cos(ĝ, g_cand), 1 真 + 99 随机负例
#   ALS: e_c 正规方程退化为 H_c = s1_c (M Mᵀ) + (s2_c + λe) I
#     s1_c = Σ_i n_{c,i} w_i²,  s2_c = Σ_j n'_{c,j}
#     rhs_c = Σ_{ctx} w_i ((g Mᵀ)) + Σ_{rep} (g_n − ĝ_n)
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
UNK = len(vocab)
PAD = UNK + 1

nprng = np.random.default_rng(7)
keys = list(dialogs.keys())
perm = nprng.permutation(len(keys))
n_tr = int(len(keys)*0.85)
tr_keys = set(keys[i] for i in perm[:n_tr])

pairs = []
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
BW1 = torch.linspace(0.5, 2.0, K1, device=dev)            # 位置 i: 越靠右(越新)越重

def ids_of(s, L):
    ids = [vmap.get(c, UNK) for c in s][-L:]
    return [PAD]*(L-len(ids)) + ids

CTX_T = torch.tensor([ids_of(x[0], K1) for x in tr], dtype=torch.int64, device=dev)
REP_T = torch.tensor([ids_of(x[1], K2) for x in tr], dtype=torch.int64, device=dev)

def enc_h(ids_t):
    h = torch.zeros(ids_t.shape[0], D, device=dev)
    for i in range(K1):
        h += E[ids_t[:, i]]*BW1[i]
    return h

def enc_g(ids_t):
    g = torch.zeros(ids_t.shape[0], D, device=dev)
    for j in range(K2):
        g += E[ids_t[:, j]]
    return g

H = enc_h(CTX_T); G = enc_g(REP_T)

# 测试候选 (固定, 真回复恒在槽 0)
rng = np.random.default_rng(11)
P = len(te)
cand = np.zeros((P, 100), dtype=np.int64)
for i in range(P):
    neg = rng.choice(P-1, 99, replace=False)
    neg = np.where(neg >= i, neg+1, neg)
    cand[i, 0] = i; cand[i, 1:] = neg
cand_t = torch.from_numpy(cand).to(dev)
te_ctx = torch.tensor([ids_of(x[0], K1) for x in te], dtype=torch.int64, device=dev)
te_rep = torch.tensor([ids_of(x[1], K2) for x in te], dtype=torch.int64, device=dev)
Hte = enc_h(te_ctx); Gte = enc_g(te_rep)

def eval_score(M):
    Hn = Hte/Hte.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Gn = Gte/Gte.norm(dim=1, keepdim=True).clamp_min(1e-9)
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

lam = 1e-2
lam_e = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
ETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
NR = int(sys.argv[3]) if len(sys.argv) > 3 else 3
print('λe=%.0f η=%.1f 轮数=%d' % (lam_e, ETA, NR))
I_D = torch.eye(D, device=dev)

NCI_CTX = torch.zeros(VOC+2, K1, device=dev)
for i in range(K1):
    NCI_CTX[:, i] = torch.bincount(CTX_T[:, i], minlength=VOC+2).to(torch.float32)
NCI_CTX[PAD] = 0
s2 = torch.bincount(REP_T.reshape(-1), minlength=VOC+2).to(torch.float32)
s2[PAD] = 0
s1 = torch.einsum('ci,i->c', NCI_CTX, BW1**2)             # (C,)
Wk = BW1                                                  # 位置 i 的权重 (与 enc_h 一致)

M = torch.eye(D, device=dev)
print('=== §23-v8: 语义袋检索 ALS (零 SGD) ===')
for rnd in range(1, NR+1):
    t0 = time.time()
    M = torch.linalg.solve((H.T@H)/len(tr) + lam*I_D, (H.T@G)/len(tr))
    a1, a10 = eval_score(M)
    print('轮%d M-解 (%.0fs): R@1 %.3f R@10 %.3f' % (rnd, time.time()-t0, a1, a10))
    open('/workspace/lm_result.txt', 'a').write('RESULT-v8 rnd%d R1 %.4f R10 %.4f\n' % (rnd, a1, a10))
    t1 = time.time()
    Gmat = M@M.T
    rhs = torch.zeros(VOC+2, D, device=dev)
    MG = G@M.T                                            # 每行 = M@g_n (行向量右乘 Mᵀ)
    for i in range(K1):
        rhs.index_add_(0, CTX_T[:, i], MG*Wk[i])
    GH = H@M
    for j in range(K2):
        rhs.index_add_(0, REP_T[:, j], G - GH)
    Eb = E.clone()
    B = 502
    for s in range(0, VOC+2, B):
        Hs = s1[s:s+B, None, None]*Gmat[None] + (s2[s:s+B]+lam_e)[:, None, None]*I_D[None]
        Eb[s:s+B] = torch.linalg.solve(Hs, rhs[s:s+B, :, None])[..., 0]
    E += ETA*(Eb - E)
    E[PAD] = 0
    H = enc_h(CTX_T); G = enc_g(REP_T)
    print('  e-解 (%.0fs) 完成 (η=%.1f λe=%.0f)' % (time.time()-t1, ETA, lam_e))

a1, a10 = eval_score(M)
print('\nv8 最终: R@1 %.3f R@10 %.3f  (词面 0.062/0.257, 随机 0.010/0.100)' % (a1, a10))
open('/workspace/lm_result.txt', 'a').write('RESULT-v8 final R1 %.4f R10 %.4f\n' % (a1, a10))
torch.save({'M': M.cpu(), 'E': E.cpu()}, '/workspace/als4_lm.pt')
