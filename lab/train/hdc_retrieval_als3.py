# -*- coding: utf-8 -*-
# §23-v9: 难负例判决 —— ALS 能否超过词面？
#   同 v8 语义袋架构, 但检索候选分两档:
#     rand: 99 随机负例 (v8 口径, 词面强信号)
#     hard: 99 同对话负例 (共享话题, 逼真语义/话轮连续性)
#   词面基线在两档下都重测。ALS 加 η 衰减 (η_t = η·decay^(t-1)) 抗过拟合,
#   逐轮记录取峰值。
import torch, csv, sys, collections, time
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
dev = 'cuda'
DATA = sys.argv[6] if len(sys.argv) > 6 else '/workspace/train_split.csv'
MODE = sys.argv[7] if len(sys.argv) > 7 else 'char'

rows = list(csv.reader(open(DATA, encoding='utf-8-sig')))
hdr = rows[0]; ui = hdr.index('Utterance'); di = hdr.index('Dialogue_ID')
dialogs = collections.OrderedDict()
for r in rows[1:]:
    dialogs.setdefault(r[di], []).append(r[ui])

TOK = (MODE == 'tok')
if TOK:
    cc = collections.Counter(w for us in dialogs.values() for u in us for w in u.split())
else:
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
        if TOK:
            ctx_u = (' '.join(us[:t])).split()
            ctx = ' '.join(ctx_u[-64:])
            rep = ' '.join(us[t].split()[:32])
        else:
            ctx = ''.join(us[:t])[-64:]
            rep = us[t][:32]
        if ctx and rep:
            pairs.append((ctx, rep, k, k in tr_keys))
tr = [(a, b, k) for a, b, k, istr in pairs if istr]
te = [(a, b, k) for a, b, k, istr in pairs if not istr]
print('对话 %d, 句对 训练 %d 测试 %d' % (len(keys), len(tr), len(te)))

K1, K2, D = 64, 32, 1024
E = torch.zeros(VOC+2, D, device=dev)
E[:UNK] = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (UNK, D)).astype(np.float32)).to(dev)
BW1 = torch.linspace(0.5, 2.0, K1, device=dev)

def ids_of(s, L):
    us = s.split() if TOK else s
    ids = [vmap.get(c, UNK) for c in us][-L:]
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

rng = np.random.default_rng(11)
P = len(te)
dlg_te = defaultdict(list)
for i, (a, b, k) in enumerate(te):
    dlg_te[k].append(i)

def build_cand(hard):
    cand = np.zeros((P, 100), dtype=np.int64)
    for i in range(P):
        cand[i, 0] = i
        if not hard:
            neg = rng.choice(P-1, 99, replace=False)
            neg = np.where(neg >= i, neg+1, neg)
        else:
            pool = np.array([j for j in dlg_te[te[i][2]] if j != i], dtype=np.int64)
            if len(pool) == 0:
                neg = rng.choice(P-1, 99, replace=False)
                neg = np.where(neg >= i, neg+1, neg)
            elif len(pool) >= 99:
                neg = rng.choice(pool, 99, replace=False)
            else:
                negset = list(dict.fromkeys(pool.tolist()))
                guard = 0
                while len(negset) < 99 and guard < 10:
                    extra = rng.choice(np.delete(np.arange(P), i), 99, replace=False)
                    negset = list(dict.fromkeys(negset + extra.tolist()))
                    guard += 1
                neg = np.array(negset[:99])
        cand[i, 1:] = neg
    return torch.from_numpy(cand).to(dev)

cand_rand = build_cand(False)
cand_hard = build_cand(True)
te_ctx = torch.tensor([ids_of(x[0], K1) for x in te], dtype=torch.int64, device=dev)
te_rep = torch.tensor([ids_of(x[1], K2) for x in te], dtype=torch.int64, device=dev)
Hte = enc_h(te_ctx); Gte = enc_g(te_rep)

def eval_score(M, cand_t):
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

def lexical(cand_np):
    sub = rng.choice(P, 3000, replace=False)
    Gte_np = [set(te[i][1]) for i in range(P)]
    l1 = l10 = 0
    for i in sub:
        cs = set(te[i][0])
        sc = [len(cs & Gte_np[c]) for c in cand_np[i]]
        order = np.argsort(sc)[::-1]
        l1 += int(order[0] == 0)
        l10 += int(0 in order[:10])
    return l1/3000, l10/3000

print('词面重叠 rand: R@1 %.3f R@10 %.3f' % lexical(cand_rand.cpu().numpy()))
print('词面重叠 hard: R@1 %.3f R@10 %.3f' % lexical(cand_hard.cpu().numpy()))

lam = 1e-2
lam_e = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
ETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
NR = int(sys.argv[3]) if len(sys.argv) > 3 else 5
DEC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
MU = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
print('λe=%.0f η=%.1f 衰减=%.2f 轮数=%d μ=%.1f' % (lam_e, ETA, DEC, NR, MU))
DLG = {}
dl = sorted(set(k for a, b, k in tr))
dlid = {k: i for i, k in enumerate(dl)}
DLG = torch.tensor([dlid[k] for a, b, k in tr], dtype=torch.int64, device=dev)
Gmean_d = torch.zeros(len(dl), D, device=dev)
cntd = torch.zeros(len(dl), device=dev)
Gmean_d.index_add_(0, DLG, G)
cntd.index_add_(0, DLG, torch.ones(len(tr), device=dev))
Gmean_d = Gmean_d/cntd.clamp_min(1)[:, None]
I_D = torch.eye(D, device=dev)

NCI_CTX = torch.zeros(VOC+2, K1, device=dev)
for i in range(K1):
    NCI_CTX[:, i] = torch.bincount(CTX_T[:, i], minlength=VOC+2).to(torch.float32)
NCI_CTX[PAD] = 0
s2 = torch.bincount(REP_T.reshape(-1), minlength=VOC+2).to(torch.float32)
s2[PAD] = 0
s1 = torch.einsum('ci,i->c', NCI_CTX, BW1**2)

M = torch.eye(D, device=dev)
eta = ETA
best = (0.0, 0.0, 0)
print('=== §23-v9: 难负例判决 (零 SGD) ===')
for rnd in range(1, NR+1):
    t0 = time.time()
    Gm = G - MU*Gmean_d[DLG] if MU > 0 else G
    M = torch.linalg.solve((H.T@H)/len(tr) + lam*I_D, (H.T@Gm)/len(tr))
    a1r, a10r = eval_score(M, cand_rand)
    a1h, a10h = eval_score(M, cand_hard)
    print('轮%d M-解 (%.0fs): rand R@1 %.3f R@10 %.3f | hard R@1 %.3f R@10 %.3f'
          % (rnd, time.time()-t0, a1r, a10r, a1h, a10h))
    open('/workspace/lm_result.txt', 'a').write(
        'RESULT-v9 rnd%d rand %.4f/%.4f hard %.4f/%.4f\n' % (rnd, a1r, a10r, a1h, a10h))
    if a1h > best[0]:
        best = (a1h, a10h, rnd)
    t1 = time.time()
    Gmat = M@M.T
    rhs = torch.zeros(VOC+2, D, device=dev)
    MG = Gm@M.T
    for i in range(K1):
        rhs.index_add_(0, CTX_T[:, i], MG*BW1[i])
    GH = H@M
    for j in range(K2):
        rhs.index_add_(0, REP_T[:, j], Gm - GH)
    Eb = E.clone()
    B = 502
    for s in range(0, VOC+2, B):
        Hs = s1[s:s+B, None, None]*Gmat[None] + (s2[s:s+B]+lam_e)[:, None, None]*I_D[None]
        Eb[s:s+B] = torch.linalg.solve(Hs, rhs[s:s+B, :, None])[..., 0]
    E += eta*(Eb - E)
    E[PAD] = 0
    H = enc_h(CTX_T); G = enc_g(REP_T)
    print('  e-解 (%.0fs) η=%.2f' % (time.time()-t1, eta))
    eta *= DEC

a1r, a10r = eval_score(M, cand_rand)
a1h, a10h = eval_score(M, cand_hard)
print('\nv9 末轮: rand %.3f/%.3f hard %.3f/%.3f' % (a1r, a10r, a1h, a10h))
print('v9 hard 峰值: R@1 %.3f R@10 %.3f @轮%d' % best)
open('/workspace/lm_result.txt', 'a').write(
    'RESULT-v9 final hard-peak R1 %.4f R10 %.4f @rnd%d\n' % best)
torch.save({'M': M.cpu(), 'E': E.cpu()}, '/workspace/als5_lm.pt')
