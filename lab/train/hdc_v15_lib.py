# -*- coding: utf-8 -*-
# §23-v15 库: LCCC 流式 + 层次塔数据 (扁平64词元新近袋 + 8句x24词元话语袋)
import torch, csv, sys, collections, time, gzip, hashlib
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
dev = 'cuda'
DATA = sys.argv[1] if len(sys.argv) > 1 else '/workspace/lccc3m.csv.gz'
TE_CAP = int(sys.argv[6]) if len(sys.argv) > 6 else 150000

def op(p):
    return gzip.open(p, 'rt', encoding='utf-8', errors='replace') if p.endswith('.gz') \
        else open(p, 'r', encoding='utf-8', errors='replace')

def dlg_train(k):
    return int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 100 < 85

t0 = time.time()
cc = collections.Counter()
ntr = nte = 0
cur = None; us = []
def flush1(k, us):
    global ntr, nte
    if k is None or len(us) < 2:
        return
    for u in us:
        cc.update(u.split())
    n = len(us) - 1
    if dlg_train(k):
        ntr += n
    else:
        nte += n
for r in csv.reader(op(DATA)):
    if r[0] != cur:
        flush1(cur, us)
        cur = r[0]; us = []
    us.append(r[1])
flush1(cur, us)
VOC = 3000
vocab = [c for c, _ in cc.most_common(VOC)]
vmap = {c: i for i, c in enumerate(vocab)}
UNK = len(vocab); PAD = UNK + 1
print('词表 %d, 句对 训练 %d 测试 %d (%.0fs)' % (UNK, ntr, nte, time.time()-t0))
K1, K2, D = 64, 32, 1024
KU, TU = 8, 24                                          # 层次塔: 8 句 x 24 词元
E0 = torch.from_numpy(np.random.default_rng(7).normal(
    0, 1.0/np.sqrt(D), (UNK, D)).astype(np.float32)).to(dev)
E_ctx = torch.zeros(VOC+2, D, device=dev); E_ctx[:UNK] = E0
E_rep = torch.zeros(VOC+2, D, device=dev); E_rep[:UNK] = E0
BW1 = torch.linspace(0.5, 2.0, K1, device=dev)
AH = torch.linspace(0.5, 2.0, KU, device=dev)           # 话语新近权 (槽7=最新=2.0)

def ids_l(ws, L):
    ids = [vmap.get(w, UNK) for w in ws][-L:]
    return [PAD]*(L-len(ids)) + ids

t0 = time.time()
CTX = np.zeros((ntr, K1), dtype=np.int16)
CTU = np.zeros((ntr, KU, TU), dtype=np.int16)
REP = np.zeros((ntr, K2), dtype=np.int16)
te_ctx = np.zeros((min(nte, TE_CAP), K1), dtype=np.int16)
te_ctu = np.zeros((min(nte, TE_CAP), KU, TU), dtype=np.int16)
te_rep = np.zeros((min(nte, TE_CAP), K2), dtype=np.int16)
te_strs = []
p_tr = p_te = 0
cur = None; us = []
def flush2(k, us):
    global p_tr, p_te
    if k is None or len(us) < 2:
        return
    tr_d = dlg_train(k)
    toks = []
    uts = []
    for t, u in enumerate(us):
        if t >= 1:
            rw = us[t].split()[:K2]
            if toks and rw:
                if tr_d:
                    CTX[p_tr] = ids_l(toks[-K1:], K1)
                    hu = uts[-KU:]
                    for ki, ut in enumerate(hu):
                        CTU[p_tr, KU-len(hu)+ki] = ids_l(ut[:TU], TU)
                    REP[p_tr] = ids_l(rw, K2); p_tr += 1
                elif p_te < TE_CAP:
                    te_ctx[p_te] = ids_l(toks[-K1:], K1)
                    hu = uts[-KU:]
                    for ki, ut in enumerate(hu):
                        te_ctu[p_te, KU-len(hu)+ki] = ids_l(ut[:TU], TU)
                    te_rep[p_te] = ids_l(rw, K2)
                    te_strs.append((' '.join(toks[-K1:]), ' '.join(rw), k)); p_te += 1
        toks.extend(u.split())
        uts.append(u.split()[:TU])
for r in csv.reader(op(DATA)):
    if r[0] != cur:
        flush2(cur, us)
        cur = r[0]; us = []
    us.append(r[1])
flush2(cur, us)
CTX_T = torch.from_numpy(CTX[:p_tr]).to(dev)
CTX_U = torch.from_numpy(CTU[:p_tr]).to(dev)
REP_T = torch.from_numpy(REP[:p_tr]).to(dev)
te_ctx_t = torch.from_numpy(te_ctx[:p_te]).to(dev)
te_ctu_t = torch.from_numpy(te_ctu[:p_te]).to(dev)
te_rep_t = torch.from_numpy(te_rep[:p_te]).to(dev)
te = te_strs
P = p_te
print('填充完成: 训练 %d 测试 %d (%.0fs)' % (p_tr, p_te, time.time()-t0))
rng = np.random.default_rng(11)
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
def enc_h(ids_f, ids_u):
    h = torch.zeros(ids_f.shape[0], D, device=dev)
    for i in range(K1):
        h += E_ctx[ids_f[:, i].long()]*BW1[i]
    for k in range(KU):
        ids_k = ids_u[:, k].long()
        for t in range(TU):
            h += E_ctx[ids_k[:, t]]*AH[k]
    return h

def enc_g(ids_r):
    g = torch.zeros(ids_r.shape[0], D, device=dev)
    for j in range(K2):
        g += E_rep[ids_r[:, j].long()]
    return g

Hte = torch.zeros(P, D, device=dev)
Gte = torch.zeros(P, D, device=dev)
for s in range(0, P, 20000):
    e = min(s+20000, P)
    Hte[s:e] = enc_h(te_ctx_t[s:e], te_ctu_t[s:e])
    Gte[s:e] = enc_g(te_rep_t[s:e])

def eval_score(M, cand_t):
    Hn = Hte/Hte.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Gn = Gte/Gte.norm(dim=1, keepdim=True).clamp_min(1e-9)
    r1 = r10 = 0
    for s in range(0, P, 2048):
        e = min(s+2048, P)
        gh = (Hn[s:e]@M).clamp(-8, 8)
        gh = gh/gh.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sc = torch.einsum('bd,bcd->bc', gh, Gn[cand_t[s:e]])
        r1 += int((sc.argmax(1) == 0).sum())
        r10 += int((sc.topk(10, 1).indices == 0).any(1).sum())
    return r1/P, r10/P

def lexical(cand_np):
    sub = rng.choice(P, 3000, replace=False)
    Gte_np = [set(te[i][1].split()) for i in range(P)]
    l1 = l10 = 0
    for i in sub:
        cs = set(te[i][0].split())
        sc = [len(cs & Gte_np[c]) for c in cand_np[i]]
        order = np.argsort(sc)[::-1]
        l1 += int(order[0] == 0)
        l10 += int(0 in order[:10])
    return l1/3000, l10/3000

s1 = torch.zeros(VOC+2, device=dev)
s2 = torch.zeros(VOC+2, device=dev)
for s in range(0, p_tr, 20000):
    e = min(s+20000, p_tr)
    B = e-s
    Wacc = torch.zeros(B, VOC+2, device=dev)
    idf = CTX_T[s:e].long()
    for i in range(K1):
        Wacc.scatter_add_(1, idf[:, i, None], BW1[i].expand(B, 1))
    idu = CTX_U[s:e].long()
    for k in range(KU):
        for t in range(TU):
            Wacc.scatter_add_(1, idu[:, k, t, None], AH[k].expand(B, 1))
    s1 += (Wacc**2).sum(0)
    idr = REP_T[s:e].long()
    Cnt = torch.zeros(B, VOC+2, device=dev)
    for j in range(K2):
        Cnt.scatter_add_(1, idr[:, j, None], torch.ones(B, 1, device=dev))
    s2 += (Cnt**2).sum(0)
s1[PAD] = 0; s1[UNK] = 0
s2[PAD] = 0; s2[UNK] = 0
