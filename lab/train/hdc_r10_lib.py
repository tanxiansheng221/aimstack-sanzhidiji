# -*- coding: utf-8 -*-
# §23-v10 库: 数据/编码/候选/评测 (n-gram 袋塔 + 语义袋)
import torch, csv, sys, collections
import numpy as np
from collections import defaultdict
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
UNK = len(vocab); PAD = UNK + 1
nprng = np.random.default_rng(7)
keys = list(dialogs.keys())
perm = nprng.permutation(len(keys))
n_tr = int(len(keys)*0.85)
tr_keys = set(keys[i] for i in perm[:n_tr])
pairs = []
for k in keys:
    us = [u for u in dialogs[k] if u.strip()]
    for t in range(1, len(us)):
        ctx = ''.join(us[:t])[-64:]; rep = us[t][:32]
        if ctx and rep:
            pairs.append((ctx, rep, k, k in tr_keys))
tr = [(a, b, k) for a, b, k, i2 in pairs if i2]
te = [(a, b, k) for a, b, k, i2 in pairs if not i2]
print('对话 %d, 句对 训练 %d 测试 %d' % (len(keys), len(tr), len(te)))
K1, K2, D = 64, 32, 1024
H2I, H3I, H2O = 6000, 6000, 3000
FIN, FOUT = D + H2I + H3I, D + H2O
OFF2I, OFF3I = D, D + H2I
E = torch.zeros(VOC+2, D, device=dev)
E[:UNK] = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (UNK, D)).astype(np.float32)).to(dev)
BW1 = torch.linspace(0.5, 2.0, K1, device=dev)
def ids_of(s, L):
    ids = [vmap.get(c, UNK) for c in s][-L:]
    return [PAD]*(L-len(ids)) + ids
CTX_T = torch.tensor([ids_of(x[0], K1) for x in tr], dtype=torch.int64, device=dev)
REP_T = torch.tensor([ids_of(x[1], K2) for x in tr], dtype=torch.int64, device=dev)
CTX_S = torch.tensor([ids_of(x[0], 16) for x in tr], dtype=torch.int64, device=dev)
REP_S = torch.tensor([ids_of(x[1], 32) for x in tr], dtype=torch.int64, device=dev)
te_ctx_s = torch.tensor([ids_of(x[0], 16) for x in te], dtype=torch.int64, device=dev)
te_rep_s = torch.tensor([ids_of(x[1], 32) for x in te], dtype=torch.int64, device=dev)
def enc_h(ids_full, ids16):
    n = ids_full.shape[0]
    h = torch.zeros(n, FIN, device=dev)
    for i in range(K1):
        h[:, :D] += E[ids_full[:, i]]*BW1[i]
    ar = torch.arange(n, device=dev)
    for i in range(15):
        a, b = ids16[:, i], ids16[:, i+1]
        m = ((a != PAD) & (b != PAD)).to(torch.float32)
        h[ar, OFF2I + (a*31 + b) % H2I] += m
    for i in range(14):
        a, b, c = ids16[:, i], ids16[:, i+1], ids16[:, i+2]
        m = ((a != PAD) & (b != PAD) & (c != PAD)).to(torch.float32)
        h[ar, OFF3I + (a*961 + b*31 + c) % H3I] += m
    return h
def enc_g(ids_r):
    n = ids_r.shape[0]
    g = torch.zeros(n, FOUT, device=dev)
    for j in range(K2):
        g[:, :D] += E[ids_r[:, j]]
    ar = torch.arange(n, device=dev)
    for j in range(31):
        a, b = ids_r[:, j], ids_r[:, j+1]
        m = ((a != PAD) & (b != PAD)).to(torch.float32)
        g[ar, D + (a*31 + b) % H2O] += m
    return g
H = enc_h(CTX_T, CTX_S); G = enc_g(REP_S)
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
cand_rand = build_cand(False); cand_hard = build_cand(True)
Hte = enc_h(torch.tensor([ids_of(x[0], K1) for x in te], dtype=torch.int64, device=dev), te_ctx_s)
Gte = enc_g(te_rep_s)
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
    Gte_np = [set(te[i][1]) for i in range(P)]
    l1 = l10 = 0
    for i in sub:
        cs = set(te[i][0])
        sc = [len(cs & Gte_np[c]) for c in cand_np[i]]
        order = np.argsort(sc)[::-1]
        l1 += int(order[0] == 0)
        l10 += int(0 in order[:10])
    return l1/3000, l10/3000
NCI_CTX = torch.zeros(VOC+2, K1, device=dev)
for i in range(K1):
    NCI_CTX[:, i] = torch.bincount(CTX_T[:, i], minlength=VOC+2).to(torch.float32)
NCI_CTX[PAD] = 0
s2 = torch.bincount(REP_T.reshape(-1), minlength=VOC+2).to(torch.float32)
s2[PAD] = 0
s1 = torch.einsum('ci,i->c', NCI_CTX, BW1**2)
