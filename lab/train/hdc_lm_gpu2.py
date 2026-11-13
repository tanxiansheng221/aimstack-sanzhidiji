# -*- coding: utf-8 -*-
# §23-v2: 修稀释 —— 位置加权上下文 + 末3字直连 + λ 扫描 (GPU 迭代)
import torch, csv, sys, collections, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
dev = 'cuda'

rows = list(csv.reader(open('/workspace/train_split.csv', encoding='utf-8-sig')))
hdr = rows[0]; ui = hdr.index('Utterance'); di = hdr.index('Dialogue_ID')
dialogs = collections.OrderedDict()
for r in rows[1:]:
    dialogs.setdefault(r[di], []).append(r[ui])
texts = [''.join(us) for us in dialogs.values()]

cc = collections.Counter(''.join(texts))
VOC = 3000
vocab = [c for c, _ in cc.most_common(VOC)]
vmap = {c: i for i, c in enumerate(vocab)}
UNK = len(vocab)

nprng = np.random.default_rng(7)
perm = nprng.permutation(len(texts))
n_tr = int(len(texts)*0.85)
tr_idx = set(perm[:n_tr].tolist())

K, D = 8, 1024
E = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (VOC+1, D)).astype(np.float32)).to(dev)
PW = torch.tensor([0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 2.0, 4.0], device=dev)  # 越近越重
PD = torch.from_numpy(nprng.normal(0, 1, (K, D)).astype(np.float32)).to(dev) # 位置绑定
PW = PW[:, None]*PD  # 位置i的绑定算子带权

def make_windows(tlist, maxn=None, seed=3):
    W, T = [], []
    for t in tlist:
        ids = [vmap.get(c, UNK) for c in t]
        for i in range(K, len(ids)):
            if ids[i] != UNK:
                W.append(ids[i-K:i]); T.append(ids[i])
    W = np.array(W, dtype=np.int64); T = np.array(T, dtype=np.int64)
    if maxn and len(T) > maxn:
        sel = np.random.default_rng(seed).choice(len(T), maxn, replace=False)
        W, T = W[sel], T[sel]
    return W, T

Wtr, Ttr = make_windows([texts[i] for i in tr_idx], maxn=400000)
Wte, Tte = make_windows([texts[i] for i in perm[n_tr:]], maxn=60000, seed=4)
print('训练 %d 测试 %d' % (len(Ttr), len(Tte)))

NF = 4*D   # [加权上下文 D | 末字 D | 次末字 D | 三末字 D]
def encode(W, s, e):
    Wi = torch.from_numpy(W[s:e]).to(dev)
    ctx = torch.zeros(Wi.shape[0], D, device=dev)
    for i in range(K):
        ctx += (E[Wi[:, i]]*PW[i])
    return torch.cat([ctx, 4.0*E[Wi[:, -1]], 2.0*E[Wi[:, -2]], 1.0*E[Wi[:, -3]]], 1)

t0 = time.time()
ATA = torch.zeros(NF, NF, device=dev)
ATY = torch.zeros(NF, VOC, device=dev)
BLK = 50000
for s in range(0, len(Ttr), BLK):
    Xb = encode(Wtr, s, min(s+BLK, len(Ttr)))
    Tb = torch.from_numpy(Ttr[s:s+BLK]).to(dev)
    Yoh = torch.zeros(Xb.shape[0], VOC, device=dev)
    Yoh[torch.arange(Xb.shape[0], device=dev), Tb] = 1.0
    ATA += Xb.T@Xb
    ATY += Xb.T@Yoh
ATA /= len(Ttr); ATY /= len(Ttr)
print('Gram 累计 %.1fs' % (time.time()-t0))

def eval_vo(vo):
    top1 = top3 = tot = 0
    for s in range(0, len(Tte), 50000):
        X = encode(Wte, s, min(s+50000, len(Tte)))
        Tb = torch.from_numpy(Tte[s:s+50000]).to(dev)
        sc = X@vo
        top1 += int((sc.argmax(1) == Tb).sum())
        top3 += int((sc.topk(3, 1).indices == Tb[:, None]).any(1).sum())
        tot += len(Tb)
    return top1/tot, top3/tot

print('\n=== §23-v2: 加权上下文+末3字直连, λ 扫描 ===')
for lam in (1e-2, 1e-3, 1e-4):
    vo = torch.linalg.solve(ATA + lam*torch.eye(NF, device=dev), ATY)
    a1, a3 = eval_vo(vo)
    print('λ=%.0e: 容器 LM top1 %.3f  top3 %.3f  (对照 trigram 0.232)' % (lam, a1, a3))
    open('/workspace/lm_result.txt', 'a').write(
        'RESULT-v2 lam %.0e top1 %.4f top3 %.4f\n' % (lam, a1, a3))
