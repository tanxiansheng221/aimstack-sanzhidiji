# -*- coding: utf-8 -*-
# §23-v3: 哈希 n-gram(精确容量) + HDC 长上下文 —— 全闭式解, 零梯度
#   特征: [one-hot 末字 3000 | 哈希pair 3000 | 哈希triple 3000 | HDC加权ctx 1024] = 10024
#   对照: 纯哈希n-gram(无HDC ctx) —— 隔离 HDC 长上下文的净贡献
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
PW = torch.tensor([0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 2.0, 4.0], device=dev)[:, None] * \
     torch.from_numpy(nprng.normal(0, 1, (K, D)).astype(np.float32)).to(dev)
PH = torch.from_numpy(nprng.uniform(0, VOC, (K, D)).astype(np.int64)).to(dev)  # 哈希用随机相位表

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

# 哈希特征: pair/triple → [0, 3000) 桶 (多重哈希降碰撞: idx=(a*31+b)*33+c % 3000 式)
NF = 3000 + 3000 + 3000 + D
OH, HP, HT, HC = 0, 3000, 6000, 9000
def encode(W, s, e, use_hdc=True):
    Wi = torch.from_numpy(W[s:e]).to(dev)
    n = Wi.shape[0]
    X = torch.zeros(n, NF, device=dev)
    X[torch.arange(n, device=dev), OH + Wi[:, -1]] = 1.0            # one-hot 末字
    X[torch.arange(n, device=dev), HP + ((Wi[:, -2]*31 + Wi[:, -1]) % VOC)] += 1.0
    X[torch.arange(n, device=dev), HT + ((Wi[:, -3]*961 + Wi[:, -2]*31 + Wi[:, -1]) % VOC)] += 1.0
    if use_hdc:
        ctx = torch.zeros(n, D, device=dev)
        for i in range(K):
            ctx += E[Wi[:, i]]*PW[i]
        X[:, HC:] = ctx
    return X

t0 = time.time()
BLK = 50000
def accumulate(use_hdc):
    ATA = torch.zeros(NF, NF, device=dev)
    ATY = torch.zeros(NF, VOC, device=dev)
    for s in range(0, len(Ttr), BLK):
        Xb = encode(Wtr, s, min(s+BLK, len(Ttr)), use_hdc)
        Tb = torch.from_numpy(Ttr[s:s+BLK]).to(dev)
        Yoh = torch.zeros(Xb.shape[0], VOC, device=dev)
        Yoh[torch.arange(Xb.shape[0], device=dev), Tb] = 1.0
        ATA += Xb.T@Xb
        ATY += Xb.T@Yoh
    return ATA/len(Ttr), ATY/len(Ttr)

def eval_vo(vo, use_hdc=True):
    top1 = top3 = tot = 0
    for s in range(0, len(Tte), 50000):
        X = encode(Wte, s, min(s+50000, len(Tte)), use_hdc)
        Tb = torch.from_numpy(Tte[s:s+50000]).to(dev)
        sc = X@vo
        top1 += int((sc.argmax(1) == Tb).sum())
        top3 += int((sc.topk(3, 1).indices == Tb[:, None]).any(1).sum())
        tot += len(Tb)
    return top1/tot, top3/tot

print('=== §23-v3: 哈希n-gram + HDC长上下文 (全闭式解) ===')
print('--- 对照臂 A: 纯哈希 n-gram (无 HDC ctx) ---')
ATA, ATY = accumulate(use_hdc=False)
for lam in (1e-3, 1e-4):
    vo = torch.linalg.solve(ATA + lam*torch.eye(NF, device=dev), ATY)
    a1, a3 = eval_vo(vo, use_hdc=False)
    print('  λ=%.0e: top1 %.3f  top3 %.3f' % (lam, a1, a3))
    open('/workspace/lm_result.txt', 'a').write('RESULT-v3A lam %.0e top1 %.4f top3 %.4f\n' % (lam, a1, a3))

print('--- 主臂 B: 哈希 n-gram + HDC 8 字上下文 ---')
t0 = time.time()
ATA, ATY = accumulate(use_hdc=True)
print('Gram %.1fs' % (time.time()-t0))
for lam in (1e-3, 1e-4):
    vo = torch.linalg.solve(ATA + lam*torch.eye(NF, device=dev), ATY)
    a1, a3 = eval_vo(vo, use_hdc=True)
    print('  λ=%.0e: top1 %.3f  top3 %.3f' % (lam, a1, a3))
    open('/workspace/lm_result.txt', 'a').write('RESULT-v3B lam %.0e top1 %.4f top3 %.4f\n' % (lam, a1, a3))
print('\n对照: bigram 0.193 / trigram 0.232 / v2 0.201')
