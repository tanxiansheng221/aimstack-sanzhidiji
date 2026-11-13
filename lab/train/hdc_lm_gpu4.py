# -*- coding: utf-8 -*-
# §23-v4: 大哈希桶 + 全量窗口 + 生成演示 (GPU)
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
HB = 16000            # 哈希桶 (v3 的 5.3 倍, 碰撞大降)
E = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (VOC+1, D)).astype(np.float32)).to(dev)
PW = torch.tensor([0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 2.0, 4.0], device=dev)[:, None] * \
     torch.from_numpy(nprng.normal(0, 1, (K, D)).astype(np.float32)).to(dev)

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

Wtr, Ttr = make_windows([texts[i] for i in tr_idx])
Wte, Tte = make_windows([texts[i] for i in perm[n_tr:]], maxn=60000, seed=4)
print('训练 %d 测试 %d' % (len(Ttr), len(Tte)))

NF = 3000 + HB + HB + D
OH, HP, HT, HC = 0, 3000, 3000+HB, 3000+2*HB
def encode(W, s, e, use_hdc=True):
    Wi = torch.from_numpy(W[s:e]).to(dev)
    n = Wi.shape[0]
    X = torch.zeros(n, NF, device=dev)
    ar = torch.arange(n, device=dev)
    X[ar, OH + Wi[:, -1]] = 1.0
    X[ar, HP + ((Wi[:, -2]*31 + Wi[:, -1]) % HB)] += 1.0
    X[ar, HT + ((Wi[:, -3]*961 + Wi[:, -2]*31 + Wi[:, -1]) % HB)] += 1.0
    if use_hdc:
        ctx = torch.zeros(n, D, device=dev)
        for i in range(K):
            ctx += E[Wi[:, i]]*PW[i]
        X[:, HC:] = ctx
    return X

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
print('Gram 累计 %.1fs, ATA %.1f GB' % (time.time()-t0, NF*NF*4/1e9))

for lam in (1e-4,):
    vo = torch.linalg.solve(ATA + lam*torch.eye(NF, device=dev), ATY)
    top1 = top3 = tot = 0
    for s in range(0, len(Tte), 50000):
        X = encode(Wte, s, min(s+50000, len(Tte)))
        Tb = torch.from_numpy(Tte[s:s+50000]).to(dev)
        sc = X@vo
        top1 += int((sc.argmax(1) == Tb).sum())
        top3 += int((sc.topk(3, 1).indices == Tb[:, None]).any(1).sum())
        tot += len(Tb)
    print('v4 λ=%.0e: top1 %.3f  top3 %.3f  (对照 trigram 0.232)' % (lam, top1/tot, top3/tot))
    open('/workspace/lm_result.txt', 'a').write('RESULT-v4 lam %.0e top1 %.4f top3 %.4f\n' % (lam, top1/tot, top3/tot))

# ---- 生成演示 (贪心+温度采样) ----
print('\n=== 生成演示 (从训练文本随机起点, 贪心 120 字) ===')
iv = {i: c for c, i in vmap.items()}
def gen(prompt_ids, n=120, temp=0.0):
    ids = list(prompt_ids)
    out = []
    for _ in range(n):
        w = np.array([ids[-K:]], dtype=np.int64)
        sc = encode(w, 0, 1)@vo
        if temp <= 0:
            nxt = int(sc.argmax())
        else:
            p = torch.softmax(sc[0]/temp, 0).cpu().numpy()
            nxt = int(np.random.default_rng().choice(VOC, p=p))
        ids.append(nxt); out.append(iv.get(nxt, ''))
    return ''.join(out)

for s in (100, 5000, 20000):
    seed_txt = texts[perm[n_tr:][s % len(perm[n_tr:])]]
    seed_ids = [vmap.get(c, UNK) for c in seed_txt[:K]]
    print('种子: %s' % seed_txt[:K])
    print('生成: %s' % gen(seed_ids))
