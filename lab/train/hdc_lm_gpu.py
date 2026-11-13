# -*- coding: utf-8 -*-
# §23 GPU 版: 容器语言模型第一块基石 —— next-char (检索+闭式解, 零梯度迭代)
#   上下文: 前 8 字 → HDC 序列编码 (字向量 × 位置绑定算子, 求和)
#   训练:   一次岭回归闭式解 (Gram 累计) —— 权重路线的训练在容器里=代数解
#   判据:   top-1/top-3 vs unigram/bigram/trigram, Dialogue_ID 级 split
import torch, csv, sys, collections, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')

# ---- 语料 ----
rows = list(csv.reader(open('/workspace/train_split.csv', encoding='utf-8-sig')))
hdr = rows[0]; ui = hdr.index('Utterance'); di = hdr.index('Dialogue_ID')
dialogs = collections.OrderedDict()
for r in rows[1:]:
    dialogs.setdefault(r[di], []).append(r[ui])
texts = [''.join(us) for us in dialogs.values()]
print('对话数 %d, 总字数 %d' % (len(texts), sum(len(t) for t in texts)))

cc = collections.Counter(''.join(texts))
VOC = 3000
vocab = [c for c, _ in cc.most_common(VOC)]
vmap = {c: i for i, c in enumerate(vocab)}
UNK = len(vocab)
print('字频覆盖 %.3f' % (sum(cc[c] for c in vocab)/sum(cc.values())))

nprng = np.random.default_rng(7)
perm = nprng.permutation(len(texts))
n_tr = int(len(texts)*0.85)
tr_idx = set(perm[:n_tr].tolist())

K, D = 8, 1024
E = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (VOC+1, D)).astype(np.float32)).to(dev)
PB = torch.from_numpy(nprng.normal(0, 1, (K, D)).astype(np.float32)).to(dev)  # 位置绑定算子

def make_windows(tlist, maxn=None, seed=3):
    W, T = [], []
    for ti, t in enumerate(tlist):
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
print('训练窗口 %d, 测试窗口 %d' % (len(Ttr), len(Tte)))

NF = D
def encode_gpu2(W):
    Wi = torch.from_numpy(W).to(dev)
    ctx = torch.zeros(Wi.shape[0], D, device=dev)
    for i in range(K):
        ctx += E[Wi[:, i]] * PB[i]
    return ctx

# ---- 闭式解: 分块 Gram 累计 (GPU matmul) ----
t0 = time.time()
ATA = torch.zeros(NF, NF, device=dev)
ATY = torch.zeros(NF, VOC, device=dev)
BLK = 50000
for s in range(0, len(Ttr), BLK):
    Xb = encode_gpu2(Wtr[s:s+BLK])
    Tb = torch.from_numpy(Ttr[s:s+BLK]).to(dev)
    Yoh = torch.zeros(Xb.shape[0], VOC, device=dev)
    Yoh[torch.arange(Xb.shape[0], device=dev), Tb] = 1.0
    ATA += Xb.T@Xb
    ATY += Xb.T@Yoh
ATA /= len(Ttr); ATY /= len(Ttr)
lam = 1e-3
vo = torch.linalg.solve(ATA + lam*torch.eye(NF, device=dev), ATY)
t_solve = time.time()-t0
print('闭式解完成: %.1fs (含 Gram 累计; 梯度迭代次数=0)' % t_solve)

# ---- 评估 ----
def eval_topk(W, T, vo, B=100000):
    top1 = top3 = tot = 0
    for s in range(0, len(T), B):
        X = encode_gpu2(W[s:s+B])
        Tb = torch.from_numpy(T[s:s+B]).to(dev)
        sc = X@vo
        top1 += int((sc.argmax(1) == Tb).sum())
        top3 += int((sc.topk(3, 1).indices == Tb[:, None]).any(1).sum())
        tot += len(Tb)
    return top1/tot, top3/tot

a1, a3 = eval_topk(Wte, Tte, vo)
uni = np.bincount(Ttr, minlength=VOC)/len(Ttr)
uni_acc = float(np.mean(Tte == int(np.argmax(uni))))
bi = collections.defaultdict(collections.Counter)
for p, t in zip(Wtr[:, -1], Ttr): bi[p][t] += 1
bi_pred = [bi[p].most_common(1)[0][0] if p in bi else int(np.argmax(uni)) for p in Wte[:, -1]]
bi_acc = float(np.mean(np.array(bi_pred) == Tte))
tri = collections.defaultdict(collections.Counter)
for p, t in zip(map(tuple, Wtr[:, -2:]), Ttr): tri[p][t] += 1
tri_pred = [tri[p2].most_common(1)[0][0] if p2 in tri else bi_pred[i]
            for i, p2 in enumerate(map(tuple, Wte[:, -2:]))]
tri_acc = float(np.mean(np.array(tri_pred) == Tte))

print('\n=== §23: 容器 LM (HDC 8 字上下文 + 一次闭式解) ===')
print('unigram:  top1 %.3f' % uni_acc)
print('bigram:   top1 %.3f' % bi_acc)
print('trigram:  top1 %.3f' % tri_acc)
print('容器 LM:  top1 %.3f  top3 %.3f  (训练=%.1fs 闭式解)' % (a1, a3, t_solve))
res = 'RESULT unigram %.4f bigram %.4f trigram %.4f container %.4f top3 %.4f solve %.1fs\n' % (
    uni_acc, bi_acc, tri_acc, a1, a3, t_solve)
open('/workspace/lm_result.txt', 'w').write(res)
print(res)
