# -*- coding: utf-8 -*-
# §23 CPU 版: 容器 LM next-char (HDC 8字上下文 + 一次闭式解), numpy/128核
import numpy as np, csv, sys, collections, time, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
os.environ.setdefault('OMP_NUM_THREADS', '96')

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
E = nprng.normal(0, 1.0/np.sqrt(D), (VOC+1, D)).astype(np.float32)
PB = nprng.normal(0, 1, (K, D)).astype(np.float32)

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
print('训练窗口 %d, 测试窗口 %d' % (len(Ttr), len(Tte)))

def encode_np(W):
    n = W.shape[0]
    ctx = np.zeros((n, D), dtype=np.float32)
    for i in range(K):
        ctx += E[W[:, i]] * PB[i]
    return ctx

t0 = time.time()
ATA = np.zeros((D, D), dtype=np.float64)
ATY = np.zeros((D, VOC), dtype=np.float64)
BLK = 50000
for s in range(0, len(Ttr), BLK):
    Xb = encode_np(Wtr[s:s+BLK])
    Yoh = np.zeros((len(Xb), VOC), dtype=np.float32)
    Yoh[np.arange(len(Xb)), Ttr[s:s+BLK]] = 1.0
    ATA += Xb.T.astype(np.float64)@Xb
    ATY += Xb.T.astype(np.float64)@Yoh
ATA /= len(Ttr); ATY /= len(Ttr)
vo = np.linalg.solve(ATA + 1e-3*np.eye(D), ATY)
t_solve = time.time()-t0
print('闭式解完成: %.1fs (梯度迭代次数=0)' % t_solve)

def eval_topk(W, T, B=50000):
    top1 = top3 = tot = 0
    for s in range(0, len(T), B):
        sc = encode_np(W[s:s+B]) @ vo
        Tb = T[s:s+B]
        top1 += int(np.sum(np.argmax(sc, 1) == Tb))
        top3 += int(np.sum(np.any(np.argsort(-sc, 1)[:, :3] == Tb[:, None], 1)))
        tot += len(Tb)
    return top1/tot, top3/tot

a1, a3 = eval_topk(Wte, Tte)
uni = np.bincount(Ttr, minlength=VOC)/len(Ttr)
uni_acc = float(np.mean(Tte == int(np.argmax(uni))))
bi = collections.defaultdict(collections.Counter)
for p, t in zip(Wtr[:, -1], Ttr): bi[p][t] += 1
bi_pred = np.array([bi[p].most_common(1)[0][0] if p in bi else int(np.argmax(uni)) for p in Wte[:, -1]])
bi_acc = float(np.mean(bi_pred == Tte))
tri = collections.defaultdict(collections.Counter)
for p, t in zip(map(tuple, Wtr[:, -2:]), Ttr): tri[p][t] += 1
tri_pred = np.array([tri[p2].most_common(1)[0][0] if p2 in tri else bi_pred[i]
                     for i, p2 in enumerate(map(tuple, Wte[:, -2:]))])
tri_acc = float(np.mean(tri_pred == Tte))

print('\n=== §23: 容器 LM (HDC 8 字上下文 + 一次闭式解, CPU/128核) ===')
print('unigram:  top1 %.3f' % uni_acc)
print('bigram:   top1 %.3f' % bi_acc)
print('trigram:  top1 %.3f' % tri_acc)
print('容器 LM:  top1 %.3f  top3 %.3f  (solve %.1fs)' % (a1, a3, t_solve))
res = 'RESULT-CPU unigram %.4f bigram %.4f trigram %.4f container %.4f top3 %.4f solve %.1fs\n' % (
    uni_acc, bi_acc, tri_acc, a1, a3, t_solve)
open('/workspace/lm_result.txt', 'a').write(res)
print(res)
