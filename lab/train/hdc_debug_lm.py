# -*- coding: utf-8 -*-
# §23: 容器语言模型第一块基石 —— next-char 预测 (检索+闭式解路线, 零梯度迭代)
#   上下文: 前 8 字 → HDC 序列编码 (字随机相位向量 × 位置相移绑定, Σ)
#   训练:   增量岭回归 (Gram 累计, 一次求解) —— 权重路线的「训练」在容器里=一次代数解
#   判据:   top-1/top-3 显著超 unigram/bigram 基线, Dialogue_ID 级 split 防泄漏
import numpy as np, csv, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# ---- 语料 ----
rows = list(csv.reader(open(r'H:\AimCode\lab\cpu-native\cped_corpus\train_split.csv',
                            encoding='utf-8-sig')))
hdr = rows[0]; ui = hdr.index('Utterance'); di = hdr.index('Dialogue_ID')
dialogs = collections.OrderedDict()
for r in rows[1:]:
    dialogs.setdefault(r[di], []).append(r[ui])
texts = [''.join(us) for us in dialogs.values()]
print('对话数 %d, 总字数 %d' % (len(texts), sum(len(t) for t in texts)))

# ---- 词表 ----
cc = collections.Counter(''.join(texts))
VOC = 3000
vocab = [c for c, _ in cc.most_common(VOC)]
vmap = {c: i for i, c in enumerate(vocab)}
UNK = len(vocab)
print('覆盖字频 %.3f' % (sum(cc[c] for c in vocab)/sum(cc.values())))

# ---- 对话级 split ----
rng = np.random.default_rng(7)
perm = rng.permutation(len(texts))
n_tr = int(len(texts)*0.85)
tr_set = set(perm[:n_tr].tolist())

K = 8                      # 上下文窗口
D = 1024                   # 复维
# 字向量: 单位相位复向量; 位置绑定: 每位置固定相移
zv = np.exp(2j*np.pi*np.random.default_rng(1).uniform(0, 1, (VOC+1, D)))
rho = np.exp(2j*np.pi*np.random.default_rng(2).uniform(0, 1, K))

def windows(tlist, targets_only=True):
    """产 (窗口字id, 目标字id)"""
    W, T = [], []
    for ti, t in enumerate(tlist):
        ids = [vmap.get(c, UNK) for c in t]
        for i in range(K, len(ids)):
            w = ids[i-K:i]
            if targets_only and ids[i] == UNK:
                continue                      # 目标限词表内
            W.append(w); T.append(ids[i])
    return np.array(W, dtype=np.int32), np.array(T, dtype=np.int32)

Wtr_all, Ttr_all = windows([texts[i] for i in tr_set])
Wte, Tte = windows([texts[i] for i in perm[n_tr:]])
# 控制训练量
MAXN = 150000
if len(Ttr_all) > MAXN:
    sel = np.random.default_rng(3).choice(len(Ttr_all), MAXN, replace=False)
    Wtr_all, Ttr_all = Wtr_all[sel], Ttr_all[sel]
print('训练窗口 %d, 测试窗口 %d' % (len(Ttr_all), len(Tte)))

# ---- 增量闭式解 (Gram 累计, 零梯度迭代) ----
NF = 2*D
ATA = np.zeros((NF, NF))
ATY = np.zeros((NF, VOC))
BLK = 3000
for s in range(0, len(Ttr_all), BLK):
    Wb, Tb = Wtr_all[s:s+BLK], Ttr_all[s:s+BLK]
    n = len(Tb)
    ctx = np.zeros((n, D), dtype=complex)
    for i in range(K):
        ctx += zv[Wb[:, i]]*rho[i]
    Xf = np.concatenate([ctx.real, ctx.imag], 1)
    Yoh = np.zeros((n, VOC))
    Yoh[np.arange(n), Tb] = 1.0
    ATA += Xf.T@Xf
    ATY += Xf.T@Yoh
lam = 1e-3
vo = np.linalg.solve(ATA/len(Ttr_all) + lam*np.eye(NF), ATY/len(Ttr_all))
print('闭式解完成 (2048→%d, 零梯度迭代)' % VOC)

# ---- 评估 ----
def eval_windows(W, T, vo, B=4000):
    top1 = top3 = tot = 0
    for s in range(0, len(T), B):
        Wb, Tb = W[s:s+B], T[s:s+B]
        n = len(Tb)
        ctx = np.zeros((n, D), dtype=complex)
        for i in range(K):
            ctx += zv[Wb[:, i]]*rho[i]
        Xf = np.concatenate([ctx.real, ctx.imag], 1)
        sc = Xf@vo
        top1 += int(np.sum(np.argmax(sc, 1) == Tb))
        top3 += int(np.sum(np.any(np.argsort(-sc, 1)[:, :3] == Tb[:, None], 1)))
        tot += n
    return top1/tot, top3/tot

acc1, acc3 = eval_windows(Wte, Tte, vo)
# 基线
uni = np.bincount(Ttr_all, minlength=VOC)/len(Ttr_all)
uni_acc = float(np.mean(uni[Tte] == np.max(uni)))  # 恒等于 max p — 改为: 预测最高频字的acc
uni_acc = float(np.mean(Tte == np.argmax(uni)))
# bigram
bi = collections.defaultdict(collections.Counter)
for Wb, Tb in ((Wtr_all, Ttr_all),):
    for p, t in zip(Wb[:, -1], Tb):
        bi[p][t] += 1
bi_pred = np.array([bi[p].most_common(1)[0][0] if p in bi else np.argmax(uni) for p in Wte[:, -1]])
bi_acc = float(np.mean(bi_pred == Tte))
tri = collections.defaultdict(collections.Counter)
for p2, t in zip(zip(Wtr_all[:, -2], Wtr_all[:, -1]), Ttr_all):
    tri[p2][t] += 1
tri_pred = np.array([tri[p2].most_common(1)[0][0] if p2 in tri else bi_pred[i]
                     for i, p2 in enumerate(zip(Wte[:, -2], Wte[:, -1]))])
tri_acc = float(np.mean(tri_pred == Tte))

print('\n=== §23: 容器 LM (HDC 8 字上下文 + 一次闭式解) vs 基线 ===')
print('unigram 基线:      top1 %.3f' % uni_acc)
print('bigram (查表):     top1 %.3f' % bi_acc)
print('trigram (查表):    top1 %.3f' % tri_acc)
print('容器 LM:           top1 %.3f  top3 %.3f' % (acc1, acc3))
print('\n判读: 容器 > trigram → 序列统计已进检索+闭式解形态, 权重的「训练」被代数解替代')
