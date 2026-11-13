# -*- coding: utf-8 -*-
# §22: 语义泛化第一枪 —— 容器 × ESConv 话题分类
#   任务: seeker 话术 → problem_type (合并小类, 6 类)
#   编码: 字符 3-gram 哈希 → 随机复相位向量 (D=2048), 句向量=Σ (经典 HDC bind/sum)
#   模型: (a) 线性岭回归基线  (b) 随机 φ 特征 + tanh + 闭式头 (容器形态, 零训练先测)
#   泛化: 按对话切 80/20 (utterance 不跨 split) —— 话题级泛化才算语义通
import numpy as np, json, sys, re, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# ---- 数据 ----
d = json.load(open(r'H:\AimCode\lab\cpu-native\esconv\ESConv.json', encoding='utf-8'))
CANON = {'ongoing depression': 'depression', 'job crisis': 'job', 'breakup with partner':
         'breakup', 'problems with friends': 'friends', 'academic pressure': 'academic'}
def canon(p):
    return CANON.get(p, 'other')

rng = np.random.default_rng(7)
order = rng.permutation(len(d))
n_tr = int(len(d)*0.8)
tr_d, te_d = order[:n_tr], order[n_tr:]

def seeker_text(x):
    return ' '.join(u['content'] for u in x['dialog'] if u['speaker'] == 'seeker')

texts, labels = [], []
for i, x in enumerate(d):
    texts.append(seeker_text(x)); labels.append(canon(x['problem_type']))
classes = sorted(set(labels))
cls_idx = {c: i for i, c in enumerate(classes)}
Y = np.array([cls_idx[l] for l in labels])
print('类别分布:', {c: labels.count(c) for c in classes})
print('train 对话 %d / test 对话 %d' % (n_tr, len(d)-n_tr))

# ---- HDC 编码: char 3-gram → 复相位向量 ----
D = 2048
phi_tab = rng.uniform(0, 1, D)
def encode(t):
    t = re.sub(r'[^a-z ]', ' ', t.lower())
    g = [t[i:i+3] for i in range(max(1, len(t)-2))]
    cnt = {}
    for s in g:
        idx = int(hashlib.md5(s.encode()).hexdigest(), 16) % D
        cnt[idx] = cnt.get(idx, 0) + 1
    v = np.zeros(D, dtype=complex)
    for idx, c in cnt.items():
        v[idx] += c * np.exp(2j*np.pi*phi_tab[idx])
    nv = np.linalg.norm(v)
    return np.concatenate([v.real, v.imag])/max(nv, 1e-9)

X = np.stack([encode(t) for t in texts])
Xtr, Xte = X[tr_d], X[te_d]
ytr, yte = Y[tr_d], Y[te_d]
print('特征维度:', X.shape[1], ' train 样本:', len(ytr))

# ---- 模型: 闭式岭回归头 (one-hot) ----
def ridge_head(A, y, n_cls, lam=1e-3):
    Yoh = np.zeros((len(y), n_cls)); Yoh[np.arange(len(y)), y] = 1.0
    G = A.T@A/len(y) + lam*np.eye(A.shape[1])
    vo = np.linalg.solve(G, A.T@Yoh/len(y))
    return vo
def acc_of(A, vo, y):
    return float(np.mean(np.argmax(A@vo, 1) == y))

maj = float(np.mean(ytr == np.bincount(ytr).argmax()))
print('\n=== §22: 语义第一枪 (6 类话题, 对话级 split) ===')
print('majority 基线: %.3f' % maj)

# (a) 线性基线
vo = ridge_head(Xtr, ytr, len(classes))
print('线性岭回归 (4096 维直接): acc %.3f  macro-F1 见下' % acc_of(Xte, vo, yte))

# (b) 容器形态: 随机 φ 特征 + tanh + 闭式头 (W 固定, 零训练先测) + 频率尺度扫描
r2 = np.random.default_rng(11)
Wfix = r2.normal(0, 0.5, (64, 512))
def feats(Xin, F):
    A = np.concatenate([np.cos(2*np.pi*Xin@F.T), -np.sin(2*np.pi*Xin@F.T)], 1)/np.sqrt(256)
    return np.tanh(A@Wfix.T)
for fs in (0.5, 2.0, 8.0):
    F = r2.uniform(-1, 1, (256, X.shape[1]))*fs/np.sqrt(X.shape[1])
    vo2 = ridge_head(feats(Xtr, F), ytr, len(classes))
    ate = acc_of(feats(Xte, F), vo2, yte)
    print('容器形态 (φ256 fs=%.1f + tanh64 + 闭式头): acc %.3f' % (fs, ate))

# macro-F1 (双模型)
def macro_f1(A, vo, y):
    pred = np.argmax(A@vo, 1)
    f1s = []
    for c in range(len(classes)):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(2*tp/max(2*tp+fp+fn, 1))
    return float(np.mean(f1s))
print('macro-F1: 线性 %.3f' % macro_f1(Xte, vo, yte))

# (c) 训练版容器: CE 联训 tanh 层 + 线性头 (对照线性 0.635)
def train_sem(H=64, steps=400, lr=0.01, fs=2.0, seed=13):
    r3 = np.random.default_rng(seed)
    F = r3.uniform(-1, 1, (256, X.shape[1]))*fs/np.sqrt(X.shape[1])
    W = r3.normal(0, 0.5, (H, 512))
    vo = np.zeros((H, len(classes)))
    Atr = np.concatenate([np.cos(2*np.pi*Xtr@F.T), -np.sin(2*np.pi*Xtr@F.T)], 1)/np.sqrt(256)
    Ate = np.concatenate([np.cos(2*np.pi*Xte@F.T), -np.sin(2*np.pi*Xte@F.T)], 1)/np.sqrt(256)
    mm = [np.zeros_like(p) for p in (W, vo)]; vv = [np.zeros_like(p) for p in (W, vo)]
    Yoh = np.zeros((len(ytr), len(classes))); Yoh[np.arange(len(ytr)), ytr] = 1.0
    for step in range(1, steps+1):
        Htr = np.tanh(Atr@W.T)
        P = np.exp(Htr@vo - Htr@vo.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        dP = (P - Yoh)/len(ytr)
        gv = Htr.T@dP
        gH = (dP@vo.T)*(1-Htr**2)
        gW = gH.T@Atr
        for i, (p, g) in enumerate(((W, gW), (vo, gv))):
            mm[i], vv[i] = 0.9*mm[i]+0.1*g, 0.999*vv[i]+0.001*g*g
            p -= lr*(mm[i]/(1-0.9**step))/(np.sqrt(vv[i]/(1-0.999**step))+1e-8)
    return acc_of(np.tanh(Ate@W.T), vo, yte)

for H in (64, 256):
    a = train_sem(H=H)
    print('训练容器 (CE, H=%d): acc %.3f' % (H, a))
print('\n判读: 训练容器 ≥ 线性(0.635) → 容器深度在真实语义上有正红利')
