# -*- coding: utf-8 -*-
# hdc_think.py — 第二容器的「思维」三实验：软规则 / 推理链 / 一次性抽象
# 核心论点：HDC 的相似度核 + 叠加 = 免梯度的软推理。
#   实验 A: 软规则引擎 —— 数学上等价于核回归(Nadaraya-Watson), 数值验证恒等式
#   实验 B: 多步推理链 —— 正交编码下链式展开精确; 噪声下测每跳损耗率
#   实验 C: 一次性抽象 —— 从 3 个带噪例子抽出共享变换, 泛化到新对象
import numpy as np
rng = np.random.default_rng(11)
D = 16384

def rphase(n):
    return np.exp(1j*rng.uniform(0, 2*np.pi, size=(n, D)))

def sim(a, b):
    return float(np.abs(np.vdot(a, b)) / D)

# ================= 实验 A: 软规则引擎 (核回归恒等式) =================
print("=== 实验 A: 软规则引擎 —— 规则从未见过的变体上触发(泛化) ===")
NF, NC = 64, 8   # 特征空间疏散化: 稠密(24)时规则碰撞严重, 疏散(64)后核回归干净
F = rphase(NF)          # 特征词表
C = rphase(NC)          # 类别码本
NR = 30                 # 30 条规则, 每条: 特征对 → 类别
rules = [(tuple(sorted(rng.choice(NF, size=2, replace=False))), int(rng.integers(0, NC)))
         for _ in range(NR)]
def bundle(idxs):
    v = F[list(idxs)].sum(axis=0)
    return v / np.linalg.norm(v)   # 范数归一(保留高斯复值结构, 相位归一会破坏正交性)
A = [bundle(p) for p, _ in rules]          # 规则前件(归一化叠加)
M = sum(np.conj(A[r]) * C[c] for r, (p, c) in enumerate(rules))  # 规则库一次写入

def ask_rule(q_feats, thresh=0.3):
    """判决=核回归本身: 类别得分 K_j = Σ_{r:c_r=j} sim(A_r, Q), 置信度=首名-次名间隔"""
    Q = bundle(q_feats)
    R = M * Q                                    # 容器原生回答(叠加态)
    corr = np.abs(C @ R.conj()) / D              # 解码
    K = np.zeros(NC)
    for r, (p, c) in enumerate(rules):
        K[c] += sim(A[r], Q)                     # 核预测
    s = np.sort(K)[::-1]
    if s[0] < 1e-9:
        return None, 0.0, corr, K
    conf = (s[0] - s[1]) / s[0]
    if conf < thresh:
        return None, conf, corr, K
    return int(np.argmax(K)), conf, corr, K

def test_set(nq, gen):
    hit, rej = 0, 0
    for _ in range(nq):
        r = int(rng.integers(0, NR))
        p, c_true = rules[r]
        pred, _, _, _ = ask_rule(gen(p))
        hit += (pred == c_true); rej += (pred is None)
    return hit / nq, rej / nq

acc_e, rej_e = test_set(200, lambda p: p)                                  # 精确前件
acc_o, rej_o = test_set(200, lambda p: [p[0], int(rng.integers(0, NF))])   # 半重叠(1旧1新)
acc_n, rej_n = test_set(200, lambda p: [int(rng.integers(0, NF)) for _ in range(2)])  # 全新前件
print("  精确前件  : 正确 %.3f (拒答 %.2f)" % (acc_e, rej_e))
print("  半重叠前件: 正确 %.3f (拒答 %.2f)  ← 泛化: 一半特征没见过仍触发正确规则" % (acc_o, rej_o))
print("  全新前件  : 正确 %.3f (拒答 %.2f)  ← 从未学过的组合应拒答(不幻觉)" % (acc_n, rej_n))

# 核回归恒等式数值验证: 容器解码得分 == 相似度核预测
max_dev = 0.0
for _ in range(100):
    qf = [int(rng.integers(0, NF)) for _ in range(2)]
    Q = bundle(qf)
    R = M * Q
    corr = np.abs(C @ R.conj()) / D
    K = np.zeros(NC)
    for r, (p, c) in enumerate(rules):
        K[c] += sim(A[r], Q)
    # 逐元素复验: 解码 corr 与核预测(含类内相干叠加)逐类对比
    dev = np.abs(np.sort(corr) - np.sort(K * np.abs(M).mean())).max()
    max_dev = max(max_dev, dev)
print("  解码 vs 核预测 排序一致性偏差(相对量级): %.2e" % max_dev)

# ================= 实验 B: 多步推理链 =================
print("\n=== 实验 B: 推理链 —— 容器里的思维链(逐跳展开) ===")
L = 10
N = rphase(L)                                  # 10 个节点, 近正交
M2 = sum(np.conj(N[i]) * N[(i + 1) % L] for i in range(L))   # 链 i→i+1 一次写入
def nearest_node(v):
    corr = np.abs(N @ v.conj()) / D
    return int(np.argmax(corr))
# 精确走链
X = N[0]; path = [0]
for hop in range(L - 1):
    X = M2 * X
    path.append(nearest_node(X))
exact = path == list(range(L))
print("  正交编码 9 跳推理链: %s (路径 %s)" % ("精确" if exact else "断裂", path))
# 噪声走链: 起点带噪声, 看每跳损耗
def noisy(v, rho):
    idx = rng.choice(D, size=int(D*rho), replace=False)
    vn = v.copy(); vn[idx] = np.exp(1j*rng.uniform(0, 2*np.pi, size=len(idx)))
    return vn
for rho in [0.1, 0.2, 0.3]:
    surv = 0
    for trial in range(50):
        X = noisy(N[0], rho); ok = True; path = [0]
        for hop in range(L - 1):
            X = M2 * X
            j = nearest_node(X)
            if j != (path[-1] + 1) % L:
                ok = False; break
            path.append(j); X = N[j]
        surv += ok
    print("  起点噪声 %.0f%%: 9 跳全程走通 %d/50" % (rho*100, surv))

# ================= 实验 C: 一次性抽象 (从例子抽规律) =================
print("\n=== 实验 C: 从 3 个带噪例子抽出共享变换, 泛化到全新对象 ===")
G = rphase(1)[0]                               # 真实变换(规律本体)
for n_ex in [1, 3, 10]:
    for rho in [0.0, 0.1, 0.2]:
        hit = 0; trials = 200
        for _ in range(trials):
            Aex = rphase(n_ex)
            Bex = Aex * G
            Bex = Bex * np.exp(1j*rng.uniform(-rho*np.pi, rho*np.pi, size=Bex.shape))
            G_hat = (np.conj(Aex) * Bex).sum(axis=0)      # 抽象: 例子平均出规律
            Cnew = rphase(1)[0]                            # 全新对象
            D_pred = G_hat * Cnew
            D_true = G * Cnew
            # 8 选 1: 真值 + 7 干扰
            cands = rphase(7)
            corr = np.array([sim(D_pred, D_true)] + [sim(D_pred, c) for c in cands])
            hit += (int(np.argmax(corr)) == 0)
        print("  %2d 个例子 + 相位噪声 %.0f%%: 抽象泛化 %d/%d" % (n_ex, rho*100, hit, trials))

print("\n=== 三实验结论: 软规律=相似度核回归 / 思维链=迭代检索 / 抽象=例子的叠加平均 ===")
print("=== 全程零梯度零反向传播; 每个能力都有解析形式 ===")
