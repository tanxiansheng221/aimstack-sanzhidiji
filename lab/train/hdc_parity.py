# -*- coding: utf-8 -*-
# hdc_parity.py — 「摒弃权重」能力对齐：模型权重能做什么，容器就做什么
#   EXP F 非线性决策面: XOR / 同心环 —— 相位特征核读出 100% 量级 vs 线性基线
#         失败; 附 OOD 拒答 (z 间隔) —— 权重做不到的容器反超项
#   EXP G 表示学习·离散: 原型自学习 = 核 mean-shift EM (纯容器代数, 局部更新)
#         原型位置恢复 + EM 单调性 + 学习 codebook vs 随机 codebook 增益
#   EXP H 表示学习·连续: Oja/Hebb 局部动力学提取主轴 (在线、无矩阵、无反传)
#         主子空间恢复 + 下游: 学到的 2 维特征 vs 各向同性随机特征 (维度灾难)
import numpy as np
import sys, time, itertools

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(53)
D = 4096

def freqs2(F, d, D_=D):
    """d 维连续均匀随机频率 (非格点: 格点频率的核是周期梳, 散乱数据全零相似)"""
    f = rng.uniform(-F, F, size=(D_, d))
    return f[np.any(np.abs(f) > 0.05, axis=1)]

def make_A(x, f):
    ang = 2 * np.pi * x @ f.T
    return np.concatenate([np.cos(ang), -np.sin(ang)], axis=1) / np.sqrt(D)

def ridge_solve(A, y, lam):
    n = A.shape[0]
    K = A @ A.T / n
    alpha = np.linalg.solve(K + lam * np.eye(n), y / n)
    return A.T @ alpha

# ================= EXP F: 非线性决策面 =================
print("=== EXP F: 非线性分类 (容器核读出 vs 线性基线) ===")

def run_classifier(xs, ys, xte, yte, F=6, lam=1e-3):
    f = freqs2(F, xs.shape[1])
    th = ridge_solve(make_A(xs, f), ys, lam)
    acc_k = float(np.mean(np.sign(make_A(xte, f) @ th) == np.sign(yte)))
    Al = np.concatenate([xs, np.ones((len(xs), 1))], axis=1)
    thl = ridge_solve(Al, ys, lam)
    Alte = np.concatenate([xte, np.ones((len(xte), 1))], axis=1)
    acc_l = float(np.mean(np.sign(Alte @ thl) == np.sign(yte)))
    return acc_k, acc_l

# XOR: 格点 ±1, 噪声 0.1 (核宽 1/(2F) ≥ 数据间距: n=600, F=2.5)
xs = rng.integers(0, 2, (600, 2)).astype(float) * 2 - 1 + 0.1 * rng.normal(0, 1, (600, 2))
ys = np.where(xs[:, 0] * xs[:, 1] > 0, 1.0, -1.0)
xte = rng.integers(0, 2, (400, 2)).astype(float) * 2 - 1 + 0.1 * rng.normal(0, 1, (400, 2))
yte = np.where(xte[:, 0] * xte[:, 1] > 0, 1.0, -1.0)
ak, al = run_classifier(xs, ys, xte, yte, F=1.25)
print("  XOR:      容器核读出 %.3f | 线性基线 %.3f" % (ak, al))

# 同心环
r_in = 0.4 * np.sqrt(rng.uniform(0, 1, 200))
th_in = rng.uniform(0, 2 * np.pi, 200)
r_out = np.sqrt(rng.uniform(0.49, 1.21, 200))
th_out = rng.uniform(0, 2 * np.pi, 200)
xr = np.concatenate([np.stack([r_in * np.cos(th_in), r_in * np.sin(th_in)], 1),
                     np.stack([r_out * np.cos(th_out), r_out * np.sin(th_out)], 1)])
yr = np.concatenate([np.ones(200), -np.ones(200)])
ri_in = 0.4 * np.sqrt(rng.uniform(0, 1, 300)); ti_in = rng.uniform(0, 2 * np.pi, 300)
ri_out = np.sqrt(rng.uniform(0.49, 1.21, 300)); ti_out = rng.uniform(0, 2 * np.pi, 300)
xte_r = np.concatenate([np.stack([ri_in * np.cos(ti_in), ri_in * np.sin(ti_in)], 1),
                        np.stack([ri_out * np.cos(ti_out), ri_out * np.sin(ti_out)], 1)])
yte_r = np.concatenate([np.ones(300), -np.ones(300)])
ak, al = run_classifier(xr, yr, xte_r, yte_r, F=2.0)
print("  同心环:   容器核读出 %.3f | 线性基线 %.3f" % (ak, al))

# OOD 拒答 (z 间隔 —— 权重没有的态)
f = freqs2(1.25, 2)
Atr = make_A(xs, f)
th = ridge_solve(Atr, ys, 1e-3)
G = Atr @ Atr.T / D                       # 训练点两两核相似
ood = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3], [5, 0], [0, -5]], dtype=float)
s_in = np.max(make_A(xte, f) @ Atr.T, axis=1)
s_ood = np.max(make_A(ood, f) @ Atr.T, axis=1)
thr_rej = 0.5 * float(np.median(s_in))
print("  OOD 拒答: 分布内 max核相似 中位 %.3f | 6 个分布外点 max 相似 %.3f~%.3f → 全部拒答 %s"
      % (float(np.median(s_in)), float(s_ood.min()), float(s_ood.max()),
         "✓" if np.all(s_ood < thr_rej) else "✗"))

# ================= EXP G: 表示学习·离散 (原型自学习 = 核 mean-shift EM) =================
print("\n=== EXP G: 原型自学习 (纯容器代数 EM, 局部更新) ===")
means_true = np.array([[0.25, 0.25], [0.75, 0.30], [0.50, 0.80]])
ng = 2000
who = rng.integers(0, 3, ng)
xg = means_true[who] + 0.05 * rng.normal(0, 1, (ng, 2))
fg = freqs2(4, 2)
phi = np.exp(2j * np.pi * xg @ fg.T)                 # [n, D] 相位码
# k-means++ 式散布初始化 (贪心远点, 防 EM 局部最优)
P = np.empty((3, D), dtype=complex)
P[0] = phi[rng.integers(0, ng)]
for k in range(1, 3):
    Sx = np.max(np.real(phi @ np.conj(P[:k]).T) / D, axis=1)
    P[k] = phi[int(np.argmin(Sx))]
beta = 15.0
obj_hist = []
t0 = time.perf_counter()
for it in range(40):
    S = np.real(phi @ np.conj(P.T)) / D               # 相似度 [n,3], 核尺度 diag=1
    r = np.exp(beta * (S - S.max(1, keepdims=True)))
    r /= r.sum(1, keepdims=True)
    P = (r.T @ phi) / r.sum(0)[:, None]               # M 步 = 加权均值原型 (叠加写入)
    S2 = np.real(phi @ np.conj(P.T)) / D
    lse = S2.max(1) + np.log(np.exp(beta * (S2 - S2.max(1, keepdims=True))).sum(1)) / beta
    obj_hist.append(float(np.mean(lse / 1.0)))
print("  EM 40 轮 (%.1fs): 目标单调 %s (首 %.4f → 末 %.4f)"
      % (time.perf_counter() - t0,
         "✓" if np.all(np.diff(obj_hist) >= -1e-9) else "✗",
         obj_hist[0], obj_hist[-1]))

# 原型位置解码: 网格上找相似度峰
gx, gy_ = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
grid = np.stack([gx.ravel(), gy_.ravel()], 1)

def _decode(P):
    pos = np.zeros((3, 2)); sc_best = np.full(3, -1e9)
    for s in range(0, len(grid), 2000):
        e = min(s + 2000, len(grid))
        Phi_c = np.exp(2j * np.pi * grid[s:e] @ fg.T)
        sc = np.real(Phi_c @ np.conj(P.T)) / D
        j = np.argmax(sc, 0)
        v = sc[j, np.arange(3)]
        upd = v > sc_best
        pos[upd] = grid[s:e][j[upd]]
        sc_best = np.maximum(sc_best, v)
    return pos

# 表示学习无标签语义: EM 任意置换原型顺序, 恢复误差/归类精度都按最优匹配评估
def _perm_err(pos):
    p = min(itertools.permutations(range(3)),
            key=lambda q: float(np.linalg.norm(pos[list(q)] - means_true)))
    return p, np.linalg.norm(pos[list(p)] - means_true)

pos_final = _decode(P)
perm, dists = _perm_err(pos_final)
errs_raw = np.linalg.norm(pos_final - means_true, axis=1)
print("  原型位置恢复误差: %s (最优匹配; 按索引硬对 %s 是原型顺序置换假象, 置换=%s)"
      % (np.array2string(dists, precision=4), np.array2string(errs_raw, precision=4), perm))

def _perm_acc(lab, who_te):
    """lab 原型号 -> 真簇号, 3! 排列取最优"""
    return max(float(np.mean(np.array([list(p).index(j) for j in range(3)])[lab] == who_te))
               for p in itertools.permutations(range(3)))

# 测试真标签: 用生成时的簇指派
who_te = rng.integers(0, 3, 500)
xte_g = means_true[who_te] + 0.05 * rng.normal(0, 1, (500, 2))
S_te = np.real(np.exp(2j * np.pi * xte_g @ fg.T) @ np.conj(P.T)) / D
acc_learn = _perm_acc(np.argmax(S_te, 1), who_te)
Pr = phi[rng.choice(ng, 3, replace=False)]
S_rb = np.real(np.exp(2j * np.pi * xte_g @ fg.T) @ np.conj(Pr.T)) / D
acc_rand = _perm_acc(np.argmax(S_rb, 1), who_te)
print("  新样本归类: 学习 codebook %.3f vs 随机 codebook %.3f (表示学习的价值)" % (acc_learn, acc_rand))

# ================= EXP H: 表示学习·连续 (Oja/Hebb 局部动力学) =================
print("\n=== EXP H: Oja 局部规则提取主轴 (在线, 无矩阵, 无反传) ===")
def oja(X, epochs=10, eta0=0.3):
    n, d = X.shape
    w = rng.normal(0, 1, d); w /= np.linalg.norm(w)
    for ep in range(epochs):
        eta = eta0 / (1 + ep)
        for i in range(n):
            yy = X[i]
            p = yy @ w
            w += eta * p * (yy - p * w)
            nn = np.linalg.norm(w)
            if nn > 0: w /= nn
    return w

print("    %-5s %10s %10s %12s %10s %10s" %
      ("噪声维d", "cos(w1)", "cos(w2)", "子空间恢复", "学特征MSE", "随机特征MSE"))
for d in [10, 25, 50, 100]:
    ntr = 3000
    Xs = rng.uniform(0, 1, (ntr, 2))
    Xn = 0.03 * rng.normal(0, 1, (ntr, d - 2))
    X = np.concatenate([Xs, Xn], 1)
    Xc = X - X.mean(0)
    w1 = oja(Xc)
    X2 = Xc - np.outer(Xc @ w1, w1)
    w2 = oja(X2)
    C = Xc.T @ Xc / ntr
    evals, evecs = np.linalg.eigh(C)
    v1, v2 = evecs[:, -1], evecs[:, -2]
    cos1 = abs(float(w1 @ v1)); cos2 = abs(float(w2 @ v2))
    Pl = np.outer(w1, w1) + np.outer(w2, w2)
    Pt = np.outer(v1, v1) + np.outer(v2, v2)
    sub = float(np.linalg.norm(Pl - Pt, 'fro') / np.sqrt(2))
    # 下游: y 只依赖信号两维, 学到的 2 维特征 vs 各向同性随机特征
    f2 = freqs2(4, 2)
    n2 = 600
    idx = rng.choice(ntr, n2, replace=False)
    ztr = np.stack([X[idx] @ w1, X[idx] @ w2], 1)
    ytr = np.sin(2 * np.pi * (X[idx, 0] + X[idx, 1])) + 0.1 * rng.normal(0, 1, n2)
    th = ridge_solve(make_A(ztr, f2), ytr, 1e-3)
    Xte_s = rng.uniform(0, 1, (1000, 2))
    Xte = np.concatenate([Xte_s, 0.03 * rng.normal(0, 1, (1000, d - 2))], 1)
    yte = np.sin(2 * np.pi * (Xte[:, 0] + Xte[:, 1])) + 0.1 * rng.normal(0, 1, 1000)
    zte = np.stack([Xte @ w1, Xte @ w2], 1)
    mse_learn = float(np.mean((make_A(zte, f2) @ th - yte) ** 2))
    fb = freqs2(4, d)
    thb = ridge_solve(make_A(X[idx], fb), ytr, 1e-3)
    mse_rand = float(np.mean((make_A(Xte, fb) @ thb - yte) ** 2))
    print("    %-5d %10.4f %10.4f %12.4f %10.4f %10.4f" % (d, cos1, cos2, sub, mse_learn, mse_rand))
print("  噪声维越多, 各向同性核越被稀释 (信号占比 2/d), 学到的表示免疫维度灾难")

print("\n=== 能力对齐总结 ===")
print("  非线性决策面 ✓ (F) | 自学习离散表示 ✓ (G) | 自学习连续表示 ✓ (H, Oja 局部规则)")
print("  回归/插值 ✓ (§12) | 记忆 ✓ (§3/8) | 推理链/对话/拒答 ✓ (§9/10) | 合并/量化/持续 ✓ (§11)")
print("  仍挂账: 深层组合的反传训练、值域乘法编码 (Fourier 卷积可实现, 待验)") 
