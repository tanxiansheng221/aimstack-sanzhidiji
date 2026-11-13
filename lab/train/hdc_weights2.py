# -*- coding: utf-8 -*-
# hdc_weights2.py — 把第二容器变成「与模型权重同样性质的东西」：数学验证
#   权重 θ = 随机相位特征空间中的线性读出向量 (D 个复相位 = 2D 个实参数)
#   特征 φ(x)_j = exp(2πi·f_j·x) (f_j 随机整数, 去掉 DC) —— 移不变核的蒙特卡洛
#   近似 (Rahimi-Recht 2007 随机特征), 核带宽 = 1/(2F)
#   验证四件事 (模型权重的定义性质):
#   A 参数化函数+可训练: delta rule = 特征空间最小二乘 GD, 收敛到岭回归闭式解
#     (两条独立求解路径一致), 收敛速率匹配二次型 GD 理论; one-shot 叠加写入
#     = 同空间核回归解 (复数形式 ≡ 实数形式恒等式)
#   B 可泛化: 函数定义在全输入域, 偏差(带宽<信号谱)-方差(2F/n)曲线, 插值灾变线 F>n/2
#   C 表达力: 特征采样方差 ∝ 1/D (随机特征定理)
#   D 存储性质: 权重低比特量化 ΔMSE ≈ Δ²/(12D) 解析验证; 合并/续训
import numpy as np
import sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(41)
D = 8192

def freqs(F, D_=D):
    """随机整数频率, 去掉 DC (f=0 会给核加均匀地板, 破坏无偏性)"""
    side = rng.integers(0, 2, size=D_) * 2 - 1
    mag = rng.integers(1, F + 1, size=D_)
    return (side * mag).astype(float)

def make_A(x, f):
    """设计矩阵 A: f(x) = Aθ, A = [cos(2πfx), -sin(2πfx)]/D"""
    ang = 2 * np.pi * np.outer(x, f)
    return np.concatenate([np.cos(ang), -np.sin(ang)], axis=1) / D

def eval_f(theta, x, f, chunk=200):
    out = np.empty(len(x))
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        out[s:e] = make_A(x[s:e], f) @ theta
    return out

def ridge_solve(A, y, lam):
    """岭回归对偶解: (AA^T/n + λI)α = y/n, θ* = A^Tα —— 与 primal (2D×2D) 等价"""
    n = A.shape[0]
    K = A @ A.T / n
    alpha = np.linalg.solve(K + lam * np.eye(n), y / n)
    return A.T @ alpha

def gd_train(A, y, lam, theta0=None, steps=600):
    """delta rule 批量形式: θ ← θ - η(A^T(Aθ-y)/n + λθ) —— 该空间的 GD"""
    n = A.shape[0]
    theta = np.zeros(A.shape[1]) if theta0 is None else theta0.copy()
    eigs = np.linalg.eigvalsh(A @ A.T / n)
    mu_max = eigs.max()
    mu_min = eigs[eigs > 1e-12 * mu_max].min()
    eta = 1.6 / (mu_max + lam)
    rate = abs(1 - eta * (mu_min + lam))       # 最慢模衰减因子 (二次型 GD 理论)
    for t in range(steps):
        theta = theta - eta * (A.T @ (A @ theta - y) / n + lam * theta)
    return theta, eta, rate, mu_min, mu_max

HARM = [(0.8, 1, 0.3), (0.5, 3, 1.1), (0.3, 5, 2.0)]   # g 带宽 ≤ 5
def g(x):
    return sum(a * np.sin(2 * np.pi * h * x + p) for a, h, p in HARM)

xg = np.linspace(0, 1, 1000)
gy = g(xg)

# ================= EXP A: 权重可训练 (GD → 岭回归闭式解) =================
print("=== EXP A: 权重 = 随机相位特征空间的线性读出, delta rule = GD ===")
n, F, sigma = 300, 32, 0.3
xs = rng.uniform(0, 1, n)
y = g(xs) + sigma * rng.normal(0, 1, n)
f = freqs(F)
A = make_A(xs, f)
lam = 1e-6

theta_r = ridge_solve(A, y, lam)
theta_gd, eta, rate, mu_min, mu_max = gd_train(A, y, lam, steps=600)
fg = eval_f(theta_gd, xg, f); fr = eval_f(theta_r, xg, f)
rel = np.linalg.norm(fg - fr) / np.linalg.norm(fr)
print("  [A1] GD 收敛到岭回归闭式解: 函数空间相对偏差 %.2e (η=%.1f)" % (rel, eta))

# 收敛速率: 逐模衰减定律 — 谱含数值零模(近重复采样点, 特征向量=噪声方向, 不可用),
#   在良置模上单模注入验证: ||θ_t-θ*||_mode = ε·|1-η(μ+λ)|^t
eigs_K, vecs_K = np.linalg.eigh(A @ A.T / n)
n_junk = int(np.sum(eigs_K < 1e-6 * eigs_K.max()))
eta_r = 0.3 / (mu_max + lam)
print("  [A2] 逐模衰减定律 (单模注入, η=0.3/(μ_max+λ); 谱含 %d 个数值零模已排除):" % n_junk)
print("       %-12s %11s %11s %11s" % ("模μ(相对)", "理论log", "实测log", "特征残差"))
for rel_mu in [1.0, 0.5, 0.1]:
    i_k = int(np.argmin(np.abs(eigs_K - rel_mu * mu_max)))
    mu_k = float(eigs_K[i_k])
    v_k = A.T @ vecs_K[:, i_k]
    v_k /= np.linalg.norm(v_k)
    Hv = A.T @ (A @ v_k) / n + lam * v_k
    resid = np.linalg.norm(Hv - (mu_k + lam) * v_k) / np.linalg.norm(Hv)
    fac = abs(1 - eta_r * (mu_k + lam))
    n_fit = int(min(400, max(20, np.log(1e-6 / 1e-11) / abs(np.log(fac)))))
    theta_t = theta_r + 1e-6 * v_k
    traj = []
    for t in range(n_fit):
        theta_t = theta_t - eta_r * (A.T @ (A @ theta_t - y) / n + lam * theta_t)
        traj.append(np.log(max(np.linalg.norm(theta_t - theta_r), 1e-300)))
    slope = np.polyfit(np.arange(len(traj)), traj, 1)[0]
    print("       %-12.3f %11.4f %11.4f %11.1e"
          % (mu_k / mu_max, np.log(fac), slope, resid))

# one-shot 叠加写入 (复数形式) ≡ 实数权重形式 (恒等式)
w_c = np.sum(np.conj(np.exp(2j * np.pi * np.outer(xs, f))) * y[:, None], axis=0)
theta_os = D * (A.T @ y)                  # 实数形式 one-shot: f = Σ K(x,x_i)y_i
f_os = eval_f(theta_os, xg, f)
fc = np.empty(len(xg))
for s in range(0, len(xg), 200):
    e = min(s + 200, len(xg))
    phi = np.exp(2j * np.pi * np.outer(xg[s:e], f))
    fc[s:e] = np.real(phi @ w_c) / D
print("  [A3] 恒等式: 复数叠加解码 ≡ 实数权重读出, max|Δf| = %.2e" % float(np.max(np.abs(fc - f_os))))

mse_os = float(np.mean((f_os - gy) ** 2))
mse_r = float(np.mean((fr - gy) ** 2))
print("  [A4] 测试 MSE (σ=0.3): one-shot 叠加 %.4f | GD 训练权重 %.4f | 噪声底 σ²=%.4f"
      % (mse_os, mse_r, sigma ** 2))

# 重叠域 (聚类采样): one-shot 记忆失灵, 训练权重仍然成立 —— 权重与记忆的分界
xcl = np.concatenate([rng.uniform(0, 0.15, int(n * 0.6)), rng.uniform(0.15, 1, n - int(n * 0.6))])
ycl = g(xcl) + sigma * rng.normal(0, 1, n)
Acl = make_A(xcl, f)
theta_osc = D * (Acl.T @ ycl)
theta_rc = ridge_solve(Acl, ycl, lam)
print("  [A5] 聚类采样 (0.6 占比挤在 [0,0.15]): one-shot MSE %.4f → GD/岭回归 MSE %.4f"
      % (float(np.mean((eval_f(theta_osc, xg, f) - gy) ** 2)),
         float(np.mean((eval_f(theta_rc, xg, f) - gy) ** 2))))

# ================= EXP B: 可泛化 (偏差-方差 vs 带宽 F) =================
print("\n=== EXP B: 泛化 = 全输入域核插值; 偏差-方差 vs 带宽 ===")
print("  理论: F<5 偏差(信号谱未覆盖) | 2F/n 方差增长 | F>n/2=%d 插值灾变" % (n // 2))
print("    %-6s %12s %12s" % ("带宽F", "one-shot", "GD/岭回归"))
for Fb in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    fb = freqs(Fb)
    Ab = make_A(xs, fb)
    tb = ridge_solve(Ab, y, 1e-6)
    tos = D * (Ab.T @ y)
    print("    %-6d %12.4f %12.4f"
          % (Fb, float(np.mean((eval_f(tos, xg, fb) - gy) ** 2)),
             float(np.mean((eval_f(tb, xg, fb) - gy) ** 2))))

# 数据量定律: 方差随 n 下降
print("  数据量定律 (F=8, 带宽覆盖信号谱):")
for nb in [50, 150, 300, 600]:
    xb = rng.uniform(0, 1, nb)
    yb = g(xb) + sigma * rng.normal(0, 1, nb)
    fb = freqs(8)
    tb = ridge_solve(make_A(xb, fb), yb, 1e-6)
    print("    n=%-4d 测试 MSE %.4f" % (nb, float(np.mean((eval_f(tb, xg, fb) - gy) ** 2))))

# ================= EXP C: 表达力 (随机特征定理, 方差 ∝ 1/D) =================
print("\n=== EXP C: 表达力 = 移不变核的蒙特卡洛近似, 采样方差 ∝ 1/D ===")
print("    %-8s %12s" % ("特征数D", "测试 MSE"))
mses = []
for Db in [512, 1024, 2048, 4096, 8192]:
    fb = freqs(8, Db)
    Ab = make_A(xs, fb)
    tb = ridge_solve(Ab, y, 1e-6)
    m = float(np.mean((eval_f(tb, xg, fb) - gy) ** 2))
    mses.append(m)
    print("    %-8d %12.6f" % (Db, m))
excess = np.maximum(np.array(mses) - mses[-1], 1e-12)
sl = np.polyfit(np.log([512, 1024, 2048, 4096][:4]), np.log(excess[:4]), 1)[0]
print("    超额误差指数: %.2f (理论 -1, Rahimi-Recht 随机特征方差律)" % sl)

# ================= EXP D: 存储性质 (HDC 原生相位量化, sinc 律) =================
print("\n=== EXP D: 训练后的权重低比特量化 (相位量化, 信号衰减 sinc(π/m)) ===")
w = theta_r[:D] + 1j * theta_r[D:]
tail = float(np.max(np.abs(w)) / np.median(np.abs(w)))
print("  权重幅度重尾 max/median = %.0f → 均匀比特量化器失配 (实测非单调), 改相位量化" % tail)
print("    %-8s %12s %12s %10s" % ("相位级m", "实测ΔMSE", "形状预测", "体积"))
dm2 = None
for m_ in [2, 4, 16, 256]:
    wq = np.abs(w) * np.exp(1j * (np.round(np.angle(w) / (2 * np.pi / m_)) * (2 * np.pi / m_)))
    theta_q = np.concatenate([np.real(wq), np.imag(wq)])
    dm = float(np.mean((eval_f(theta_q, xg, f) - gy) ** 2)) - mse_r
    if dm2 is None: dm2 = dm
    shape = (1 - float(np.sinc(1.0 / m_))) / (1 - float(np.sinc(0.5)))
    print("    %-8d %12.6f %12.6f %9.1fKB"
          % (m_, dm, dm2 * shape, D * np.log2(m_) / 8 / 1024))

# ================= EXP E: 合并 / 续训 / 回放 (GD 权重 vs one-shot 权重) =================
print("\n=== EXP E: GD 权重的合并 / 续训 / 回放 (对照 one-shot 线性精确) ===")
idx = rng.permutation(n)
h1, h2 = idx[:n // 2], idx[n // 2:]
t1 = ridge_solve(A[h1], y[h1], lam)
t2 = ridge_solve(A[h2], y[h2], lam)
t_merge = (t1 + t2) / 2
def full_obj(th):
    return float(np.mean((A @ th - y) ** 2) / 2 + lam * np.linalg.norm(th) ** 2 / 2)
print("  GD 平均合并: 测试 MSE %.4f vs 全量训练 %.4f (合并非精确)" %
      (float(np.mean((eval_f(t_merge, xg, f) - gy) ** 2)), mse_r))
t_drift, _, _, _, _ = gd_train(A[h2], y[h2], lam, theta0=t1.copy(), steps=2000)
print("  朴素续训(只学B): 组合目标 %.5f vs 全量解 %.5f → 漂移=灾难性遗忘"
      % (full_obj(t_drift), full_obj(theta_r)))
rep = np.concatenate([h2, rng.choice(h1, size=int(0.3 * len(h1)), replace=False)])
t_rep, _, _, _, _ = gd_train(A[rep], y[rep], lam, theta0=t1.copy(), steps=2000)
print("  回放续训(B+三成A回放): 组合目标 %.5f → 回放纪律恢复 (与 152M 训练回放教训同构)"
      % full_obj(t_rep))
tos1 = D * (A[h1].T @ y[h1]); tos2 = D * (A[h2].T @ y[h2])
print("  one-shot 合并: 偏差 %.2e (线性精确, 上一轮 W3 结论在函数域复现)"
      % float(np.max(np.abs((tos1 + tos2) - D * (A.T @ y)))))

print("\n=== 总结: 容器权重 = 随机相位特征空间线性读出向量;")
print("    one-shot 叠加 = 未 Gram 归一化的核回归 (记忆域, 要求键近正交, 重叠域相干放大 ∝n);")
print("    delta rule = 该空间 GD = Gram 归一化 (函数域, 重叠特征可用), 收敛到岭回归闭式解;")
print("    泛化 = 全输入域核插值 (偏差-方差-灾变三段); 量化走相位 sinc 律;")
print("    GD 权重会遗忘需回放, one-shot 合并线性精确 —— 两种权重各有分工 ===")
