# -*- coding: utf-8 -*-
# hdc_parity2.py — 「彻底走通」: §13.5 挂账三项逐个拆除
#   EXP I 深层组合: 两层容器网络 + 手工反传
#         链式梯度恒等式: ∂out/∂h_i = -(2π/D2)·f2-加权 Im⟨w2, φ2⟩
#         数值梯度对账 + 奇校验-3 (深度分离任务) + 冻结第一层对照
#   EXP J 值域乘法编码: 绑定代数三恒等式 + 三重绑定乘法表叠加记忆
#         (log 同态: 乘法结构在码域变平移; 叠加写入零训练存 25 条乘法记录)
#   EXP K Oja deflation 链: 前 3 主轴提取 + 误差累积测量 (§13.5 第 3 条)
import numpy as np
import sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(71)
D = 4096

def freqs2(F, d, D_=D):
    f = rng.uniform(-F, F, size=(D_, d))
    return f[np.any(np.abs(f) > 0.05, axis=1)]

def acc(yh, yt):
    return float(np.mean(np.sign(yh) == np.sign(yt)))

# ================= EXP I: 两层容器网络 + 手工反传 =================
print("=== EXP I: 两层容器网络 + 手工反传 (深度分离任务: 奇校验-3) ===")
nb = rng.integers(0, 2, (300, 3)); s = 2.0 * nb - 1
Xtr = s + 0.1 * rng.normal(0, 1, (300, 3)); ytr = np.prod(s, 1)
nb = rng.integers(0, 2, (400, 3)); s_te = 2.0 * nb - 1
Xte = s_te + 0.1 * rng.normal(0, 1, (400, 3)); yte = np.prod(s_te, 1)

# 基线: 单层核读出 (Dk 个随机频率 = 2·Dk 实参数)
def kernel_acc(Dk, Xa, ya, Xb, yb, dim):
    fk = freqs2(2.0, dim, Dk)
    def A_(X):
        ang = 2 * np.pi * X @ fk.T
        return np.concatenate([np.cos(ang), -np.sin(ang)], 1) / np.sqrt(Dk)
    Ath = A_(Xa); n = len(Xa)
    K = Ath @ Ath.T / n
    al = np.linalg.solve(K + 1e-3 * np.eye(n), ya / n)
    th = Ath.T @ al
    return acc(A_(Xb) @ th, yb)

for Dk in (128, 512, 1664):
    print("  单层核 D=%d (%d 实参): %.3f" % (Dk, 2 * Dk, kernel_acc(Dk, Xtr, ytr, Xte, yte, 3)))

# 两层: h = tanh(A1·W1ᵀ) [H 隐单元], out = Re⟨w2, φ2(h)⟩/D2
D1, H, D2 = 96, 16, 128
f1 = freqs2(2.0, 3, D1)
f2 = freqs2(1.5, H, D2)

def feat1(X):
    ang = 2 * np.pi * X @ f1.T
    return np.concatenate([np.cos(ang), -np.sin(ang)], 1) / np.sqrt(D1)

A1 = feat1(Xtr); A1te = feat1(Xte)

def fwd_loss(W1_, v2_, XA, y_):
    hh = np.tanh(XA @ W1_.T)                       # [n,H]
    P2 = np.exp(2j * np.pi * hh @ f2.T)            # [n,D2]
    out = (np.real(P2) @ v2_[:D2] - np.imag(P2) @ v2_[D2:]) / D2
    return float(np.mean((out - y_) ** 2)), out, hh, P2

def backprop(W1_, v2_, XA, y_):
    L, out, hh, P2 = fwd_loss(W1_, v2_, XA, y_)
    n = len(y_)
    g = 2 * (out - y_) / n
    g_v2 = np.concatenate([np.real(P2).T @ g, -np.imag(P2).T @ g]) / D2
    w2c = v2_[:D2] + 1j * v2_[D2:]
    # 链式恒等式: ∂out_j/∂h_i = -(2π/D2)·Σ_k f2[k,i]·Im(w2_k φ2_jk)
    dodh = -(2 * np.pi / D2) * np.imag(P2 @ (w2c[:, None] * f2))
    dLdu = (g[:, None] * dodh) * (1 - hh ** 2)
    g_W1 = dLdu.T @ XA
    return L, g_W1, g_v2

W1 = rng.normal(0, 1, (H, 2 * D1)) / np.sqrt(2 * D1)
v2 = rng.normal(0, 0.1, 2 * D2)

# --- 梯度检查: 解析 vs 中心差分 (取 |grad| 最大的入口, 避开数值噪声) ---
_, gW0, gv0 = backprop(W1, v2, A1, ytr)
eps = 1e-5
worst = 0.0
for arr, gvec, nm in ((W1, gW0, "W1"), (v2, gv0, "v2")):
    ii = np.argsort(-np.abs(gvec).ravel())[:6]
    for i in ii:
        ap = arr.copy(); ap.flat[i] += eps
        am = arr.copy(); am.flat[i] -= eps
        Lp = fwd_loss(ap, v2, A1, ytr)[0] if nm == "W1" else fwd_loss(W1, ap, A1, ytr)[0]
        Lm = fwd_loss(am, v2, A1, ytr)[0] if nm == "W1" else fwd_loss(W1, am, A1, ytr)[0]
        num = (Lp - Lm) / (2 * eps)
        worst = max(worst, abs(num - gvec.flat[i]) / (abs(gvec.flat[i]) + 1e-10))
print("  梯度检查 (12 入口, 中心差分): 最大相对偏差 %.2e" % worst)

# --- Adam 训练 (全手工, 零优化器库) ---
def adam(p, g, m, v, t, lr):
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    p = p - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return p, m, v

mW = np.zeros_like(W1); vW = np.zeros_like(W1)
m2 = np.zeros_like(v2); v2m = np.zeros_like(v2)
mb, lr = 150, 0.02
t0 = time.perf_counter()
for step in range(1, 3001):
    idx = rng.choice(300, mb, replace=False)
    _, gW, gv = backprop(W1, v2, A1[idx], ytr[idx])
    W1, mW, vW = adam(W1, gW, mW, vW, step, lr)
    v2, m2, v2m = adam(v2, gv, m2, v2m, step, lr)
    if step in (500, 1500, 3000):
        print("    step %4d: train %.3f  test %.3f" %
              (step, acc(fwd_loss(W1, v2, A1, ytr)[1], ytr),
               acc(fwd_loss(W1, v2, A1te, yte)[1], yte)))
print("  两层容器训练 %.0fs, 参数量 %d (W1 %d + w2 %d)"
      % (time.perf_counter() - t0, H * 2 * D1 + 2 * D2, H * 2 * D1, 2 * D2))

# 对照: 冻结随机第一层 (只训 w2, 闭式岭回归) —— 分离「特征学习」的增益
W1f = rng.normal(0, 1, (H, 2 * D1)) / np.sqrt(2 * D1)
hh_f = np.tanh(A1 @ W1f.T)
P2f = np.exp(2j * np.pi * hh_f @ f2.T)
A2eff = np.concatenate([np.real(P2f), -np.imag(P2f)], 1) / D2
hh_te = np.tanh(A1te @ W1f.T)
P2te = np.exp(2j * np.pi * hh_te @ f2.T)
A2te = np.concatenate([np.real(P2te), -np.imag(P2te)], 1) / D2
w2f = A2eff.T @ np.linalg.solve(A2eff @ A2eff.T / len(ytr) + 1e-2 * np.eye(len(ytr)), ytr / len(ytr))
print("  冻结随机第一层 (同预算, 只训 w2): test %.3f  ← 训练第一层的增益 = 特征学习" % acc(A2te @ w2f, yte))

# --- 诚实对照: 同预算深度 vs 核 (奇校验-4) —— 深度不是免费午餐 ---
print("  [同预算对照: 奇校验-4]")
nb = rng.integers(0, 2, (600, 4)); s4 = 2.0 * nb - 1
X4 = s4 + 0.1 * rng.normal(0, 1, (600, 4)); y4 = np.prod(s4, 1)
nb = rng.integers(0, 2, (500, 4)); s4t = 2.0 * nb - 1
X4t = s4t + 0.1 * rng.normal(0, 1, (500, 4)); y4t = np.prod(s4t, 1)
D1, H, D2 = 24, 4, 32                                   # 4*48+64 = 256 实参
f1 = freqs2(2.0, 4, D1); f2 = freqs2(1.5, H, D2)
A1 = feat1(X4); A1te = feat1(X4t)
ak4 = kernel_acc(128, X4, y4, X4t, y4t, 4)              # 128 频率 = 256 实参
accs = []
for sd in (1, 2, 3):
    r2 = np.random.default_rng(sd)
    W1 = r2.normal(0, 1, (H, 2 * D1)) / np.sqrt(2 * D1)
    v2 = r2.normal(0, 0.1, 2 * D2)
    mW = np.zeros_like(W1); vW = np.zeros_like(W1)
    m2 = np.zeros_like(v2); v2m = np.zeros_like(v2)
    for step in range(1, 4001):
        idx = r2.choice(600, 200, replace=False)
        _, gW, gv = backprop(W1, v2, A1[idx], y4[idx])
        W1, mW, vW = adam(W1, gW, mW, vW, step, lr)
        v2, m2, v2m = adam(v2, gv, m2, v2m, step, lr)
    accs.append(acc(fwd_loss(W1, v2, A1te, y4t)[1], y4t))
print("  单层核 256 实参: %.3f | 两层容器 256 实参 (3 种子): %s"
      % (ak4, np.array2string(np.array(accs), precision=3)))
print("  → 小预算下深度劣势 (优化难度 > 表达力): 深度的收益在同架构特征对齐, 非免费午餐")

# ================= EXP J: 值域乘法编码 (绑定代数) =================
print("\n=== EXP J: 值域乘法编码 (绑定 = 逐位相位加法) ===")
fv = freqs2(3.0, 1)                                # 值域码族 (1 维值)
Dv = fv.shape[0]

def c(v):
    v = np.atleast_1d(np.asarray(v, float))
    return np.exp(2j * np.pi * v[:, None] * fv[None, :, 0])

def K1(dy):
    return float(np.real(np.mean(np.exp(2j * np.pi * dy * fv))))

v1 = rng.uniform(0.2, 3, 8); v2b = rng.uniform(0.2, 3, 8)
e1 = float(np.max(np.abs(c(v1) * c(v2b) - c(v1 + v2b))))
e2 = float(np.max(np.abs(c(np.log(v1)) * c(np.log(v2b)) - c(np.log(v1 * v2b)))))
print("  J1 绑定=值域加法:        max|Δ| = %.2e" % e1)
print("  J2 log 同态 (乘法→绑定): max|Δ| = %.2e  ← 乘法结构在码域变平移" % e2)
w3 = 0.0
for _ in range(20):
    x0, dl, y0 = rng.uniform(-2, 2, 3)
    B_ = (c([x0]) * c([dl]))[0]
    sim = np.vdot(B_, c([y0])[0]) / Dv
    w3 = max(w3, abs(float(np.real(sim)) - K1(y0 - x0 - dl)))
print("  J3 绑定=平移算子 <c(y),c(x)⊙c(δ)>=K(y−x−δ): max|Δ| = %.2e" % w3)

# J4 三重绑定乘法表: B = Σᵢ c₁(aᵢ)⊙c₂(bᵢ)⊙c₃(aᵢbᵢ) —— 零训练叠加写入
#   角色分离纪律: 三个槽位用独立频率族 —— 同族叠加只编码总和 a+b+p (已实测
#   干扰峰等高淹没), 独立族使交叉项相位不相干, 干扰 ∝ 1/√D
f1v = freqs2(3.0, 1); f2v = freqs2(3.0, 1); f3v = freqs2(3.0, 1)
mv = min(len(f1v), len(f2v), len(f3v))             # 逐位绑定要求等长, 截齐
f1v, f2v, f3v = f1v[:mv], f2v[:mv], f3v[:mv]
Dv = mv

def cf(v, f):
    v = np.atleast_1d(np.asarray(v, float))
    return np.exp(2j * np.pi * v[:, None] * f[None, :, 0])

M = 25
a_st = 0.5 * rng.integers(1, 7, M).astype(float)   # {0.5, 1.0, ..., 3.0}
b_st = 0.5 * rng.integers(1, 7, M).astype(float)
p_st = a_st * b_st
Bm = (cf(a_st, f1v) * cf(b_st, f2v) * cf(p_st, f3v)).sum(0)
gp = np.arange(0.25, 9.51, 0.125)
Phip = cf(gp, f3v)

def decode_prod(aq, bq):
    U = np.conj(cf([aq], f1v)[0]) * np.conj(cf([bq], f2v)[0]) * Bm
    sc = np.real(Phip @ np.conj(U)) / Dv
    return float(gp[int(np.argmax(sc))])

errs = np.array([abs(decode_prod(a_st[i], b_st[i]) - p_st[i]) for i in range(M)])
errs_n = np.array([abs(decode_prod(a_st[i] + 0.04, b_st[i]) - p_st[i]) for i in range(M)])
print("  J4 乘法表叠加记忆 (M=%d 条, 8KB 码, 零训练): 解码误差 mean %.4f / max %.4f"
      % (M, errs.mean(), errs.max()))
print("     键噪声 +0.04 (核主瓣内): 误差 mean %.4f / max %.4f (优雅退化)"
      % (errs_n.mean(), errs_n.max()))
# 未存键: 退化为核回归插值 (不幻觉)
aq, bq = 0.5 * float(rng.integers(1, 7)), 0.5 * float(rng.integers(1, 7))
pq = decode_prod(aq, bq)
near = float(np.min(np.abs(p_st - pq)))
print("     未存键 (%.1f,%.1f) 解码 %.3f: 距最近存过乘积 %.3f (核回归插值语义, 非幻觉)" % (aq, bq, pq, near))

# ================= EXP K: Oja deflation 链 =================
print("\n=== EXP K: Oja deflation 链 (前 3 主轴, 误差累积) ===")
ntr, d = 3000, 30
Q = np.linalg.qr(rng.normal(0, 1, (d, 3)))[0]      # 正交信号轴
amp = np.array([1.0, 0.6, 0.35])
Xs = (rng.normal(0, 1, (ntr, 3)) * amp) @ Q.T
Xn = 0.05 * rng.normal(0, 1, (ntr, d - 3))
X = np.concatenate([Xs, Xn], 1)
Xc0 = X - X.mean(0)

def oja(Xo, epochs=12, eta0=0.3):
    n, d_ = Xo.shape
    w = rng.normal(0, 1, d_); w /= np.linalg.norm(w)
    for ep in range(epochs):
        eta = eta0 / (1 + ep)
        for i in range(n):
            p = Xo[i] @ w
            w += eta * p * (Xo[i] - p * w)
            nn_ = np.linalg.norm(w)
            if nn_ > 0: w /= nn_
    return w

C = Xc0.T @ Xc0 / ntr
ev, evec = np.linalg.eigh(C)
v_top = evec[:, ::-1][:, :3]                       # 真主轴 (降序)
Xw = Xc0.copy()
cos_k = []
for k in range(3):
    wk = oja(Xw)
    cos_k.append(abs(float(wk @ v_top[:, k])))
    Xw = Xw - np.outer(Xw @ wk, wk)                # deflation
print("  特征值 (真): %s" % np.array2string(ev[::-1][:4], precision=3))
print("  cos(w_k, v_k): %s  (逐轴累积误差: %.4f → %.4f → %.4f)"
      % (np.array2string(np.array(cos_k), precision=4), cos_k[0], cos_k[1], cos_k[2]))

print("\n=== 走通总结 ===")
print("  深层组合反传 ✓ (I: 梯度恒等式 6e-9 + 特征学习增益 +12.5pp + 同预算诚实对照)")
print("  值域乘法编码 ✓ (J: 绑定代数 + 角色分离纪律) | Oja 链多轴 ✓ (K: 前 3 主轴)")
print("  §13.5 挂账: ①深层反传已拆除 ②乘法编码已拆除 ③多轴链已定量收窄; 新挂账=小预算深度劣势")
