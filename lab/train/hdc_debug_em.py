# -*- coding: utf-8 -*-
# 诊断: EXP G 的 EM 是真局部最优还是指标错配?
#   1) 逐轮解码原型位置, 看轨迹 (塌缩? 停在边界?)
#   2) 按最优匹配 (3! 排列穷举) 重新计算恢复误差
#   3) 看责任矩阵结构 (是否两原型共享同一簇)
import numpy as np
import sys, itertools

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(53)
D = 4096

def freqs2(F, d, D_=D):
    f = rng.uniform(-F, F, size=(D_, d))
    return f[np.any(np.abs(f) > 0.05, axis=1)]

means_true = np.array([[0.25, 0.25], [0.75, 0.30], [0.50, 0.80]])
ng = 2000
who = rng.integers(0, 3, ng)
xg = means_true[who] + 0.05 * rng.normal(0, 1, (ng, 2))
fg = freqs2(4, 2)
phi = np.exp(2j * np.pi * xg @ fg.T)

# 真簇中心的码 (无噪声), 作参照
phi_true = np.exp(2j * np.pi * means_true @ fg.T)

# --- 与主脚本相同的 k-means++ 初始化 ---
P = np.empty((3, phi.shape[1]), dtype=complex)
P[0] = phi[rng.integers(0, ng)]
for k in range(1, 3):
    Sx = np.max(np.real(phi @ np.conj(P[:k]).T) / D, axis=1)
    P[k] = phi[int(np.argmin(Sx))]

# 初始原型落在哪个簇?
print("初始原型到真中心相似度:")
S0 = np.real(phi_true @ np.conj(P.T)) / D
print(np.array2string(S0, precision=3))
print("初始原型归簇:", np.argmax(S0, 0))

beta = 15.0
print("\nEM 轨迹 (每 5 轮解码一次):")
gx, gy_ = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120))
grid = np.stack([gx.ravel(), gy_.ravel()], 1)

def decode_all(P):
    pos = np.zeros((3, 2)); sc_best = np.full(3, -1e9)
    for s in range(0, len(grid), 2000):
        e = min(s + 2000, len(grid))
        Phi_c = np.exp(2j * np.pi * grid[s:e] @ fg.T)
        sc = np.real(Phi_c @ np.conj(P.T)) / D          # [g, 3]
        j = np.argmax(sc, 0)
        v = sc[j, np.arange(3)]
        upd = v > sc_best
        pos[upd] = grid[s:e][j[upd]]
        sc_best = np.maximum(sc_best, v)
    return pos

hist = []
for it in range(40):
    S = np.real(phi @ np.conj(P.T)) / D
    r = np.exp(beta * (S - S.max(1, keepdims=True)))
    r /= r.sum(1, keepdims=True)
    P = (r.T @ phi) / r.sum(0)[:, None]
    if it % 5 == 0 or it == 39:
        pos = decode_all(P)
        # 每个原型的责任总质量 (看塌缩)
        S2 = np.real(phi @ np.conj(P.T)) / D
        r2 = np.exp(beta * (S2 - S2.max(1, keepdims=True)))
        r2 /= r2.sum(1, keepdims=True)
        mass = r2.sum(0)
        # 原型范数
        nrm = np.abs(P).mean(1) / np.sqrt(D)
        print("  it%02d pos=%s mass=%s |P|=%s" % (
            it, np.array2string(pos, precision=3),
            np.array2string(mass, precision=3),
            np.array2string(nrm, precision=3)))
        hist.append(pos.copy())

# --- 最优匹配误差 (3! = 6 排列穷举) ---
pos_final = decode_all(P)
print("\n最终原型位置:", np.array2string(pos_final, precision=4))
print("按索引硬对误差:", np.array2string(np.linalg.norm(pos_final - means_true, axis=1), precision=4))
best_perm, best_err = None, 1e9
for perm in itertools.permutations(range(3)):
    err = float(np.linalg.norm(pos_final[list(perm)] - means_true))
    if err < best_err:
        best_err, best_perm = err, perm
print("最优匹配 %s 误差: %.4f" % (best_perm, best_err))

# 责任矩阵: 簇视角 (每簇数据对各原型的平均责任)
S2 = np.real(phi @ np.conj(P.T)) / D
r2 = np.exp(beta * (S2 - S2.max(1, keepdims=True)))
r2 /= r2.sum(1, keepdims=True)
print("\n按真簇分组的平均责任 (行=真簇, 列=原型):")
for c in range(3):
    print("  簇%d: %s" % (c, np.array2string(r2[who == c].mean(0), precision=3)))
