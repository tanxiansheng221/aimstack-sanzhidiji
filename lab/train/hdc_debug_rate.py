# -*- coding: utf-8 -*-
# hdc_debug_rate.py — 解剖 A2 速率背离: v_min 是不是真的最小模? 单模衰减实测
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(41)
D = 8192

def freqs(F, D_=D):
    side = rng.integers(0, 2, size=D_) * 2 - 1
    mag = rng.integers(1, F + 1, size=D_)
    return (side * mag).astype(float)

def make_A(x, f):
    ang = 2 * np.pi * np.outer(x, f)
    return np.concatenate([np.cos(ang), -np.sin(ang)], axis=1) / D

HARM = [(0.8, 1, 0.3), (0.5, 3, 1.1), (0.3, 5, 2.0)]
def g(x):
    return sum(a * np.sin(2 * np.pi * h * x + p) for a, h, p in HARM)

n, F, sigma, lam = 300, 32, 0.3, 1e-6
xs = rng.uniform(0, 1, n)
y = g(xs) + sigma * rng.normal(0, 1, n)
f = freqs(F)
A = make_A(xs, f)
K = A @ A.T / n
eigs, vecs = np.linalg.eigh(K)
print("谱: 最小5个 =", np.array2string(eigs[:5], precision=3), "| 最大3个 =",
      np.array2string(eigs[-3:], precision=3))
mu_min = eigs[0]
mu_max = eigs[-1]

# 特征关系检查: H(A^T u) = (μ+λ) A^T u ?
u0 = vecs[:, 0]
v0 = A.T @ u0
Hv = A.T @ (A @ v0) / n + lam * v0
ray = float((v0 @ Hv) / (v0 @ v0))
resid = np.linalg.norm(Hv - ray * v0) / np.linalg.norm(Hv)
print("H 作用在 v0=A^T u_min: Rayleigh 商 = %.6e (理论 μ_min+λ = %.6e), 残差 %.2e"
      % (ray, mu_min + lam, resid))

# 单模衰减实测
eta = 0.3 / (mu_max + lam)
theta_r = A.T @ np.linalg.solve(K + lam * np.eye(n), y / n)
vn = v0 / np.linalg.norm(v0)
theta_t = theta_r + 1e-6 * vn
print("\n单模衰减 (每20步 ||θt-θr||):")
for t in range(0, 241, 20):
    if t > 0:
        for _ in range(20):
            theta_t = theta_t - eta * (A.T @ (A @ theta_t - y) / n + lam * theta_t)
    print("  t=%3d  ||θt-θr||=%.3e  (理论 ε·%.4f^t = %.3e)"
          % (t, np.linalg.norm(theta_t - theta_r), abs(1 - eta * (mu_min + lam)),
             1e-6 * abs(1 - eta * (mu_min + lam)) ** t))
