# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(1)
D = 4096

def freqs2(F, d, D_=D):
    f = rng.uniform(-F, F, size=(D_, d))
    return f[np.any(np.abs(f) > 0.05, axis=1)]

def make_A(x, f):
    ang = 2 * np.pi * x @ f.T
    return np.concatenate([np.cos(ang), -np.sin(ang)], axis=1) / np.sqrt(D)

f = freqs2(1.25, 2)
xs = rng.integers(0, 2, (600, 2)).astype(float) * 2 - 1 + 0.1 * rng.normal(0, 1, (600, 2))
xte = rng.integers(0, 2, (400, 2)).astype(float) * 2 - 1 + 0.1 * rng.normal(0, 1, (400, 2))
A1, A2 = make_A(xs, f), make_A(xte, f)
S = A2 @ A1.T / D
print("max-sim quantiles:", np.quantile(S.max(1), [0.1, 0.5, 0.9]))
print("kernel shape check: A1[0]·A1[0] =", float(A1[0] @ A1[0] / D))
dd = np.sqrt(((xte[0][None] - xs) ** 2).sum(1))
jmin = int(np.argmin(dd))
print("nearest dist =", float(dd[jmin]), "| S[0, jmin] =", float(S[0, jmin]))
