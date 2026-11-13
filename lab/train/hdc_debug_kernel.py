# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(1)
D = 4096
f = rng.uniform(-1.25, 1.25, size=(D, 2))
f = f[np.any(np.abs(f) > 0.05, axis=1)]
print("D_eff =", len(f), "| f range:", f.min(0), f.max(0))
d = np.array([0.2, 0.0])
print("kernel(0.2,0) =", np.cos(2 * np.pi * (f @ d)).mean(),
      "| analytic sinc^2:", float(np.sinc(2 * 1.25 * 0.2)) ** 2)
d2 = np.array([0.05, 0.03])
print("kernel(0.05,0.03) =", np.cos(2 * np.pi * (f @ d2)).mean())

xs = rng.integers(0, 2, (600, 2)).astype(float) * 2 - 1 + 0.1 * rng.normal(0, 1, (600, 2))
xte = rng.integers(0, 2, (400, 2)).astype(float) * 2 - 1 + 0.1 * rng.normal(0, 1, (400, 2))
dd = np.sqrt(((xte[0][None] - xs) ** 2).sum(1))
print("nearest train dist to xte[0]:", dd.min())
