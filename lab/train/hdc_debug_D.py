# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
rng = np.random.default_rng(23)
D = 16384
def rphase(n): return np.exp(1j*rng.uniform(0, 2*np.pi, size=(n, D)))

K = 100
keys_u = rphase(K)
codebook = rphase(8)
vidx = rng.integers(0, 8, size=K)
vals = codebook[vidx]
Mu = (keys_u.conj() * vals).sum(axis=0)
print("|Mu| 均值: %.3f (理论 ~sqrt(K)=%.1f)" % (np.abs(Mu).mean(), np.sqrt(K)))
i = 0
R = Mu * keys_u[i]
print("|R| 均值: %.3f" % np.abs(R).mean())
corr = np.abs(codebook @ R.conj()) / D
print("corr:", np.round(corr, 4), " 真值 idx:", vidx[i])
# 信号检查: R 与 v_i 的直接相关度
print("sim(R, v_true)=%.4f" % (np.abs(np.vdot(codebook[vidx[i]], R))/D))
# R 与 v_1 的理论: R = v_1 + noise → 逐项检查
term1 = keys_u[1].conj() * keys_u[0] * vals[1]
print("单条噪声项 |·| 均值: %.3f (应为1)" % np.abs(term1).mean())
