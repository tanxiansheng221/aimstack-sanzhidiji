# -*- coding: utf-8 -*-
import numpy as np
rng = np.random.default_rng(11)
D = 16384
def rphase(n): return np.exp(1j*rng.uniform(0, 2*np.pi, size=(n, D)))
def sim(a, b): return float(np.abs(np.vdot(a, b)) / D)

NF, NC = 24, 8
F = rphase(NF); C = rphase(NC)
NR = 30
rules = [(tuple(sorted(rng.choice(NF, size=2, replace=False))), int(rng.integers(0, NC))) for _ in range(NR)]
def bundle(idxs):
    v = F[list(idxs)].sum(axis=0)
    return v / np.abs(v)
A = [bundle(p) for p, _ in rules]
M = sum(np.conj(A[r]) * C[c] for r, (p, c) in enumerate(rules))

# 精确查询调试
p, c_true = rules[5]
Q = bundle(p)
R = M * Q
corr = np.abs(C @ R.conj()) / D
print("corr 向量:", np.round(corr, 4))
s = np.sort(corr)[::-1]
print("top1=%.4f top2=%.4f std=%.4f z=%.2f  真类别=%d argmax=%d" %
      (s[0], s[1], np.std(corr), (s[0]-s[1])/(np.std(corr)+1e-9), c_true, int(np.argmax(corr))))

# 检查信号项: conj(A_k)⊙Q 是否真是全 1
k = 5
chk = np.conj(A[k]) * Q
print("|conj(A_k)⊙Q| 均值=%.4f (应为1), 相位方差=%.4f (应为0)" % (np.abs(chk).mean(), np.var(np.angle(chk))))
# 检查 M 中信号项幅度
term = np.conj(A[k]) * C[c_true]
print("M 中信号项与 C_k 相关度: %.4f" % sim(term, C[c_true]))
# 整体 M 的模
print("|M| 均值=%.4f (30 项叠加, 理论 ~sqrt(30)=%.1f)" % (np.abs(M).mean(), np.sqrt(30)))
