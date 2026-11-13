# -*- coding: utf-8 -*-
# 最小复现: ctx 块更新是否单调? 数值 Hessian vs s1*Gmat 对账
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
import numpy as np
rng = np.random.default_rng(0)

N, V, D, L = 3000, 50, 32, 12
K = 8
BW = torch.linspace(0.5, 2.0, K)
# 随机窗口 (有重复字, 模拟真实)
CTX = torch.from_numpy(rng.integers(0, V, (N, K))).long()
REP = torch.from_numpy(rng.integers(0, V, (N, 4))).long()

E = torch.zeros(V, D)
E[:] = torch.from_numpy(rng.normal(0, 1/np.sqrt(D), (V, D)).astype(np.float32))
F = torch.zeros(V, D)
F[:] = torch.from_numpy(rng.normal(0, 1/np.sqrt(D), (V, D)).astype(np.float32))

def enc_h():
    h = torch.zeros(N, D)
    for i in range(K):
        h += E[CTX[:, i]]*BW[i]
    return h

def enc_g():
    g = torch.zeros(N, D)
    for j in range(4):
        g += F[REP[:, j]]
    return g

def loss():
    h = enc_h(); g = enc_g()
    M = Mmat
    return float(((h@M - g)**2).sum())

# M 闭式
def solve_M():
    h = enc_h(); g = enc_g()
    return torch.linalg.solve(h.T@h/N + 1e-2*torch.eye(D), h.T@g/N)

Mmat = torch.eye(D)
Mmat = solve_M()
L0 = loss()
print('初始 L:', round(L0, 4))

# s1 精确
s1 = torch.zeros(V)
for s in range(N):
    Wacc = torch.zeros(V)
    for i in range(K):
        Wacc[CTX[s, i]] += BW[i]
    s1 += Wacc**2

# 块更新 (与 v15 实现同构)
Gmat = Mmat@Mmat.T
b = s1[:, None]*E.clone()
h = enc_h(); g = enc_g()
R = g - h@Mmat
RM = R@Mmat.T
for i in range(K):
    b.index_add_(0, CTX[:, i], RM*BW[i])
lam = 0.01*float(Gmat.trace())/D
print('λrel=', round(lam, 5), ' Gmat 特征值范围:',
      round(float(torch.linalg.eigvalsh(Gmat).min()), 5),
      round(float(torch.linalg.eigvalsh(Gmat).max()), 5))
E_init = E.clone()
for c in range(V):
    Hc = s1[c]*(Gmat + lam*torch.eye(D)) + 1e-8*torch.eye(D)
    E[c] = torch.linalg.solve(Hc, b[c])
L1 = loss()
print('块更新后 L:', round(L1, 4), ' 单调?', L1 <= L0)

# 判据 0: autograd 三方对账
E.copy_(E_init)
c = 0
e0 = E[c].clone()

def block_loss(ec):
    old = E[c].clone()
    E[c] = ec
    v = loss()
    E[c] = old
    return v

def grad_auto():
    Er = E.clone().requires_grad_(True)
    h = torch.zeros(N, D)
    for i in range(K):
        h = h + Er[CTX[:, i]]*BW[i]
    g = enc_g()
    Lq = ((h@Mmat - g)**2).sum()
    Lq.backward()
    return Er.grad[c].detach()

gauto = grad_auto()
ga_negb = -b[c]
ga_theo = b[c] - s1[c]*(Gmat@e0)
def relerr(a, t):
    return round(float((a - t).norm()/(t.norm() + 1e-12)), 8)
print('autograd grad/2 [:5]:', [round(float(x), 4) for x in (gauto/2)[:5]])
print('rel err(-b vs auto):', relerr(ga_negb, gauto/2))
print('rel err(b-s1Ge0 vs auto):', relerr(ga_theo, gauto/2))
print('rel err(s1Ge0-b vs auto):', relerr(-ga_theo, gauto/2))

ga = gauto/2  # 以 autograd 为准

# 分解对账: 单独重算 RM 散射, 逐项核对
bs = torch.zeros(V, D)
for i in range(K):
    bs.index_add_(0, CTX[:, i], RM*BW[i])
print('rel err(bs vs auto):', relerr(bs[c], gauto/2))
print('rel err(bs+s1Ge0 vs auto):', relerr(bs[c] + s1[c]*(Gmat@e0), gauto/2))
print('rel err(bs+s1*(G-I)e0 vs auto):', relerr(bs[c] + s1[c]*((Gmat-torch.eye(D))@e0), gauto/2))
print('差向量范数:', round(float((bs[c] + s1[c]*(Gmat@e0) - gauto/2).norm()), 6),
      ' grad范数:', round(float((gauto/2).norm()), 6))
# 检查 s1 是否正确: 用 autograd 对 BW 加倍验证二次项
def grad_auto_scale(fac):
    Er = E.clone().requires_grad_(True)
    h = torch.zeros(N, D)
    for i in range(K):
        h = h + Er[CTX[:, i]]*(BW[i]*fac)
    g = enc_g()
    Lq = ((h@Mmat - g)**2).sum()
    Lq.backward()
    return Er.grad[c].detach()/2
g_f2 = grad_auto_scale(2.0)
print('BW 加倍后 grad/2 [:5]:', [round(float(x), 4) for x in g_f2[:5]])
print('原 grad/2 [:5]:', [round(float(x), 4) for x in (gauto/2)[:5]])

# 终审: 理论 grad/2 = s1*G*e0 - bs
print('rel err(auto vs s1Ge0-bs):', relerr(gauto/2, s1[c]*(Gmat@e0) - bs[c]))
print('rel err(auto-s1Ge0 vs -bs):', relerr(gauto/2 - s1[c]*(Gmat@e0), -bs[c]))

# 二次项数值验证: Hessian 对角 (中心二阶差分) vs 2*s1*G 对角
dH = torch.zeros(D)
for d in range(D):
    ep = e0.clone(); ep[d] += 1e-3
    em = e0.clone(); em[d] -= 1e-3
    dH[d] = (block_loss(ep) + block_loss(em) - 2*L0)/(1e-3**2)
print('数值 H 对角[:5]:', [round(float(x), 3) for x in dH[:5]])
print('理论 2*s1*G 对角[:5]:', [round(float(x), 3) for x in (2*s1[c]*torch.diag(Gmat))[:5]])

# 判据 2: 小步长单调性  E[c] = e0 - eta * grad (grad = 2*ga)
for eta in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
    E[c] = e0 - eta*2*ga
    L1 = loss()
    E[c] = e0
    print(f'eta={eta:g}: L={round(L1,4)} 单调? {L1 <= L0}')

# 判据 3: 终版块解 e* = H^{-1}(bs + s1*G*e0)  [= H^{-1}(b + s1*(G-I)*e0)]
Hc = s1[c]*(Gmat + lam*torch.eye(D)) + 1e-8*torch.eye(D)
E[c] = torch.linalg.solve(Hc, bs[c] + s1[c]*(Gmat@e0))
L_fix = loss()
print('终版块解 L:', round(L_fix, 4), ' 单调?', L_fix <= L0)
E[c] = e0

# 判据 4: 终版块解全 V 扫一遍 (Jacobi 风格, 全用旧 h)
E2 = E.clone()
for cc in range(V):
    E2[cc] = torch.linalg.solve(s1[cc]*(Gmat + lam*torch.eye(D)) + 1e-8*torch.eye(D),
                                bs[cc] + s1[cc]*(Gmat@E[cc]))
E = E2
L_all = loss()
print('全 V 终版块扫 L:', round(L_all, 4), ' 单调?', L_all <= L0)

# 判据 5: 真实训练场景 - 多轮 ALS (每轮重算 RM 散射), 看 L 收敛
print('--- 多轮 ALS 收敛 (阻尼扫描) ---')
for omega in (1.0, 0.5, 0.25, 0.1):
    E.copy_(E_init)
    Ls = []
    for it in range(5):
        h = enc_h(); g = enc_g()
        R = g - h@Mmat
        RM = R@Mmat.T
        bsx = torch.zeros(V, D)
        for i in range(K):
            bsx.index_add_(0, CTX[:, i], RM*BW[i])
        Enew = E.clone()
        for cc in range(V):
            Enew[cc] = torch.linalg.solve(s1[cc]*(Gmat + lam*torch.eye(D)) + 1e-8*torch.eye(D),
                                          bsx[cc] + s1[cc]*(Gmat@E[cc]))
        E.copy_(E + omega*(Enew - E))
        Ls.append(round(loss(), 2))
    print(f'omega={omega}: {Ls}')

# 判据 6: 双塔 (E_ctx + F_rep) 交替阻尼 ALS, M 固定, 5 轮
print('--- 双塔交替阻尼 ALS ---')
E.copy_(E_init)
K2 = 4
OM = 0.25
for it in range(6):
    # ctx 块
    h = enc_h(); g = enc_g()
    R = g - h@Mmat
    RM = R@Mmat.T
    bsx = torch.zeros(V, D)
    for i in range(K):
        bsx.index_add_(0, CTX[:, i], RM*BW[i])
    Enew = E.clone()
    for cc in range(V):
        Enew[cc] = torch.linalg.solve(s1[cc]*(Gmat + lam*torch.eye(D)) + 1e-8*torch.eye(D),
                                      bsx[cc] + s1[cc]*(Gmat@E[cc]))
    E.copy_(E + OM*(Enew - E))
    # rep 块 (F): grad/2(e0) = s2f*e0 - Sf, 块解 = (s2f*e0 - Sf)/(s2f+1)
    h = enc_h(); g = enc_g()
    R = g - h@Mmat
    s2f = torch.zeros(V)
    cntm = torch.zeros(V)
    for j in range(K2):
        cntm.zero_()
        cntm.index_add_(0, REP[:, j], torch.ones(N))
        s2f += cntm**2
    Sf = torch.zeros(V, D)
    for j in range(K2):
        Sf.index_add_(0, REP[:, j], R)
    Fnew = F.clone()
    act = s2f > 0
    Fnew[act] = (s2f[act, None]*F[act] - Sf[act])/(s2f[act, None] + 1.0)
    F.copy_(F + OM*(Fnew - F))
    print(f'轮 {it+1}: L={round(loss(), 4)}')
