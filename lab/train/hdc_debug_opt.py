# -*- coding: utf-8 -*-
# 快测: 深层容器优化器攻坚 —— 诊断 + 三种原则性干预
#   O0 诊断: 逐层梯度范数 / tanh 饱和度 / train-test 背离
#   O1 深监督: 每个隐层挂辅助容器读出头 (直接梯度通路)
#   O2 AdamW 权重衰减: 针对「训练 1.0 / 测试 0.55」记忆化
#   O3 课程学习: 奇校验 2→3→4 (末位比特掩码递进)
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(113)

def freqs_fd(Fd, Dl, r_):
    return r_.uniform(-1, 1, (Dl, len(Fd))) * np.array(Fd)

def acc(yh, yt): return float(np.mean(np.sign(yh) == np.sign(yt)))

nb = rng.integers(0, 2, (600, 4)); s = 2.0*nb-1
X4 = s + 0.1*rng.normal(0,1,(600,4)); y4 = np.prod(s,1)
nb = rng.integers(0, 2, (500, 4)); s4t = 2.0*nb-1
X4t = s4t + 0.1*rng.normal(0,1,(500,4)); y4t = np.prod(s4t,1)

def phi_feat(h, f, Dl):
    ph = np.exp(2j*np.pi*h@f.T)
    return ph, np.concatenate([ph.real, -ph.imag],1)/np.sqrt(Dl)

def make(Ds=(32,32), Hs=(8,8), Do=48, seed=7, F=2.0, use_aux=False):
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([F]*4, Ds[0], r2); W1 = r2.normal(0,1,(Hs[0],2*Ds[0]))/np.sqrt(2*Ds[0])
    f2 = freqs_fd([F]*Hs[0], Ds[1], r2); W2 = r2.normal(0,1,(Hs[1],2*Ds[1]))/np.sqrt(2*Ds[1])
    fo = freqs_fd([F]*Hs[1], Do, r2); vo = r2.normal(0,0.3,2*Do)
    net = dict(f1=f1,W1=W1,f2=f2,W2=W2,fo=fo,vo=vo,D1=Ds[0],D2=Ds[1],Do=Do)
    if use_aux:
        net['fa1'] = freqs_fd([F]*Hs[0], 32, r2); net['va1'] = r2.normal(0,0.3,64)
        net['fa2'] = freqs_fd([F]*Hs[1], 32, r2); net['va2'] = r2.normal(0,0.3,64)
    return net

def fwd(net, X, use_aux=False):
    ph1,A1 = phi_feat(X, net['f1'], net['D1']); u1 = A1@net['W1'].T; h1 = np.tanh(u1)
    ph2,A2 = phi_feat(h1, net['f2'], net['D2']); u2 = A2@net['W2'].T; h2 = np.tanh(u2)
    pho,Ao = phi_feat(h2, net['fo'], net['Do'])
    out = Ao@net['vo']
    C = dict(ph1=ph1,A1=A1,u1=u1,h1=h1,ph2=ph2,A2=A2,u2=u2,h2=h2,pho=pho,Ao=Ao)
    if use_aux:
        pha1,Aa1 = phi_feat(h1, net['fa1'], 32); aux1 = Aa1@net['va1']; C['Aa1']=Aa1; C['pha1']=pha1
        pha2,Aa2 = phi_feat(h2, net['fa2'], 32); aux2 = Aa2@net['va2']; C['Aa2']=Aa2; C['pha2']=pha2
        C['out'] = out; C['aux1']=aux1; C['aux2']=aux2
    return out, C

def ident(ph, f, wc, Dl):
    """恒等式: ∂Re⟨w,φ⟩/∂h_k = -(2π/√D)·Im⟨w⊙f, φ⟩  (wc 为复向量)"""
    return -(2*np.pi/np.sqrt(Dl))*np.imag(ph@(wc[:,None]*f))

def bwd(net, y, C, use_aux=False, aux_w=0.5):
    n = len(y)
    out = C['out'] if use_aux else C['Ao']@net['vo']
    g = 2*(out-y)/n
    Do = net['Do']; wco = net['vo'][:Do]+1j*net['vo'][Do:]
    g_vo = C['Ao'].T@g
    dLdh2 = g[:,None]*ident(C['pho'], net['fo'], wco, Do)
    g_va2 = g_va1 = None
    if use_aux:
        ga2 = 2*aux_w*(C['aux2']-y)/n
        g_va2 = C['Aa2'].T@ga2
        wa2 = net['va2'][:32]+1j*net['va2'][32:]
        dLdh2 = dLdh2 + ga2[:,None]*ident(C['pha2'], net['fa2'], wa2, 32)
        ga1 = 2*aux_w*(C['aux1']-y)/n
        g_va1 = C['Aa1'].T@ga1
    dLdu2 = dLdh2*(1-C['h2']**2)
    g_W2 = dLdu2.T@C['A2']
    wc2 = net['W2'][:,:net['D2']]+1j*net['W2'][:,net['D2']:]
    T2 = np.einsum('nd,hd,dk->nhk', C['ph2'], wc2, net['f2'])
    dLdh1 = np.einsum('ni,nik->nk', dLdu2, -(2*np.pi/np.sqrt(net['D2']))*np.imag(T2))
    if use_aux:
        wa1 = net['va1'][:32]+1j*net['va1'][32:]
        dLdh1 = dLdh1 + ga1[:,None]*ident(C['pha1'], net['fa1'], wa1, 32)
    dLdu1 = dLdh1*(1-C['h1']**2)
    g_W1 = dLdu1.T@C['A1']
    return [g_W1, g_W2], [g_vo, g_va2, g_va1]

def train3(steps=2500, lr=0.02, mb=200, use_aux=False, wd=0.0, curriculum=False,
           tag="", diag=False):
    net = make(use_aux=use_aux)
    ps = [net['W1'], net['W2'], net['vo']] + ([net['va1'], net['va2']] if use_aux else [])
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    stages = ([(0, 700, X4[:, :2].copy(), y4), (700, 1400, X4[:, :3].copy(), y4),
               (1400, steps, X4, y4)] if curriculum else [(0, steps, X4, y4)])
    # 课程数据: 末位比特置 +1 (真实 4 维输入上的掩码递进)
    if curriculum:
        Xp2 = X4.copy(); Xp2[:, 2:] = 1.0
        Xp3 = X4.copy(); Xp3[:, 3:] = 1.0
        stages = [(0, 700, Xp2, y4), (700, 1400, Xp3, y4), (1400, steps, X4, y4)]
    best, best_at = 0.0, 0
    for step in range(1, steps+1):
        Xa, ya = next((sX, sy) for a, b, sX, sy in stages if a < step <= b)
        idx = np.arange(600) if mb >= 600 else np.random.default_rng(step).choice(600, mb, replace=False)
        out, C = fwd(net, Xa[idx], use_aux)
        gW, gV = bwd(net, ya[idx], C, use_aux)
        gs = gW + gV
        for i, p in enumerate(ps):
            if p is None: continue
            gg = gs[i]
            m[i] = 0.9*m[i]+0.1*gg; v[i] = 0.999*v[i]+0.001*gg*gg
            ps[i] = ps[i] - lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8) + wd*p)
        net['W1'], net['W2'], net['vo'] = ps[0], ps[1], ps[2]
        if use_aux: net['va1'], net['va2'] = ps[3], ps[4]
        if step % 100 == 0 or step == steps:
            otr, _ = fwd(net, X4, use_aux); ote, _ = fwd(net, X4t, use_aux)
            a = acc(ote, y4t)
            if a > best: best, best_at = a, step
            if diag and step % 300 == 0:
                gWd, gVd = bwd(net, ya[idx], C, use_aux)
                sat1 = float(np.mean(np.abs(C['h1'])>0.9)); sat2 = float(np.mean(np.abs(C['h2'])>0.9))
                print("    d step %d: |gW1| %.2e |gW2| %.2e |gv| %.2e 饱和 h1 %.2f h2 %.2f train %.2f"
                      % (step, np.linalg.norm(gWd[0]), np.linalg.norm(gWd[1]),
                         np.linalg.norm(gVd[0]), sat1, sat2, acc(otr, y4)))
    otr, _ = fwd(net, X4, use_aux); ote, _ = fwd(net, X4t, use_aux)
    print("  %-32s best %.3f @%d | 末 train %.3f test %.3f"
          % (tag, best, best_at, acc(otr, y4), acc(ote, y4t)))
    return best

print("=== O0 诊断 (基线, mb=200) ===")
train3(1200, diag=True, tag="base")
print("\n=== O1-O3 干预扫描 ===")
train3(2500, tag="base 2500")
train3(2500, use_aux=True, tag="O1 深监督")
train3(2500, wd=1e-3, tag="O2 wd 1e-3")
train3(2500, wd=1e-2, mb=600, tag="O2 wd 1e-2 全批")
train3(2500, curriculum=True, tag="O3 课程 2→3→4")
train3(2500, use_aux=True, wd=1e-3, tag="O1+O2")
train3(2500, use_aux=True, curriculum=True, tag="O1+O3")
train3(2500, use_aux=True, wd=1e-3, curriculum=True, tag="O1+O2+O3")

# ---- O4: 交替闭式头 —— 输出读出每步岭回归精确解, 只对隐层 SGD ----
def train3_alt(steps=2500, lr=0.02, mb=200, lam=1e-3, wd=0.0, tag=""):
    net = make(use_aux=False)
    ps = [net['W1'], net['W2']]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    best, best_at = 0.0, 0
    for step in range(1, steps+1):
        idx = np.arange(600) if mb >= 600 else np.random.default_rng(step).choice(600, mb, replace=False)
        out, C = fwd(net, X4[idx], False)
        # 闭式头: 给定 h2 特征, 输出读出 = 岭回归精确解
        Aob = C['Ao']; yb = y4[idx]
        G = Aob.T@Aob/len(yb) + lam*np.eye(2*net['Do'])
        net['vo'] = np.linalg.solve(G, Aob.T@yb/len(yb))
        out = Aob@net['vo']
        # 只对隐层 SGD (头已最优, 隐层面对稳定回归目标)
        g = 2*(out-yb)/len(yb)
        Do = net['Do']; wco = net['vo'][:Do]+1j*net['vo'][Do:]
        dLdh2 = g[:,None]*ident(C['pho'], net['fo'], wco, Do)
        dLdu2 = dLdh2*(1-C['h2']**2)
        g_W2 = dLdu2.T@C['A2']
        wc2 = net['W2'][:,:net['D2']]+1j*net['W2'][:,net['D2']:]
        T2 = np.einsum('nd,hd,dk->nhk', C['ph2'], wc2, net['f2'])
        dLdh1 = np.einsum('ni,nik->nk', dLdu2, -(2*np.pi/np.sqrt(net['D2']))*np.imag(T2))
        dLdu1 = dLdh1*(1-C['h1']**2)
        g_W1 = dLdu1.T@C['A1']
        gs = [g_W1, g_W2]
        for i, p in enumerate(ps):
            m[i] = 0.9*m[i]+0.1*gs[i]; v[i] = 0.999*v[i]+0.001*gs[i]*gs[i]
            ps[i] = ps[i]*(1-lr*wd) - lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8))
        net['W1'], net['W2'] = ps[0], ps[1]
        if step % 100 == 0:
            ote, _ = fwd(net, X4t, False)
            a = acc(ote, y4t)
            if a > best: best, best_at = a, step
    otr, _ = fwd(net, X4, False); ote, _ = fwd(net, X4t, False)
    print("  %-32s best %.3f @%d | 末 train %.3f test %.3f"
          % (tag, best, best_at, acc(otr, y4), acc(ote, y4t)))
    return best

print("\n=== O4 交替闭式头 ===")
train3_alt(2500, tag="O4 mb=200 λ=1e-3")
train3_alt(2500, mb=600, tag="O4 全批 λ=1e-3")
train3_alt(2500, mb=600, lam=1e-2, tag="O4 全批 λ=1e-2")
train3_alt(2500, mb=600, lam=1e-2, wd=1e-3, tag="O4 全批 λ=1e-2 wd")

# ---- O5: 无限样本增强 —— 每步从 16 顶点重新采样 (记忆化无固定点可抓) ----
VERT = np.array([[a,b,c,d] for a in (-1,1) for b in (-1,1) for c in (-1,1) for d in (-1,1)], float)

def train3_aug(steps=2500, lr=0.02, mb=200, noise=0.15, wd=0.0, tag=""):
    net = make(use_aux=False)
    ps = [net['W1'], net['W2'], net['vo']]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    best, best_at = 0.0, 0
    r2 = np.random.default_rng(11)
    for step in range(1, steps+1):
        vi = VERT[r2.integers(0, 16, mb)]
        Xb = vi + noise*r2.normal(0, 1, (mb, 4))
        yb = np.prod(vi, 1)                       # 标签用干净顶点 (噪声不变性)
        out, C = fwd(net, Xb, False)
        gW, gV = bwd(net, yb, C, False)
        gs = gW + gV
        for i, p in enumerate(ps):
            m[i] = 0.9*m[i]+0.1*gs[i]; v[i] = 0.999*v[i]+0.001*gs[i]*gs[i]
            ps[i] = ps[i]*(1-lr*wd) - lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8))
        net['W1'], net['W2'], net['vo'] = ps
        if step % 100 == 0:
            ote, _ = fwd(net, X4t, False)
            a = acc(ote, y4t)
            if a > best: best, best_at = a, step
    otr, _ = fwd(net, X4, False); ote, _ = fwd(net, X4t, False)
    print("  %-32s best %.3f @%d | 末 train %.3f test %.3f"
          % (tag, best, best_at, acc(otr, y4), acc(ote, y4t)))
    return best

print("\n=== O5 无限样本增强 ===")
train3_aug(2500, tag="O5 aug σ=0.15")
train3_aug(2500, noise=0.3, tag="O5 aug σ=0.3")
train3_aug(2500, noise=0.15, wd=1e-3, tag="O5 aug + wd")
train3_aug(4000, noise=0.15, tag="O5 aug 4000 步")
