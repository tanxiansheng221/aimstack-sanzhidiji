# -*- coding: utf-8 -*-
# 快测: 深层容器优化器——残差层插入 + 闭式头 (走通新假设)
#   P0 几何诊断: 训练后的 h1 空间里 16 个顶点簇间距 vs 层2核宽 0.25
#   P1a 暖体插入: 练好的两层 W1 + 残差层2(h2=h1+tanh(u2), W2小) + 闭式头 → 继续训
#   P1b 随机体插入: 同架构, W1 随机
#   P1c 稠密: 层2输入 [h1, x] + 闭式头
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
VERT = np.array([[a,b,c,d] for a in (-1,1) for b in (-1,1) for c in (-1,1) for d in (-1,1)], float)

def phi_feat(h, f, Dl):
    ph = np.exp(2j*np.pi*h@f.T)
    return ph, np.concatenate([ph.real, -ph.imag],1)/np.sqrt(Dl)

def ident(ph, f, wc, Dl):
    return -(2*np.pi/np.sqrt(Dl))*np.imag(ph@(wc[:,None]*f))

def train2l(steps=4000, lr=0.02, mb=200, seed=7):
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([2.0]*4, 40, r2); W1 = r2.normal(0,1,(10,80))/9.0
    fo = freqs_fd([2.0]*10, 56, r2); vo = r2.normal(0,0.3,112)
    def fwd(X):
        ph1,A1 = phi_feat(X, f1, 40); h1 = np.tanh(A1@W1.T)
        pho,Ao = phi_feat(h1, fo, 56)
        return Ao@vo, (ph1,A1,h1,pho,Ao)
    def bwd(y,out,C):
        n=len(y); g=2*(out-y)/n
        g_vo = C[4].T@g
        dLdh1 = g[:,None]*ident(C[3], fo, vo[:56]+1j*vo[56:], 56)
        dLdu1 = dLdh1*(1-C[2]**2)
        return [dLdu1.T@C[1]], g_vo
    m=[np.zeros_like(W1),np.zeros_like(vo)]; v=[np.zeros_like(W1),np.zeros_like(vo)]
    ps=[W1,vo]
    for step in range(1,steps+1):
        idx = r2.choice(600, mb, replace=False)
        out,C = fwd(X4[idx])
        gW,gv = bwd(y4[idx],out,C)
        for i,gg in enumerate([gW[0],gv]):
            m[i]=0.9*m[i]+0.1*gg; v[i]=0.999*v[i]+0.001*gg*gg
            ps[i]=ps[i]-lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8))
        W1,vo = ps
    a2 = acc(fwd(X4t)[0], y4t)
    return dict(f1=f1,W1=W1,fo=fo,vo=vo), a2

def make3(net2=None, seed=17, res=True, dense=False, w2_scale=0.05, Do=48):
    r2 = np.random.default_rng(seed)
    h1dim = net2['W1'].shape[0] if net2 is not None else 10
    D1 = net2['f1'].shape[0] if net2 is not None else 40
    d2 = h1dim + (4 if dense else 0)
    net = dict(D1=D1, D2=48, Do=Do, res=res, dense=dense, h1dim=h1dim)
    if net2 is not None:
        net['f1'], net['W1'] = net2['f1'], net2['W1']
    else:
        net['f1'] = freqs_fd([2.0]*4, D1, r2)
        net['W1'] = r2.normal(0,1,(h1dim, 2*D1))/np.sqrt(2*D1)
    net['f2'] = freqs_fd([2.0]*d2, 48, r2)
    net['W2'] = r2.normal(0, w2_scale, (h1dim, 2*net['D2']))
    net['fo'] = freqs_fd([2.0]*h1dim, Do, r2)
    net['vo'] = np.zeros(2*Do)
    return net

def fwd3(net, X):
    ph1,A1 = phi_feat(X, net['f1'], net['D1']); h1 = np.tanh(A1@net['W1'].T)
    x2 = np.concatenate([h1, X],1) if net['dense'] else h1
    ph2,A2 = phi_feat(x2, net['f2'], net['D2']); u2 = A2@net['W2'].T
    t2 = np.tanh(u2); h2 = h1 + t2 if net['res'] else t2
    pho,Ao = phi_feat(h2, net['fo'], net['Do'])
    return Ao@net['vo'], dict(ph1=ph1,A1=A1,h1=h1,ph2=ph2,A2=A2,u2=u2,t2=t2,h2=h2,pho=pho,Ao=Ao)

def bwd3(net, y, C, out):
    n=len(y); g=2*(out-y)/n
    Do=net['Do']; wco=net['vo'][:Do]+1j*net['vo'][Do:]
    dLdh2 = g[:,None]*ident(C['pho'], net['fo'], wco, Do)
    dLdu2 = dLdh2*(1-C['t2']**2)
    g_W2 = dLdu2.T@C['A2']
    wc2 = net['W2'][:,:net['D2']]+1j*net['W2'][:,net['D2']:]
    T2 = np.einsum('nd,hd,dk->nhk', C['ph2'], wc2, net['f2'])
    dLdh1 = np.einsum('ni,nik->nk', dLdu2, -(2*np.pi/np.sqrt(net['D2']))*np.imag(T2))
    dLdh1 = dLdh1[:, :net['h1dim']]            # dense 时 x2=[h1,x], 只取 h1 分量
    if net['res']: dLdh1 = dLdh1 + dLdh2
    dLdu1 = dLdh1*(1-C['h1']**2)
    g_W1 = dLdu1.T@C['A1']
    return [g_W1, g_W2]

def closed_head(net, C, y, lam=1e-4):
    Ao = C['Ao']
    G = Ao.T@Ao/len(y) + lam*np.eye(2*net['Do'])
    net['vo'] = np.linalg.solve(G, Ao.T@y/len(y))
    return float(np.mean((Ao@net['vo']-y)**2))

def adam_up(p, g, m, v, t, lr):
    m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
    return p-lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8), m, v

def grow_train(net, steps=1500, lr=0.005, mb=600, lam=1e-4):
    ps = [net['W1'], net['W2']]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    best = 0.0
    for step in range(1, steps+1):
        idx = np.arange(600) if mb>=600 else np.random.default_rng(step).choice(600, mb, replace=False)
        out, C = fwd3(net, X4[idx])
        closed_head(net, C, y4[idx], lam)
        out = C['Ao']@net['vo']
        gW = bwd3(net, y4[idx], C, out)
        for i in range(2):
            ps[i], m[i], v[i] = adam_up(ps[i], gW[i], m[i], v[i], step, lr)
        net['W1'], net['W2'] = ps[0], ps[1]
        if step % 100 == 0:
            best = max(best, acc(fwd3(net, X4t)[0], y4t))
    return best, acc(fwd3(net, X4t)[0], y4t)

# ---- P0: 几何诊断 ----
print("=== P0: h1 空间顶点簇几何 (用 P1a 暖体训练后的 W1) ===")
net2, acc2 = train2l()
print("  两层基线 (3000 步): test %.3f" % acc2)
ph1v, A1v = phi_feat(VERT, net2['f1'], net2['f1'].shape[0])
h1v = np.tanh(A1v@net2['W1'].T)
dmat = np.linalg.norm(h1v[:,None,:]-h1v[None,:,:], axis=2)
np.fill_diagonal(dmat, 1e9)
print("  16 顶点中心在 h1 空间: 最近邻距 mean %.3f min %.3f (层2核宽 0.25, 层1核宽 0.25)"
      % (float(np.mean(np.min(dmat,1))), float(np.min(np.min(dmat,1)))))

# ---- P1a: 暖体插入 (W2 scale 扫描: 残差抖动 vs 核几何衰减) ----
print("\n=== P1: 残差层插入 + 闭式头 ===")
# 对账 0: 两层体自己的特征 φ2(h1) 上, GD 头 vs 闭式头
_, C2 = None, None
ph1v_, A1v_ = phi_feat(X4, net2['f1'], net2['f1'].shape[0])
h1_tr = np.tanh(A1v_@net2['W1'].T)
pho2, Ao2 = phi_feat(h1_tr, net2['fo'], net2['fo'].shape[0])
G2 = Ao2.T@Ao2/600 + 1e-4*np.eye(2*net2['fo'].shape[0])
vo2r = np.linalg.solve(G2, Ao2.T@y4/600)
ph1t_, A1t_ = phi_feat(X4t, net2['f1'], net2['f1'].shape[0])
h1t_ = np.tanh(A1t_@net2['W1'].T)
_, Ao2t = phi_feat(h1t_, net2['fo'], net2['fo'].shape[0])
print("  对账0 同特征族: GD 头 %.3f | 闭式头 %.3f (GD 头=vo 逐点同前向)"
      % (acc(Ao2t@net2['vo'], y4t), acc(Ao2t@vo2r, y4t)))
for ws in (0.003, 0.01, 0.03):
    net_w = make3(net2, res=True, w2_scale=ws)
    out, C = fwd3(net_w, X4)
    closed_head(net_w, C, y4)
    out2, _ = fwd3(net_w, X4)
    print("  P1a 插入 w2_scale=%.3f: test %.3f (train mse %.4f)"
          % (ws, acc(fwd3(net_w, X4t)[0], y4t), float(np.mean((out2-y4)**2))))
net_a = make3(net2, res=True, w2_scale=0.01)
out, C = fwd3(net_a, X4)
tr_mse = closed_head(net_a, C, y4)
a_ins = acc(fwd3(net_a, X4t)[0], y4t)
print("  P1a 暖体: 插入后(闭式头, 零梯度步) test %.3f (train mse %.4f) [两层体 %.3f]"
      % (a_ins, tr_mse, acc2))
best_a, fin_a = grow_train(net_a)
print("  P1a 暖体: 继续训 1500 步 → best %.3f 末 %.3f" % (best_a, fin_a))

# ---- P1b: 随机体插入 ----
net_b = make3(None, res=True, w2_scale=0.01)
out, C = fwd3(net_b, X4)
closed_head(net_b, C, y4)
a_insb = acc(fwd3(net_b, X4t)[0], y4t)
best_b, fin_b = grow_train(net_b)
print("  P1b 随机体: 插入后 %.3f → 继续训 best %.3f 末 %.3f" % (a_insb, best_b, fin_b))

# ---- P1c: 稠密 [h1, x] ----
net_c = make3(None, res=True, dense=True, w2_scale=0.01)
out, C = fwd3(net_c, X4)
closed_head(net_c, C, y4)
a_insc = acc(fwd3(net_c, X4t)[0], y4t)
best_c, fin_c = grow_train(net_c)
print("  P1c 稠密: 插入后 %.3f → 继续训 best %.3f 末 %.3f" % (a_insc, best_c, fin_c))

# ---- 终局矩阵: 3 种子 × {两层, 三层-grow(Do=48), 三层-grow(Do=96)} ----
print("\n=== P2: 终局矩阵 (3 种子) ===")
rows = {'2L': [], '3L-grow D48': [], '3L-grow D96': []}
for sd in (7, 17, 27):
    net2s, a2s = train2l(seed=sd)
    rows['2L'].append(a2s)
    ng = make3(None, seed=sd+100, res=True, w2_scale=0.01, Do=48)
    out, C = fwd3(ng, X4); closed_head(ng, C, y4)
    b48, f48 = grow_train(ng, steps=3000)
    rows['3L-grow D48'].append(b48)
    ng96 = make3(None, seed=sd+100, res=True, w2_scale=0.01, Do=96)
    out, C = fwd3(ng96, X4); closed_head(ng96, C, y4)
    b96, f96 = grow_train(ng96, steps=3000)
    rows['3L-grow D96'].append(b96)
    print("  seed %d: 2L %.3f | grow48 %.3f | grow96 %.3f" % (sd, a2s, b48, b96))
for k, vv in rows.items():
    print("  %-14s mean %.3f ± %.3f  (%s)" % (k, float(np.mean(vv)), float(np.std(vv)),
          np.array2string(np.array(vv), precision=3)))
