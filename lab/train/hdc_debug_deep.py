# -*- coding: utf-8 -*-
# 快测: grow 配方深度递推 L=1..5 + 两时间尺度 + 残差层利用率
#   奇校验-4, 闭式头每步, 残差层 2..L (层1 plain), 3 种子
import numpy as np, sys, time
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

def make_res(L, seed, D1=40, Dr=32, H=10, Do=48, w2_scale=0.01):
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([2.0]*4, D1, r2); W1 = r2.normal(0,1,(H,2*D1))/np.sqrt(2*D1)
    rest = []
    for l in range(L-1):
        f = freqs_fd([2.0]*H, Dr, r2)
        W = r2.normal(0, w2_scale, (H, 2*Dr))
        rest.append([f, W])
    fo = freqs_fd([2.0]*H, Do, r2); vo = np.zeros(2*Do)
    return dict(f1=f1, W1=W1, rest=rest, fo=fo, vo=vo, D1=D1, Dr=Dr, Do=Do, H=H)

def fwd_res(net, X):
    ph1, A1 = phi_feat(X, net['f1'], net['D1'])
    h1 = np.tanh(A1@net['W1'].T)
    h = h1; caches = []
    for f, W in net['rest']:
        ph, A = phi_feat(h, f, net['Dr'])
        u = A@W.T; t = np.tanh(u)
        caches.append((ph, A, t, h))
        h = h + t
    pho, Ao = phi_feat(h, net['fo'], net['Do'])
    return Ao@net['vo'], (ph1, A1, h1, caches, pho, Ao)

def bwd_res(net, y, out, C):
    ph1, A1, h1, caches, pho, Ao = C
    n = len(y); g = 2*(out-y)/n
    Do = net['Do']; wco = net['vo'][:Do]+1j*net['vo'][Do:]
    dLdh = g[:, None]*(-(2*np.pi/np.sqrt(Do)))*np.imag(pho@((wco[:, None])*net['fo']))
    gWs = [None]*len(net['rest'])
    for li in range(len(net['rest'])-1, -1, -1):
        ph, A, t, h_in = caches[li]
        f, W = net['rest'][li]
        dLdu = dLdh*(1-t**2)
        gWs[li] = dLdu.T@A
        wc = W[:,:net['Dr']]+1j*W[:,net['Dr']:]
        T = np.einsum('nd,hd,dk->nhk', ph, wc, f)
        dLdh_prev = np.einsum('ni,nik->nk', dLdu, -(2*np.pi/np.sqrt(net['Dr']))*np.imag(T)) + dLdh
        dLdh = dLdh_prev
    dLdu1 = dLdh*(1-h1**2)
    g_W1 = dLdu1.T@A1
    return [g_W1]+gWs

def closed_head(net, Ao, y, lam=1e-4):
    G = Ao.T@Ao/len(y) + lam*np.eye(2*net['Do'])
    net['vo'] = np.linalg.solve(G, Ao.T@y/len(y))

def train_res(L, seed, steps=2000, lr=0.005, lr_res=None, mb=600, tag=""):
    net = make_res(L, seed)
    lr_res = lr_res if lr_res is not None else lr
    ps = [net['W1']] + [W for _, W in net['rest']]
    lrs = [lr] + [lr_res]*(len(ps)-1)
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    r2 = np.random.default_rng(seed+500)
    t0 = time.perf_counter()
    for step in range(1, steps+1):
        idx = np.arange(600) if mb >= 600 else r2.choice(600, mb, replace=False)
        out, C = fwd_res(net, X4[idx])
        closed_head(net, C[5], y4[idx])
        out = C[5]@net['vo']
        gWs = bwd_res(net, y4[idx], out, C)
        for i in range(len(ps)):
            gg = gWs[i]
            m[i] = 0.9*m[i]+0.1*gg; v[i] = 0.999*v[i]+0.001*gg*gg
            ps[i] = ps[i]-lrs[i]*(m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8)
        net['W1'] = ps[0]
        for li in range(len(net['rest'])): net['rest'][li][1] = ps[li+1]
        if step == 200: t_step = (time.perf_counter()-t0)/200
    out, C = fwd_res(net, X4t)
    a = acc(out, y4t)
    util = [float(np.mean(np.abs(c[2]))) for c in C[3]]
    return a, t_step, util

print("=== Q1: 深度递推 L=1..5 (grow 配方, 3 种子, 2000 步全批) ===")
table = {}
for L in (1, 2, 3, 4, 5):
    accs, t_step, utils = [], None, None
    for sd in (7, 17, 27):
        a, ts, util = train_res(L, sd)
        accs.append(a); t_step = ts; utils = util
    table[L] = accs
    print("  L=%d (%d 隐层): mean %.3f ± %.3f  (%s)  %.1f ms/步  残差利用率 %s"
          % (L, L, float(np.mean(accs)), float(np.std(accs)),
             np.array2string(np.array(accs), precision=3), t_step*1000,
             np.array2string(np.array(utils), precision=3) if utils else "-"))

print("\n=== Q2: 两时间尺度 (L=4, 残差层 lr=0.02, 体 lr=0.005) ===")
accs = []
for sd in (7, 17, 27):
    a, _, util = train_res(4, sd, lr_res=0.02)
    accs.append(a)
print("  L=4 两时间尺度: mean %.3f ± %.3f  (%s)"
      % (float(np.mean(accs)), float(np.std(accs)), np.array2string(np.array(accs), precision=3)))
