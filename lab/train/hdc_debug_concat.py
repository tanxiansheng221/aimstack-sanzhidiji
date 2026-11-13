# -*- coding: utf-8 -*-
# 快测: 串联式生长 (concat-growth) —— 新层输出作为新坐标附加, 分块闭式头
#   机理: 叠加式残差移动=核几何漂移(§17.2 衰减律); 串联式旧块几何结构不变
#   → 深度单调性有构造性保障, 且新块利用率可以拉满 (无衰减税)
#   对比: 叠加式 (§18) L=1..5 = 0.749/0.655/0.599/0.533/0.481 单调劣化
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

H, Do_b = 10, 32

def make_cat(L, seed, w2=0.3):
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([2.0]*4, 40, r2); W1 = r2.normal(0,1,(H,80))/np.sqrt(80)
    rest = []
    for j in range(1, L):                       # 层 j 输入 = segs[0..j-1] (j*H 维)
        f = freqs_fd([2.0]*(j*H), 32, r2)
        W = r2.normal(0, w2, (H, 64))
        rest.append([f, W])
    fos = [freqs_fd([2.0]*H, Do_b, r2) for _ in range(L)]   # 每段一个头块
    vo = np.zeros(L*2*Do_b)
    return dict(f1=f1, W1=W1, rest=rest, fos=fos, vo=vo, L=L)

def fwd_cat(net, X):
    ph1, A1 = phi_feat(X, net['f1'], 40)
    segs = [np.tanh(A1@net['W1'].T)]
    caches = []
    for j in range(1, net['L']):
        f, W = net['rest'][j-1]
        hin = np.concatenate(segs[:j], 1)       # segs[0..j-1]
        ph, A = phi_feat(hin, f, 32)
        t = np.tanh(A@W.T)
        caches.append((ph, A, t))
        segs.append(t)
    blocks = [phi_feat(segs[j], net['fos'][j], Do_b)[1] for j in range(net['L'])]
    Aall = np.concatenate(blocks, 1)
    out = Aall@net['vo']
    return out, (segs, caches, Aall, A1)

def bwd_cat(net, y, out, C):
    segs, caches, Aall, A1 = C
    n = len(y); g = 2*(out-y)/n
    L = net['L']
    # 头: 每段经 ident 回到段坐标 (w = 该段块的复权重)
    dseg = [None]*L
    for j in range(L):
        wj = net['vo'][j*2*Do_b:j*2*Do_b+Do_b] + 1j*net['vo'][j*2*Do_b+Do_b:(j+1)*2*Do_b]
        dseg[j] = g[:, None]*(-(2*np.pi/np.sqrt(Do_b)))*np.imag(
            phi_feat(segs[j], net['fos'][j], Do_b)[0] @ ((wj[:, None])*net['fos'][j]))
    gs = [None]*L
    for j in range(L-1, 0, -1):                 # 层 j 产 segs[j], 输入 segs[0..j-1]
        f, W = net['rest'][j-1]
        ph, A, t = caches[j-1]
        dLdt = dseg[j]
        dLdu = dLdt*(1-t**2)
        gs[j] = dLdu.T@A
        wc = W[:,:32]+1j*W[:,32:]
        T = np.einsum('nd,hd,dk->nhk', ph, wc, f)
        chain = np.einsum('ni,nik->nk', dLdu, -(2*np.pi/np.sqrt(32))*np.imag(T))
        for k in range(j*H):
            dseg[k // H][:, k % H] += chain[:, k]
    dLdu1 = dseg[0]*(1-segs[0]**2)
    gs[0] = dLdu1.T@A1
    return gs

def closed_head(net, Aall, y, lam=1e-4):
    G = Aall.T@Aall/len(y) + lam*np.eye(len(net['vo']))
    net['vo'] = np.linalg.solve(G, Aall.T@y/len(y))
    return float(np.mean((Aall@net['vo']-y)**2))

def adam_up(p, g, m, v, t, lr):
    m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
    return p-lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8), m, v

def train_cat(L, seed, steps=1500, lr=0.005, lr_new=None, mb=600):
    net = make_cat(L, seed)
    lr_new = lr_new if lr_new is not None else lr
    ps = [net['W1']] + [W for _, W in net['rest']]
    lrs = [lr] + [lr_new]*(len(ps)-1)
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    r2 = np.random.default_rng(seed+900)
    tr_mse = None
    for step in range(1, steps+1):
        idx = np.arange(600) if mb >= 600 else r2.choice(600, mb, replace=False)
        out, C = fwd_cat(net, X4[idx])
        tr_mse = closed_head(net, C[2], y4[idx])
        out = C[2]@net['vo']
        gs = bwd_cat(net, y4[idx], out, C)
        for i in range(len(ps)):
            ps[i], m[i], v[i] = adam_up(ps[i], gs[i], m[i], v[i], step, lrs[i])
        net['W1'] = ps[0]
        for j in range(len(net['rest'])): net['rest'][j][1] = ps[j+1]
    out, C = fwd_cat(net, X4t)
    a = acc(out, y4t)
    util = [float(np.mean(np.abs(c[2]))) for c in C[1]]
    otr, _ = fwd_cat(net, X4)
    return a, util, float(np.mean((otr-y4)**2))

# 梯度检查 (L=3)
net_c = make_cat(3, 5)
out, C = fwd_cat(net_c, X4[:200])
closed_head(net_c, C[2], y4[:200])
out = C[2]@net_c['vo']
gs = bwd_cat(net_c, y4[:200], out, C)
def loss_cat(net, X, y):
    return float(np.mean((fwd_cat(net, X)[0]-y)**2))
eps = 1e-6; worst = 0.0
allp = [(0, net_c['W1'])] + [(j+1, net_c['rest'][j][1]) for j in range(len(net_c['rest']))]
for li, arr in allp:
    gv = gs[li]
    for i in np.argsort(-np.abs(gv).ravel())[:3]:
        ap = arr.copy(); ap.flat[i] += eps
        am = arr.copy(); am.flat[i] -= eps
        if li == 0: net_c['W1'] = ap
        else: net_c['rest'][li-1][1] = ap
        Lp = loss_cat(net_c, X4[:200], y4[:200])
        if li == 0: net_c['W1'] = am
        else: net_c['rest'][li-1][1] = am
        Lm = loss_cat(net_c, X4[:200], y4[:200])
        num = (Lp-Lm)/(2*eps)
        worst = max(worst, abs(num-gv.flat[i])/(abs(gv.flat[i])+1e-10))
print("梯度检查 (concat L=3, 9 入口): 最大相对偏差 %.2e" % worst)

print("\n=== R: 串联式生长 (concat-growth, 分块闭式头) L=1..4, 3 种子 ===")
rows = {}
for L in (1, 2, 3, 4):
    accs, utils, mses = [], None, []
    for sd in (7, 17, 27):
        a, util, tm = train_cat(L, sd)
        accs.append(a); utils = util; mses.append(tm)
    rows[L] = accs
    print("  L=%d: mean %.3f ± %.3f (%s)  train mse %.3f  利用率 %s"
          % (L, float(np.mean(accs)), float(np.std(accs)),
             np.array2string(np.array(accs), precision=3), float(np.mean(mses)),
             np.array2string(np.array(utils), precision=3) if utils else "-"))
print("\n  对照 叠加式 (§18): L=1..5 = 0.749/0.655/0.599/0.533/0.481 单调劣化")
print("\n=== R2: 两时间尺度 (新层 lr=0.02) ===")
for L in (3, 4):
    accs = []
    for sd in (7, 17, 27):
        a, _, _ = train_cat(L, sd, lr_new=0.02)
        accs.append(a)
    print("  L=%d: mean %.3f ± %.3f (%s)"
          % (L, float(np.mean(accs)), float(np.std(accs)),
             np.array2string(np.array(accs), precision=3)))
