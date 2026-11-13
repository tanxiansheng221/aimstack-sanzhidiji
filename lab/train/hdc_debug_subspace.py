# -*- coding: utf-8 -*-
# §20: 频率子空间共享 / 权重绑定迭代层 —— 攻 concat 深度劣化第二成因
#   机理: concat 劣化第二成因 = hin 维随深度增长 (jH) → φ 频率维增长 → W 优化景观变难
#   两臂分离变量:
#     S2 全绑定 (tied):   每段同 F0 同 W0, 层 j 输入 = Σ_{s<j} φ(segs[s],F0)@W0.T, 参数恒定 640
#     S3 频率共享 (free): 每段同 F0 但各层 W0 独立, 参数仍随深度增长 (对照)
#   判读: S2 若深度单调劣化消失 → 第二成因确证 (参数/景观增长); S3 劣化而 S2 不劣化
#         → 增长的是 W 参数本身; S2 也劣化 → 第二成因不成立, 另有根源
#   对照 §19 concat: L=1..4 = 0.709/0.533/0.527/0.505 (train mse 0/0/0.283/1.183)
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

H, Do_b = 10, 32

def make_iter(L, seed, w2=0.3, tied=True, db=None):
    db = Do_b if db is None else db
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([2.0]*4, 40, r2); W1 = r2.normal(0,1,(H,80))/np.sqrt(80)
    F0 = freqs_fd([2.0]*H, 32, r2)                    # 段频率库 (32,H) 全层共享
    if tied:
        W0 = r2.normal(0, w2, (H, 64))                # 单一 W, 参数恒定
    else:
        W0 = [r2.normal(0, w2, (H, 64)) for _ in range(L-1)]
    fos = [freqs_fd([2.0]*H, db, r2) for _ in range(L)]
    return dict(f1=f1, W1=W1, F0=F0, W0=W0, fos=fos, vo=np.zeros(L*2*db), L=L, tied=tied, db=db)

def fwd_iter(net, X):
    db = net['db']
    ph1, A1 = phi_feat(X, net['f1'], 40)
    segs = [np.tanh(A1@net['W1'].T)]
    caches = []
    F0 = net['F0']
    for j in range(1, net['L']):
        A_segs = [phi_feat(sv, F0, 32)[1] for sv in segs]        # 各段同 F0
        Wj = net['W0'] if net['tied'] else net['W0'][j-1]
        t = np.tanh(sum(A@Wj.T for A in A_segs))
        caches.append((A_segs, t))
        segs.append(t)
    blocks = [phi_feat(segs[j], net['fos'][j], db)[1] for j in range(net['L'])]
    Aall = np.concatenate(blocks, 1)
    out = Aall@net['vo']
    return out, (segs, caches, Aall, A1)

def bwd_iter(net, y, out, C):
    segs, caches, Aall, A1 = C
    n = len(y); g = 2*(out-y)/n
    L = net['L']; db = net['db']
    dseg = [None]*L
    for j in range(L):                                            # 头: 每段独立块 (同 §19)
        wj = net['vo'][j*2*db:j*2*db+db] + 1j*net['vo'][j*2*db+db:(j+1)*2*db]
        dseg[j] = g[:, None]*(-(2*np.pi/np.sqrt(db)))*np.imag(
            phi_feat(segs[j], net['fos'][j], db)[0] @ ((wj[:, None])*net['fos'][j]))
    gs = [None]*L
    F0 = net['F0']
    gs_W0 = np.zeros_like(net['W0']) if (net['tied'] and L > 1) else None
    for j in range(L-1, 0, -1):
        A_segs, t = caches[j-1]
        Wj = net['W0'] if net['tied'] else net['W0'][j-1]
        dLdu = dseg[j]*(1-t**2)
        Asum = sum(A_segs)
        if net['tied']:
            gs_W0 = gs_W0 + dLdu.T@Asum        # 绑定 W0: 跨层梯度求和
        else:
            gs[j] = dLdu.T@Asum
        wc = Wj[:, :32]+1j*Wj[:, 32:]
        # 链入: 层 j 对每段 s 的梯度 (同 F0, wc), 段映射无需 k%H — 每段整列
        for si, A_s in enumerate(A_segs):
            ph_s = phi_feat(segs[si], F0, 32)[0]
            T = np.einsum('nd,hd,dk->nhk', ph_s, wc, F0)
            chain = np.einsum('ni,nik->nk', dLdu, -(2*np.pi/np.sqrt(32))*np.imag(T))
            dseg[si] = dseg[si] + chain
    if net['tied'] and L > 1:
        gs[1] = gs_W0
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

def train_iter(L, seed, steps=1500, lr=0.005, tied=True, lam=1e-4, db=None):
    net = make_iter(L, seed, tied=tied, db=db)
    extra = []
    if L > 1:
        extra = [net['W0']] if tied else list(net['W0'])
    ps = [net['W1']] + extra
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    tr_mse = None
    for step in range(1, steps+1):
        out, C = fwd_iter(net, X4)
        tr_mse = closed_head(net, C[2], y4, lam=lam)
        out = C[2]@net['vo']
        gs = bwd_iter(net, y4, out, C)
        for i in range(len(ps)):
            ps[i], m[i], v[i] = adam_up(ps[i], gs[i], m[i], v[i], step, lr)
        net['W1'] = ps[0]
        if L > 1:
            if tied: net['W0'] = ps[1]
            else:
                for j in range(len(net['W0'])): net['W0'][j] = ps[j+1]
    out, C = fwd_iter(net, X4t)
    a = acc(out, y4t)
    otr, _ = fwd_iter(net, X4)
    return a, float(np.mean((otr-y4)**2))

# ---- 梯度检查 (L=3, tied 与 free 各 6 入口) ----
def loss_iter(net, X, y):
    return float(np.mean((fwd_iter(net, X)[0]-y)**2))

eps = 1e-6
print("=== S0: 梯度检查 (L=3) ===")
for tied in (True, False):
    net_c = make_iter(3, 5, tied=tied)
    out, C = fwd_iter(net_c, X4[:200])
    closed_head(net_c, C[2], y4[:200])
    out = C[2]@net_c['vo']
    gs = bwd_iter(net_c, y4[:200], out, C)
    params = [(0, net_c['W1'])] + ([(1, net_c['W0'])] if tied
             else [(j+1, net_c['W0'][j]) for j in range(len(net_c['W0']))])
    worst = 0.0
    for li, arr in params:
        gv = gs[li]
        for i in np.argsort(-np.abs(gv).ravel())[:2]:
            old = arr.flat[i]
            arr.flat[i] = old + eps; Lp = loss_iter(net_c, X4[:200], y4[:200])
            arr.flat[i] = old - eps; Lm = loss_iter(net_c, X4[:200], y4[:200])
            arr.flat[i] = old
            num = (Lp-Lm)/(2*eps)
            worst = max(worst, abs(num-gv.flat[i])/(abs(gv.flat[i])+1e-10))
    print("  %s: 最大相对偏差 %.2e" % ("S2 全绑定" if tied else "S3 频率共享", worst))

print("\n=== S: 频率子空间共享 L=1..4, 3 种子 ===")
for tag, tied in (("S2 全绑定(参数恒定)", True), ("S3 仅共享F0(W独立)", False)):
    print(" ---", tag)
    for L in (1, 2, 3, 4):
        accs, mses = [], []
        for sd in (7, 17, 27):
            a, tm = train_iter(L, sd, tied=tied)
            accs.append(a); mses.append(tm)
        print("  L=%d: mean %.3f ± %.3f (%s)  train mse %.3f"
              % (L, float(np.mean(accs)), float(np.std(accs)),
                 np.array2string(np.array(accs), precision=3), float(np.mean(mses))))
print("\n  对照 §19 concat: L=1..4 = 0.709/0.533/0.527/0.505 (train mse 0/0/0.283/1.183)")
print("  对照 §18 叠加式: L=1..5 = 0.749/0.655/0.599/0.533/0.481")

print("\n=== S4: 容量假说判别 — L=1 但头加倍 (Do_b=64, 头参数 128 = L=2 头, 零深度) ===")
accs, mses = [], []
for sd in (7, 17, 27):
    a, tm = train_iter(1, sd, db=64)
    accs.append(a); mses.append(tm)
print("  L=1/db=64: mean %.3f ± %.3f (%s)  train mse %.3f"
      % (float(np.mean(accs)), float(np.std(accs)),
         np.array2string(np.array(accs), precision=3), float(np.mean(mses))))
print("  判读: 掉到 ~0.53 → 容量假说成立 (头参数是深度税); 守住 ~0.72 → 深度本身有税")

print("\n=== S5: 头正则救援 — L=2/3, λ 扫描 ===")
for L in (2, 3):
    for lam in (1e-4, 1e-2, 1e-1):
        accs, mses = [], []
        for sd in (7, 17, 27):
            a, tm = train_iter(L, sd, lam=lam)
            accs.append(a); mses.append(tm)
        print("  L=%d λ=%.0e: mean %.3f (%s)  train mse %.3f"
              % (L, lam, float(np.mean(accs)),
                 np.array2string(np.array(accs), precision=3), float(np.mean(mses))))

print("\n=== S7: 分阶段真生长 — 阶段1 只训 W1 → 阶段2 附加层2 冻结 W1 只训 W0 ===")
def grow_staged(seed, s1=1500, s2=1500, lr=0.005, lr2=0.005, lam2=1e-4):
    net = make_iter(2, seed, tied=False)
    db = net['db']
    # ---- 阶段1: L=1 视图, 头=block0 ----
    net['L'] = 1; net['vo'] = np.zeros(2*db)
    ps = [net['W1']]; m = [np.zeros_like(net['W1'])]; v = [np.zeros_like(net['W1'])]
    for step in range(1, s1+1):
        out, C = fwd_iter(net, X4)
        closed_head(net, C[2], y4)
        out = C[2]@net['vo']
        gs = bwd_iter(net, y4, out, C)
        ps[0], m[0], v[0] = adam_up(ps[0], gs[0], m[0], v[0], step, lr)
        net['W1'] = ps[0]
    out, C = fwd_iter(net, X4t); a1 = acc(out, y4t)
    # ---- 阶段2: 附加层2, W1 冻结 (旧块几何零移动), 头闭式重解 128 维 ----
    net['L'] = 2; net['vo'] = np.zeros(4*db)
    out, C = fwd_iter(net, X4)                    # 附加后至少解一次头 (s2=0 也生效)
    closed_head(net, C[2], y4, lam=lam2)
    ps = [net['W0'][0]]; m = [np.zeros_like(ps[0])]; v = [np.zeros_like(ps[0])]
    for step in range(1, s2+1):
        out, C = fwd_iter(net, X4)
        closed_head(net, C[2], y4, lam=lam2)
        out = C[2]@net['vo']
        gs = bwd_iter(net, y4, out, C)
        ps[0], m[0], v[0] = adam_up(ps[0], gs[1], m[0], v[0], step, lr2)
        net['W0'][0] = ps[0]
    out, C = fwd_iter(net, X4t); a2 = acc(out, y4t)
    otr, _ = fwd_iter(net, X4)
    return a1, a2, float(np.mean((otr-y4)**2))

rows = []
for sd in (7, 17, 27):
    a1, a2, tm = grow_staged(sd)
    rows.append((a1, a2, tm))
    print("  seed %d: 阶段1 %.3f → 阶段2 %.3f  (Δ%+.3f)  train mse %.3f"
          % (sd, a1, a2, a2-a1, tm))
a1s = np.array([r[0] for r in rows]); a2s = np.array([r[1] for r in rows])
print("  mean: %.3f → %.3f  单调率 %d/3" % (a1s.mean(), a2s.mean(), int(np.sum(a2s >= a1s - 0.01))))

print("\n=== S7b: 纯附加 (s2=0, W0 不训, 全靠闭式头裁决) ===")
rows_b = []
for sd in (7, 17, 27):
    a1, a2, tm = grow_staged(sd, s2=0)
    rows_b.append((a1, a2))
    print("  seed %d: 阶段1 %.3f → 附加 %.3f  (Δ%+.3f)  train mse %.3f" % (sd, a1, a2, a2-a1, tm))
print("  mean: %.3f → %.3f" % (np.mean([r[0] for r in rows_b]), np.mean([r[1] for r in rows_b])))

print("\n=== S7c: 阶段2 加大 λ (W0 训练+强脊) ===")
for lam2 in (1e-2, 1e-1, 1e0):
    rows_c = []
    for sd in (7, 17, 27):
        a1, a2, tm = grow_staged(sd, lam2=lam2)
        rows_c.append((a1, a2))
    print("  λ=%.0e: 阶段1 %.3f → 阶段2 %.3f  单调率 %d/3"
          % (lam2, np.mean([r[0] for r in rows_c]), np.mean([r[1] for r in rows_c]),
             int(np.sum([r[1] >= r[0]-0.01 for r in rows_c]))))
