# -*- coding: utf-8 -*-
# §21-M: margin 机制 —— 插值饱和下深度红利的解锁
#   根因: mse 目标在 train mse≈0 后梯度消失, Adam 归一化把噪声放大到满步长 (§20.2)
#   解法: L = mean(relu(m − y·out)²) —— 分界面被推离样本直到 margin 满足,
#         梯度不消失且方向=扩真分界面; 深度层的工作从拟合残差 → 扩 margin
#   全梯度训练 (头不再闭式), parity-4, 对照闭式头基线 0.72 / Bayes 1.0
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exec(open(r'H:\AimCode\lab\cpu-native\hdc_debug_subspace.py', encoding='utf-8')
     .read().split('print("\\n=== S4')[0])

def bwd_g(net, g, C):
    """同 bwd_iter 但 g 外部给定 (margin 梯度). 返回 [gW1, gW0..., gvo] (untied)"""
    segs, caches, Aall, A1 = C
    L = net['L']; db = net['db']
    dseg = [None]*L
    for j in range(L):
        wj = net['vo'][j*2*db:j*2*db+db] + 1j*net['vo'][j*2*db+db:(j+1)*2*db]
        dseg[j] = g[:, None]*(-(2*np.pi/np.sqrt(db)))*np.imag(
            phi_feat(segs[j], net['fos'][j], db)[0] @ ((wj[:, None])*net['fos'][j]))
    gs = [None]*L
    F0 = net['F0']
    for j in range(L-1, 0, -1):
        A_segs, t = caches[j-1]
        Wj = net['W0'] if net['tied'] else net['W0'][j-1]
        dLdu = dseg[j]*(1-t**2)
        gs[j] = dLdu.T@sum(A_segs)
        wc = Wj[:, :32]+1j*Wj[:, 32:]
        for si, A_s in enumerate(A_segs):
            ph_s = phi_feat(segs[si], F0, 32)[0]
            T = np.einsum('nd,hd,dk->nhk', ph_s, wc, F0)
            chain = np.einsum('ni,nik->nk', dLdu, -(2*np.pi/np.sqrt(32))*np.imag(T))
            dseg[si] = dseg[si] + chain
    dLdu1 = dseg[0]*(1-segs[0]**2)
    gs[0] = dLdu1.T@A1
    return gs + [Aall.T@g]          # 末位 = vo 梯度

def train_margin(L, seed, m, steps=3000, lr=0.01, w2=0.3, verbose=False):
    net = make_iter(L, seed, tied=False, w2=w2)
    ps = [net['W1']] + list(net['W0']) + [net['vo']]
    m_v = [np.zeros_like(p) for p in ps]; v_v = [np.zeros_like(p) for p in ps]
    for step in range(1, steps+1):
        out, C = fwd_iter(net, X4)
        viol = np.maximum(m - y4*out, 0.0)          # margin 违反量
        g = -2.0*y4*viol/len(y4)                     # dL/dout
        gs = bwd_g(net, g, C)
        for i in range(len(ps)):
            ps[i], m_v[i], v_v[i] = adam_up(ps[i], gs[i], m_v[i], v_v[i], step, lr)
        net['W1'] = ps[0]
        for j in range(len(net['W0'])): net['W0'][j] = ps[j+1]
        net['vo'] = ps[-1]
    out, C = fwd_iter(net, X4)
    tr_acc = acc(out, y4); tr_viol = float(np.mean(np.maximum(m - y4*out, 0)**2))
    out, C = fwd_iter(net, X4t)
    a = acc(out, y4t)
    return a, tr_acc, tr_viol

print("=== M: margin 目标深度扫描 (parity-4, 全梯度, 3 种子) ===")
print("  基线: 闭式头 L=1 = 0.723  |  Bayes ≈ 1.0")
for m in (0.3, 0.6, 0.9):
    for L in (1, 2, 3):
        accs, tras, viols = [], [], []
        for sd in (7, 17, 27):
            a, ta, tv = train_margin(L, sd, m)
            accs.append(a); tras.append(ta); viols.append(tv)
        print("  m=%.1f L=%d: test %.3f (%s)  train acc %.3f  viol %.4f"
              % (m, L, float(np.mean(accs)),
                 np.array2string(np.array(accs), precision=3),
                 float(np.mean(tras)), float(np.mean(viols))))
print("\n=== M2: 初始化尺度假设 —— w2 扫描 (L=2, 找悬崖边界) ===")
for w2 in (0.3, 0.1, 0.03, 0.01, 0.003):
    for L in (2, 3):
        accs, tras = [], []
        for sd in (7, 17, 27):
            a, ta, tv = train_margin(L, sd, m=0.6, w2=w2)
            accs.append(a); tras.append(ta)
        print("  w2=%.3f L=%d: test %.3f (%s)  train acc %.3f"
              % (w2, L, float(np.mean(accs)),
                 np.array2string(np.array(accs), precision=3), float(np.mean(tras))))
print("\n=== M3: 冲深度红利 —— w2=0.01 + m=0.9 + 步数/lr/阶段化 ===")
print("  L=1 基线 (m=0.9, w2=0.3): 0.781")
for lr, steps in ((0.01, 3000), (0.01, 8000), (0.003, 8000), (0.03, 8000)):
    accs = []
    for sd in (7, 17, 27):
        a, ta, tv = train_margin(2, sd, m=0.9, steps=steps, lr=lr, w2=0.01)
        accs.append(a)
    print("  L=2 lr=%.3f steps=%d: test %.3f (%s)" % (lr, steps, float(np.mean(accs)),
          np.array2string(np.array(accs), precision=3)))

def train_margin_staged(seed, m=0.9, s1=3000, s2=6000, lr=0.01, w2=0.01):
    """阶段1: L=1 margin 训 W1+vo64 → 阶段2: 附加层2(w2 小), W1 冻结, margin 训 W0+vo128"""
    net = make_iter(2, seed, tied=False, w2=w2)
    db = net['db']
    net['L'] = 1; net['vo'] = np.zeros(2*db)
    ps = [net['W1'], net['vo']]
    m_v = [np.zeros_like(p) for p in ps]; v_v = [np.zeros_like(p) for p in ps]
    for step in range(1, s1+1):
        out, C = fwd_iter(net, X4)
        viol = np.maximum(m - y4*out, 0.0)
        gs = bwd_g(net, -2.0*y4*viol/len(y4), C)
        for i in range(2):
            ps[i], m_v[i], v_v[i] = adam_up(ps[i], gs[i], m_v[i], v_v[i], step, lr)
        net['W1'] = ps[0]; net['vo'] = ps[1]
    out, C = fwd_iter(net, X4t); a1 = acc(out, y4t)
    net['L'] = 2
    vo128 = np.zeros(4*db); vo128[:2*db] = net['vo']; net['vo'] = vo128
    ps = [net['W0'][0], net['vo']]
    m_v = [np.zeros_like(p) for p in ps]; v_v = [np.zeros_like(p) for p in ps]
    for step in range(1, s2+1):
        out, C = fwd_iter(net, X4)
        viol = np.maximum(m - y4*out, 0.0)
        gs = bwd_g(net, -2.0*y4*viol/len(y4), C)   # [gW1, gW0, gvo], W1 冻结
        for i, gi in ((0, 1), (1, 2)):
            ps[i], m_v[i], v_v[i] = adam_up(ps[i], gs[gi], m_v[i], v_v[i], step, lr)
        net['W0'][0] = ps[0]; net['vo'] = ps[1]
    out, C = fwd_iter(net, X4t); a2 = acc(out, y4t)
    return a1, a2

rows = []
for sd in (7, 17, 27):
    a1, a2 = train_margin_staged(sd)
    rows.append((a1, a2))
    print("  阶段化 seed %d: %.3f → %.3f (Δ%+.3f)" % (sd, a1, a2, a2-a1))
print("  阶段化 mean: %.3f → %.3f" % (np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])))
print("\n  判读: 任一臂 L=2 ≥ 0.78 → 深度红利兑现")
