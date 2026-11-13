# -*- coding: utf-8 -*-
# §21 定版表: 最优配方 (margin m=0.9 + w2=0.01 + lr=0.03) L=1..4 全扫
#   加速: 链式 einsum 分解 — Σ_h dLdu·Σ_d φ·wc·f 中 h/d 可分离:
#     q = dLdu@wc (n,d);  chain = -(2π/√32)·Im((q·φ)@f)   [成本 n·d·k, 降 ~10×]
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
exec(open(r'H:\AimCode\lab\cpu-native\hdc_debug_margin.py', encoding='utf-8')
     .read().split('print("=== M:')[0])

def bwd_g_fast(net, g, C):
    segs, caches, Aall, A1 = C
    L = net['L']; db = net['db']; F0 = net['F0']
    dseg = [None]*L
    for j in range(L):
        wj = net['vo'][j*2*db:j*2*db+db] + 1j*net['vo'][j*2*db+db:(j+1)*2*db]
        dseg[j] = g[:, None]*(-(2*np.pi/np.sqrt(db)))*np.imag(
            phi_feat(segs[j], net['fos'][j], db)[0] @ ((wj[:, None])*net['fos'][j]))
    gs = [None]*L
    for j in range(L-1, 0, -1):
        A_segs, t = caches[j-1]
        Wj = net['W0'][j-1]
        dLdu = dseg[j]*(1-t**2)
        gs[j] = dLdu.T@sum(A_segs)
        wc = Wj[:, :32]+1j*Wj[:, 32:]
        q = dLdu@wc                                    # (n, 32)
        for si, A_s in enumerate(A_segs):
            ph_s = phi_feat(segs[si], F0, 32)[0]
            chain = -((2*np.pi/np.sqrt(32)))*np.imag((q*ph_s)@F0)
            dseg[si] = dseg[si] + chain
    dLdu1 = dseg[0]*(1-segs[0]**2)
    gs[0] = dLdu1.T@A1
    return gs + [Aall.T@g]

# --- 加速版自检 (对照慢速参考, L=3) ---
net_c = make_iter(3, 5, tied=False)
out, C = fwd_iter(net_c, X4)
gg = -2.0*y4*np.maximum(0.9 - y4*out, 0)/len(y4)
g_ref = bwd_g(net_c, gg, C); g_fast = bwd_g_fast(net_c, gg, C)
print("加速链式自检: 最大偏差 %.2e" % max(np.abs(a-b).max() for a, b in zip(g_ref, g_fast)))

def train_final(L, seed, m=0.9, steps=6000, lr=0.03, w2=0.01):
    net = make_iter(L, seed, tied=False, w2=w2)
    ps = [net['W1']] + list(net['W0']) + [net['vo']]
    m_v = [np.zeros_like(p) for p in ps]; v_v = [np.zeros_like(p) for p in ps]
    for step in range(1, steps+1):
        out, C = fwd_iter(net, X4)
        viol = np.maximum(m - y4*out, 0.0)
        gs = bwd_g_fast(net, -2.0*y4*viol/len(y4), C)
        for i in range(len(ps)):
            ps[i], m_v[i], v_v[i] = adam_up(ps[i], gs[i], m_v[i], v_v[i], step, lr)
        net['W1'] = ps[0]
        for j in range(len(net['W0'])): net['W0'][j] = ps[j+1]
        net['vo'] = ps[-1]
    out, C = fwd_iter(net, X4); tr_acc = acc(out, y4)
    out, C = fwd_iter(net, X4t)
    return acc(out, y4t), tr_acc

print("=== FINAL: 修复配方深度曲线 (parity-4, m=0.9, w2=0.01, lr=0.03, 6000 步, 3 种子) ===")
print("  (对照修复前 §20: 0.723/0.533/0.527/0.505 悬崖式单调劣化)")
for L in (1, 2, 3, 4):
    accs, tras = [], []
    for sd in (7, 17, 27):
        a, ta = train_final(L, sd)
        accs.append(a); tras.append(ta)
    print("  L=%d: test %.3f ± %.3f (%s)  train acc %.3f"
          % (L, float(np.mean(accs)), float(np.std(accs)),
             np.array2string(np.array(accs), precision=3), float(np.mean(tras))))
