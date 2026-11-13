# -*- coding: utf-8 -*-
# §21-N2: 深度红利终局检验 —— parity-8 × 6000 样本 (L=1 必然欠拟合)
#   配方全集: margin 目标(m=0.9) + 小初始化(w2=0.01) + 大步长(lr=0.03)
#   判据: L=1 train acc<1 (信号未耗尽) 且 test 随 L 上升 → 深度红利定案
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
exec(open(r'H:\AimCode\lab\cpu-native\hdc_debug_subspace.py', encoding='utf-8')
     .read().split('print("\\n=== S4')[0])

rng8 = np.random.default_rng(113)
SIG = 0.5                                 # 大噪声: 标签歧义 → train 不可完美拟合
nb = rng8.integers(0, 2, (600, 4)); s8 = 2.0*nb-1
X8 = s8 + SIG*rng8.normal(0, 1, (600, 4)); y8 = np.prod(s8, 1)
nb = rng8.integers(0, 2, (2000, 4)); s8t = 2.0*nb-1
X8t = s8t + SIG*rng8.normal(0, 1, (2000, 4)); y8t = np.prod(s8t, 1)
X4, y4, X4t, y4t = X8, y8, X8t, y8t      # 覆盖全局

def make_iter8(L, seed, w2=0.01, db=32):
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([2.0]*4, 40, r2); W1 = r2.normal(0, 1, (H, 80))/np.sqrt(80)
    F0 = freqs_fd([2.0]*H, 32, r2)
    W0 = [r2.normal(0, w2, (H, 64)) for _ in range(L-1)]
    fos = [freqs_fd([2.0]*H, db, r2) for _ in range(L)]
    return dict(f1=f1, W1=W1, F0=F0, W0=W0, fos=fos, vo=np.zeros(L*2*db), L=L, tied=False, db=db)

# Bayes 上界: 用干净顶点的真 parity 直接对噪声 test 打分
clean = np.prod(np.sign(X8t), 1)
print("Bayes 参考 (真 parity 面 @σ=%.1f): %.3f" % (SIG, acc(clean, y8t)))

def bwd_g(net, g, C):
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
        Wj = net['W0'][j-1]
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
    return gs + [Aall.T@g]

def train_m8(L, seed, m=0.9, steps=4000, lr=0.03, w2=0.01):
    net = make_iter8(L, seed, w2=w2)
    ps = [net['W1']] + list(net['W0']) + [net['vo']]
    m_v = [np.zeros_like(p) for p in ps]; v_v = [np.zeros_like(p) for p in ps]
    for step in range(1, steps+1):
        out, C = fwd_iter(net, X8)
        viol = np.maximum(m - y8*out, 0.0)
        gs = bwd_g(net, -2.0*y8*viol/len(y8), C)
        for i in range(len(ps)):
            ps[i], m_v[i], v_v[i] = adam_up(ps[i], gs[i], m_v[i], v_v[i], step, lr)
        net['W1'] = ps[0]
        for j in range(len(net['W0'])): net['W0'][j] = ps[j+1]
        net['vo'] = ps[-1]
    out, C = fwd_iter(net, X8); tr_acc = acc(out, y8)
    out, C = fwd_iter(net, X8t)
    return acc(out, y8t), tr_acc

print("=== N3: parity-4 × σ=0.5 (标签歧义, train 不可完美拟合) ===")
for L in (1, 2, 3):
    accs, tras = [], []
    for sd in (7, 17, 27):
        a, ta = train_m8(L, sd)
        accs.append(a); tras.append(ta)
    print("  L=%d: test %.3f (%s)  train acc %.3f"
          % (L, float(np.mean(accs)), np.array2string(np.array(accs), precision=3),
             float(np.mean(tras))))
