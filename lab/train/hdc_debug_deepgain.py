# -*- coding: utf-8 -*-
# §21: 深度红利求解 —— 路线N: 难任务(parity-8)下深度扫描
#   §20 定案: 深度税=W训练动力学×插值饱和; 反之, L=1 不饱和(train mse>0)时深度应有正空间
#   parity-8: 256 patterns(16^2 于 parity-4), L=1 的 64 参数头未必能插值
#   判读: train mse>0 且 test 随 L 上升 → §20 理论兑现, 深度红利解决
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exec(open(r'H:\AimCode\lab\cpu-native\hdc_debug_subspace.py', encoding='utf-8')
     .read().split('print("\\n=== S4')[0])

# ---- parity-8 数据 ----
rng8 = np.random.default_rng(113)
nb = rng8.integers(0, 2, (600, 8)); s8 = 2.0*nb-1
X8 = s8 + 0.1*rng8.normal(0, 1, (600, 8)); y8 = np.prod(s8, 1)
nb = rng8.integers(0, 2, (500, 8)); s8t = 2.0*nb-1
X8t = s8t + 0.1*rng8.normal(0, 1, (500, 8)); y8t = np.prod(s8t, 1)
# 覆盖全局, train_iter/fwd 等直接工作
X4, y4, X4t, y4t = X8, y8, X8t, y8t

def make_iter8(L, seed, w2=0.3, tied=True, db=None):
    db = Do_b if db is None else db
    r2 = np.random.default_rng(seed)
    f1 = freqs_fd([2.0]*8, 40, r2); W1 = r2.normal(0, 1, (H, 80))/np.sqrt(80)
    F0 = freqs_fd([2.0]*H, 32, r2)
    if tied:
        W0 = r2.normal(0, w2, (H, 64))
    else:
        W0 = [r2.normal(0, w2, (H, 64)) for _ in range(L-1)]
    fos = [freqs_fd([2.0]*H, db, r2) for _ in range(L)]
    return dict(f1=f1, W1=W1, F0=F0, W0=W0, fos=fos, vo=np.zeros(L*2*db), L=L, tied=tied, db=db)

def train8(L, seed, steps=1500, lr=0.005, tied=True, lam=1e-4):
    net = make_iter8(L, seed, tied=tied)
    extra = []
    if L > 1:
        extra = [net['W0']] if tied else list(net['W0'])
    ps = [net['W1']] + extra
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    tr = None
    for step in range(1, steps+1):
        out, C = fwd_iter(net, X8)
        tr = closed_head(net, C[2], y8, lam=lam)
        out = C[2]@net['vo']
        gs = bwd_iter(net, y8, out, C)
        for i in range(len(ps)):
            ps[i], m[i], v[i] = adam_up(ps[i], gs[i], m[i], v[i], step, lr)
        net['W1'] = ps[0]
        if L > 1:
            if tied: net['W0'] = ps[1]
            else:
                for j in range(len(net['W0'])): net['W0'][j] = ps[j+1]
    out, C = fwd_iter(net, X8t)
    a = acc(out, y8t)
    return a, float(tr)

# ---- N0: Bayes headroom 对照 (真 parity 面在噪声输入上) ----
print("=== N0: parity-8 headroom ===")
print("  真 parity 面在 test 噪声输入上 acc = %.3f (Bayes≈1, headroom 存在)"
      % acc(np.prod(np.sign(X8t), 1), y8t))

print("\n=== N1: parity-8 深度扫描 (全栈共训, L=1..4, 3 种子) ===")
for L in (1, 2, 3, 4):
    accs, trs = [], []
    for sd in (7, 17, 27):
        a, tr = train8(L, sd)
        accs.append(a); trs.append(tr)
    print("  L=%d: mean %.3f ± %.3f (%s)  train mse %.4f"
          % (L, float(np.mean(accs)), float(np.std(accs)),
             np.array2string(np.array(accs), precision=3), float(np.mean(trs))))
print("\n  判读: train mse>0 → L=1 不饱和; test 随 L 上升 → 深度红利兑现")
