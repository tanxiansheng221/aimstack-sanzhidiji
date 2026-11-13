# -*- coding: utf-8 -*-
# 逐条目对账 (单条目扰动, 全量 loss): layer 0 (W1) 与 layer 1 (rest W)
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exec(open(r'H:\AimCode\lab\cpu-native\hdc_debug_concat.py', encoding='utf-8').read().split('# 梯度检查')[0])

net = make_cat(2, 5)
Xb, yb = X4[:300], y4[:300]
closed_head(net, fwd_cat(net, Xb)[1][2], yb)
out, C = fwd_cat(net, Xb)
gs = bwd_cat(net, yb, out, C)

def loss_f():
    return float(np.mean((fwd_cat(net, Xb)[0]-yb)**2))

eps = 1e-6
for li, arr in [(0, net['W1']), (1, net['rest'][0][1])]:
    gv = gs[li]
    ii = np.argsort(-np.abs(gv).ravel())[:5]
    print('--- layer', li)
    for i in ii:
        old = arr.flat[i]
        arr.flat[i] = old + eps
        Lp = loss_f()
        arr.flat[i] = old - eps
        Lm = loss_f()
        arr.flat[i] = old
        num = (Lp-Lm)/(2*eps)
        print('  entry %d: ana %+.6e  num %+.6e  ratio %.3f'
              % (i, gv.flat[i], num, num/(gv.flat[i]+1e-12)))
