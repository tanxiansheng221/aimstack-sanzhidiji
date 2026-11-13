# -*- coding: utf-8 -*-
# §20补: 深度上限扫描 —— 纯附加形态能堆几层?
#   配方: 层0 W1 按 L=1 配方训练(唯一训练), 层 1..L-1 随机 W 冻结零训练
#         闭式头一次重解 (S7b 已证 L=2 时 Δ≈0 无损)
#   问题: L→16 时头维度 2·db·L 增长 + ridge 条件数, test 是否最终衰减?
#   两臂: w2=0.3 (标准随机) / w2=0.01 (近恒等小扰动)
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
exec(open(r'H:\AimCode\lab\cpu-native\hdc_debug_subspace.py', encoding='utf-8')
     .read().split('print("\\n=== S4')[0])

def depth_cap(L, seed, s1=1500, lr=0.005, w2=0.3, db=32):
    net = make_iter(L, seed, tied=False, w2=w2, db=db)
    # ---- 阶段1: L=1 视图, 只训 W1 ----
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
    # ---- 阶段2: 全栈展开, W0 全冻结, 头一次重解 ----
    net['L'] = L; net['vo'] = np.zeros(L*2*db)
    out, C = fwd_iter(net, X4)
    tr = closed_head(net, C[2], y4)
    out, C = fwd_iter(net, X4t); a2 = acc(out, y4t)
    bn = [float(np.linalg.norm(net['vo'][j*2*db:(j+1)*2*db])) for j in range(L)]
    return a1, a2, tr, bn

print("=== S8: 纯附加深度上限 L=1..16, 3 种子 ===")
for w2 in (0.3, 0.01):
    print(" --- w2=%.2f ---" % w2)
    for L in (1, 2, 4, 8, 16):
        if L == 1 and w2 == 0.01:
            pass  # L=1 无附加层, 两臂相同, 只跑一次
        a1s, a2s, trs = [], [], []
        for sd in (7, 17, 27):
            a1, a2, tr, bn = depth_cap(L, sd, w2=w2)
            a1s.append(a1); a2s.append(a2); trs.append(tr)
        print("  L=%2d: 阶段1 %.3f → 纯附加 %.3f (Δ%+.3f)  train mse %.4f"
              % (L, float(np.mean(a1s)), float(np.mean(a2s)),
                 float(np.mean(a2s)-np.mean(a1s)), float(np.mean(trs))))
        if L == 16:
            print("        L=16 头块范数分布: %s"
                  % np.array2string(np.array(bn), precision=2))
print("\n  判读: 纯附加列全程 ≈ 阶段1 → 结构上限=无 (头/ridge 不崩);")
print("        出现衰减的 L = 纯附加形态的结构上限")
