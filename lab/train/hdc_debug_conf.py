# -*- coding: utf-8 -*-
# 调试: T6 里 13 个漏网未知问题的置信度从哪来
import numpy as np
rng = np.random.default_rng(7)
exec(open(r"H:\AimCode\lab\cpu-native\hdc_brain_v2.py", encoding="utf-8").read().split("# ---------------- 仿真时间线")[0])

brain = Brain(m_codebook=64)
facts = rand_phase(300); fe = rng.integers(0, brain.m, size=300)
for i in range(300):
    brain.hippo_write(facts[i], int(fe[i]))
facts2 = rand_phase(300); fe2 = rng.integers(0, brain.m, size=300)
for i in range(300):
    brain.hippo_write(facts2[i], int(fe2[i]))
brain.consolidate([(facts[i], int(fe[i])) for i in range(300)], 100)

unk = rand_phase(200)
confs = []
for i in range(200):
    cj, cc = brain.cortex_recall(unk[i])
    hj, hc, _ = brain.hippo_recall(unk[i])
    confs.append((max(cc, hc), cc, hc, cj, hj))
confs.sort(reverse=True)
print("未知问题置信度 TOP10 (总conf, 皮层, 海马, 皮层entry, 海马entry):")
for c in confs[:10]:
    print("  %.4f  cortex=%.4f hippo=%.4f  (%d,%d)" % c[:2] + tuple(c[2:]) if False else "  %.4f cortex=%.4f hippo=%.4f entry=(%d,%d)" % c)
