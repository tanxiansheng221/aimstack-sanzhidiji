# -*- coding: utf-8 -*-
# build_pack_v2fmt.py — 四遍（格式工程补刀）打包：fmt 混料 → T=512 窗口缓存
import sys, os
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A
import numpy as np

ROOT = r"H:\AimCode\lab\cpu-native"
raw = os.path.join(ROOT, "v2fmt_cache_packed.npz")
A.build_packed_cache(ROOT + r"\v2fmt_1p5b_full_mix.jsonl", raw, T=512, seed=42)
z = np.load(raw)
print("windows:", len(z["win_tok"]), " val rows:", len(z["vstarts"]))
comp = os.path.join(ROOT, "v2fmt_pack_cache.npz")
np.savez_compressed(comp, **{k: z[k] for k in z.files})
z.close()
print("raw %.1fMB -> compressed %.1fMB" % (os.path.getsize(raw) / 1e6, os.path.getsize(comp) / 1e6))
print("V2FMT PACK DONE")
