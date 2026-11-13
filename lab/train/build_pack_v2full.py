# -*- coding: utf-8 -*-
# build_pack_v2full.py — 十三科全量（158 万行）T=512 打包缓存 + 压缩
import sys, os
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A
import numpy as np

ROOT = r"H:\AimCode\lab\cpu-native"
raw = os.path.join(ROOT, "v2full_cache_packed.npz")
A.build_packed_cache(ROOT + r"\v2_1p5b_full_mix.jsonl", raw, T=512, seed=42)
z = np.load(raw)
print("windows:", len(z["win_tok"]), " val rows:", len(z["vstarts"]))
comp = os.path.join(ROOT, "v2full_pack_cache.npz")
np.savez_compressed(comp, **{k: z[k] for k in z.files})
z.close()
print("raw %.1fMB -> compressed %.1fMB" % (os.path.getsize(raw) / 1e6, os.path.getsize(comp) / 1e6))
print("V2FULL PACK DONE")
