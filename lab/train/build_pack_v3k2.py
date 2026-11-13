# -*- coding: utf-8 -*-
# build_pack_v3k2.py — v3k2 全量打包缓存（T=512）+ 压缩落盘供 relay 上传
import sys, os
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A
import numpy as np

ROOT = r"H:\AimCode\lab\cpu-native"
raw = os.path.join(ROOT, "v3k2_804m_cache_packed.npz")
A.build_packed_cache(ROOT + r"\v3k2_804m_mix.jsonl", raw, T=512, seed=42)
z = np.load(raw)
comp = os.path.join(ROOT, "v3k2_pack_cache.npz")
np.savez_compressed(comp, **{k: z[k] for k in z.files})
z.close()
print("raw %.1fMB -> compressed %.1fMB" % (os.path.getsize(raw) / 1e6, os.path.getsize(comp) / 1e6))
