# -*- coding: utf-8 -*-
# compress_lang1.py - lang1_pack.npz 无损重压缩
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

P = r"H:\AimCode\lab\cpu-native\lang1_pack.npz"
t0 = time.time()
z = np.load(P)
data = {k: z[k] for k in z.files}
z.close()
np.savez_compressed(P, **data)
print("compressed: %.1fMB (%.0fs)" % (os.path.getsize(P) / 1e6, time.time() - t0))
