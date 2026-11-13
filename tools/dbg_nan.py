# -*- coding: utf-8 -*-
# 查 m2b_q8_0.gguf 各张量中 f16 scale 非有限的块数
import sys, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf

kvs, tensors, data_start, raw = read_gguf('data/m2b_q8_0.gguf')
for nm, dims, qt, off in tensors:
    if qt != 8:
        continue
    ne0 = dims[0]; ne1 = dims[1] if len(dims) > 1 else 1
    n = ne0 * ne1
    nb = n // 32
    raw.seek(data_start + off)
    CH = 1 << 22
    bad = 0
    for s in range(0, nb, CH):
        e = min(s + CH, nb)
        h16 = np.frombuffer(raw.read((e - s) * 2), np.uint16).view(np.float16)
        bad += int((~np.isfinite(h16.astype(np.float32))).sum())
    print('%-28s blocks=%d bad_scale=%d' % (nm, nb, bad))
