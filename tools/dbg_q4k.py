# -*- coding: utf-8 -*-
import sys, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8
from q4k_baseline_err import q4k_deq

kvs, tensors, data_start, raw = read_gguf('data/m2b_q8_0.gguf')
name = 'blk.0.ffn_down.weight'
for nm, dims, qt, off in tensors:
    if nm == name:
        ne0 = dims[0]; ne1 = dims[1] if len(dims) > 1 else 1
        n = ne0 * ne1
        raw.seek(data_start + off)
        buf = raw.read((n + 31) // 32 * 34)
        w = deq_q8(buf, n)
        break
print('w norm', float(np.linalg.norm(w)), 'min', float(w.min()), 'max', float(w.max()))
wq = q4k_deq(w)
print('wq norm', float(np.linalg.norm(wq)), 'min', float(wq.min()), 'max', float(wq.max()))
