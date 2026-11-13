# -*- coding: utf-8 -*-
import sys, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8

kvs, tensors, data_start, raw = read_gguf('data/m2b_q8_0.gguf')
print('data_start =', data_start)
name = 'blk.0.ffn_down.weight'
for nm, dims, qt, off in tensors:
    if nm == name:
        ne0 = dims[0]; ne1 = dims[1] if len(dims) > 1 else 1
        n = ne0 * ne1
        raw.seek(data_start + off)
        buf = raw.read((n + 31) // 32 * 34)
        print('read bytes =', len(buf))
        # 手工算前 4 块
        for b in range(4):
            blk = buf[b*34:(b+1)*34]
            sc = np.frombuffer(blk[:2], np.uint16).astype(np.float16)[0]
            q = np.frombuffer(blk[2:], np.int8)
            print('blk', b, 'scale', float(sc), 'q0..3', q[:4].tolist(), 'deq0..3', (float(sc)*q[:4].astype(np.float32)).tolist())
        w = deq_q8(buf, n)
        print('deq_q8 norm', float(np.linalg.norm(w)), 'min', float(w.min()), 'max', float(w.max()))
        print('first 8 deq:', w[:8].tolist())
        break
