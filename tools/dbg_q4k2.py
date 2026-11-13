# -*- coding: utf-8 -*-
import sys, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8

kvs, tensors, data_start, raw = read_gguf('data/m2b_q8_0.gguf')
for nm, dims, qt, off in tensors[:12]:
    print(nm, dims, 'qtype=', qt)
# 看 ffn_down 头部原始字节
for nm, dims, qt, off in tensors:
    if nm == 'blk.0.ffn_down.weight':
        ne0 = dims[0]; ne1 = dims[1] if len(dims) > 1 else 1
        n = ne0 * ne1
        print('n=', n, 'off=', off)
        raw.seek(data_start + off)
        head = raw.read(40)
        print('first 40 bytes:', head[:40].hex())
        # Q8_0 块头应是 f16 scale (小值) + 32 int8
        scale = np.frombuffer(head[:2], dtype=np.float16)[0]
        ints = np.frombuffer(head[2:34], dtype=np.int8)
        print('scale f16 =', float(scale), 'int8 sample =', ints[:8].tolist())
        break
