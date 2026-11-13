# -*- coding: utf-8 -*-
# 对照 quant_v2 deq 与文件解码: token_embd row3 前 8 元素
import sys, os, struct
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw_v2 import quant_v2

rng = np.random.default_rng(77)
D, VOC = 256, 256
te = rng.normal(0, 1, D * VOC).astype(np.float32)
payload, cb, deq, err = quant_v2(te)
print('quant deq row3[0:8]:', np.round(deq[3*256:3*256+8], 4).tolist())
# 文件解码对照 (dbg_v2decode 的 blk3 = row3)
buf = open('build/mini_v2.aimw', 'rb').read()
pos = 4
ver, nt = struct.unpack_from('<II', buf, pos); pos += 8
import struct
for i in range(nt):
    (nl,) = struct.unpack_from('<I', buf, pos); pos += 4
    name = buf[pos:pos+nl].decode('utf-8'); pos += nl
    dt, nd = struct.unpack_from('<II', buf, pos); pos += 8
    dims = struct.unpack_from('<%dI' % nd, buf, pos); pos += 4*nd
    n = int(np.prod(dims))
    (nb,) = struct.unpack_from('<I', buf, pos); pos += 4
    payload = buf[pos:pos+nb]; pos += nb
    if name == 'token_embd.weight' and dt == 3:
        cbf = np.frombuffer(payload[:32], np.float16).astype(np.float32)
        blocks = payload[32:]
        def scale_min(j, q):
            if j < 4:
                return q[j] & 63, q[j + 4] & 63
            return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), \
                   (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        b = blocks[3*144:4*144]
        d = float(np.frombuffer(b[:2], np.float16)[0])
        dmin = float(np.frombuffer(b[2:4], np.float16)[0])
        vals = []
        for j in range(8):
            sd, sm = scale_min(j, b[4:16])
            d1 = np.float32(d) * np.float32(sd); m1 = np.float32(dmin) * np.float32(sm)
            qg = b[16 + (j//2)*32: 16 + (j//2)*32 + 32]
            for l in range(32):
                nib = (qg[l] & 0xF) if j % 2 == 0 else (qg[l] >> 4)
                vals.append(np.float32(d1 * cbf[nib] - m1))
        print('file  deq row3[0:8]:', np.round(vals[:8], 4).tolist())
        print('codebook match:', bool(np.allclose(cbf, cb)))
        break
