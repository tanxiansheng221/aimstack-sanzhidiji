# -*- coding: utf-8 -*-
# 从 mini_v2.aimw 文件字节按 C 引擎语义解码 token_embd 前两超块, 对照 quant_v2 的 deq
import struct, sys, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')

buf = open('build/mini_v2.aimw', 'rb').read()
pos = 4
ver, nt = struct.unpack_from('<II', buf, pos); pos += 8
for i in range(nt):
    (nl,) = struct.unpack_from('<I', buf, pos); pos += 4
    name = buf[pos:pos+nl].decode('utf-8'); pos += nl
    dt, nd = struct.unpack_from('<II', buf, pos); pos += 8
    dims = struct.unpack_from('<%dI' % nd, buf, pos); pos += 4*nd
    n = int(np.prod(dims))
    (nb,) = struct.unpack_from('<I', buf, pos); pos += 4
    payload = buf[pos:pos+nb]; pos += nb
    if name == 'token_embd.weight' and dt == 3:
        cb = np.frombuffer(payload[:32], np.float16).astype(np.float32)
        print('codebook:', np.round(cb, 3).tolist())
        blocks = payload[32:]
        # C q4k_scale_min
        def scale_min(j, q):
            if j < 4:
                return q[j] & 63, q[j + 4] & 63
            return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), \
                   (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        def f16_to_f32(h):
            return struct.unpack('<e', struct.pack('<H', h))[0]
        for blk_i in range(2):
            b = blocks[blk_i*144:(blk_i+1)*144]
            d = f16_to_f32(struct.unpack('<H', b[:2])[0])
            dmin = f16_to_f32(struct.unpack('<H', b[2:4])[0])
            vals = []
            for j in range(8):
                sd, sm = scale_min(j, b[4:16])
                d1 = d * sd; m1 = dmin * sm
                qg = b[16 + (j//2)*32 : 16 + (j//2)*32 + 32]
                for l in range(32):
                    nib = (qg[l] & 0xF) if j % 2 == 0 else (qg[l] >> 4)
                    vals.append(d1 * cb[nib] - m1)
            print('blk%d deq[0:8]:' % blk_i, np.round(vals[:8], 4).tolist())
            print('blk%d deq norm:' % blk_i, float(np.linalg.norm(vals)))
        break
