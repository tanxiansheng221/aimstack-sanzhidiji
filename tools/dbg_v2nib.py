# -*- coding: utf-8 -*-
# 对照 quant_v2 的 nq 与文件内 nibble (token_embd block 3)
import sys, os, struct
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
import gguf2aimw_v2 as gv

rng = np.random.default_rng(77)
D, VOC = 256, 256
te = rng.normal(0, 1, D * VOC).astype(np.float32)
payload, cb, deq, err = gv.quant_v2(te)

# 复算 nq (与 quant_v2 内部一致的路径)
w = te
NB = len(w) // 256
B = w.reshape(NB, 8, 32)
wmax = B.max(axis=2); wmin = B.min(axis=2)
eff = (wmax - wmin) / np.float32(15.0)
offm = np.maximum(-wmin, np.float32(0))
d_stored = np.float16(eff.max(axis=1) / np.float32(63.0))
dmin_stored = np.float16(offm.max(axis=1) / np.float32(63.0))
d32 = d_stored.astype(np.float32); dmin32 = dmin_stored.astype(np.float32)
sc = np.clip(np.round(eff / np.maximum(d32[:, None], 1e-12)), 0, 63).astype(np.int32)
ms = np.clip(np.round(offm / np.maximum(dmin32[:, None], 1e-12)), 0, 63).astype(np.int32)
d1 = d32[:, None] * sc.astype(np.float32)
m1 = dmin32[:, None] * ms.astype(np.float32)
z = ((B + m1[:, :, None]) / np.maximum(d1[:, :, None], 1e-12)).astype(np.float32)
mid = ((cb[:-1] + cb[1:]) * np.float32(0.5))
nq_ref = np.clip(np.searchsorted(mid, z.reshape(-1)), 0, 15).astype(np.uint8).reshape(NB, 8, 32)
print('quant nq blk3:', nq_ref[3].tolist())

# 文件 nibble
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
    payload_f = buf[pos:pos+nb]; pos += nb
    if name == 'token_embd.weight' and dt == 3:
        blocks = payload_f[32:]
        b = blocks[3*144:4*144]
        nib = np.zeros((8, 32), np.uint8)
        for j in range(8):
            qg = np.frombuffer(b, np.uint8, 32, 16 + (j//2)*32)
            nib[j] = (qg & 0xF) if j % 2 == 0 else (qg >> 4)
        print('file  nib blk3:', nib.tolist())
        print('nq match:', bool((nib == nq_ref[3]).all()))
        break
