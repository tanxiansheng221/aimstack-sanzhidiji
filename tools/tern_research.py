# -*- coding: utf-8 -*-
# 三值坍缩护道研究: 不同 scale 粒度 + 残差级数的量化误差实测 (真模型权重)
# 指标: 相对 Frobenius 误差 err = ||w-wq||/||w|| ; 非零码占比 ; 等效 bit
import struct, sys, os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8

print('loading gguf header...')
kvs, tensors, data_start, raw = read_gguf('data/m2b_q8_0.gguf')

def get_weight(name):
    for nm, dims, qt, off in tensors:
        if nm == name:
            ne0 = dims[0]; ne1 = dims[1] if len(dims) > 1 else 1
            n = ne0 * ne1
            raw.seek(data_start + off)
            buf = raw.read((n + 31) // 32 * 34)
            return deq_q8(buf, n)
    raise KeyError(name)

def tern(w, group):
    """分组三值: 每 group 权重独立 absmax scale"""
    wq = np.empty_like(w)
    nz = 0
    n = len(w)
    for s in range(0, n, group):
        b = w[s:s + group]
        scale = np.float32(np.abs(b).max()) + 1e-12
        codes = np.clip(np.round(b / scale), -1, 1)
        wq[s:s + group] = codes * scale
        nz += int((codes != 0).sum())
    return wq, nz / n

def tern_res(w, group):
    """双三值残差: 第一级 + 第二级对残差再三值 (等效 9 点阵)"""
    wq = np.empty_like(w)
    nz = 0
    n = len(w)
    for s in range(0, n, group):
        b = w[s:s + group]
        s1 = np.float32(np.abs(b).max()) + 1e-12
        c1 = np.clip(np.round(b / s1), -1, 1)
        r = b - c1 * s1
        s2 = np.float32(np.abs(r).max()) + 1e-12
        c2 = np.clip(np.round(r / s2), -1, 1)
        wq[s:s + group] = c1 * s1 + c2 * s2
        nz += int((c1 != 0).sum()) + int((c2 != 0).sum())
    return wq, nz / (2 * n)

CASES = [
    ('blk.0.ffn_down.weight',  'FFN.d0'),
    ('blk.10.ffn_down.weight', 'FFN.d10'),
    ('blk.0.attn_q.weight',    'ATT.q0'),
    ('blk.10.attn_q.weight',   'ATT.q10'),
    ('token_embd.weight',      'EMBD'),
]
GROUPS = [None, 256, 64, 32]

print('%-8s %-14s %10s %10s %8s %8s' % ('tensor', 'scheme', 'rel_err', 'cos_sim', 'nz%', 'eff.bit'))
for name, tag in CASES:
    w = get_weight(name)
    base = np.linalg.norm(w.astype(np.float64))
    # per-tensor
    wq, nz = tern(w, len(w))
    err = np.linalg.norm((w - wq).astype(np.float64)) / base
    print('%-8s %-14s %10.4f %10.4f %8.1f %8.2f' % (tag, 'tern/tensor', err, 1 - err, nz * 100, 1.58))
    for g in GROUPS[1:]:
        wq, nz = tern(w, g)
        err = np.linalg.norm((w - wq).astype(np.float64)) / base
        bits = 2.0 + 16.0 / g   # 2bit 码 + 组内 f16 scale 摊销
        print('%-8s %-14s %10.4f %10.4f %8.1f %8.2f' % (tag, 'tern/g%d' % g, err, 1 - err, nz * 100, bits))
    # 双三值残差
    for g in (256, 64):
        wq, nz = tern_res(w, g)
        err = np.linalg.norm((w - wq).astype(np.float64)) / base
        bits = 4.0 + 32.0 / g
        print('%-8s %-14s %10.4f %10.4f %8.1f %8.2f' % (tag, 'tern2/g%d' % g, err, 1 - err, nz * 100, bits))
    print()
