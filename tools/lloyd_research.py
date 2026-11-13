# -*- coding: utf-8 -*-
# AIM-W v2 研究: k 点非均匀点阵(Lloyd) x 分组 scale 的体积-精度曲线 (真模型权重)
# 点阵候选: k=3(1.58bit)/4(2bit)/8(3bit)/16(4bit); 组: 256/64
# 基线对照: Q4_K_M 等效误差 ~0.06-0.08 (业界), Q8 ~0.02
import struct, sys, os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8

print('loading...')
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

def lloyd_k(w, k, group, iters=8):
    """分组 k 点 Lloyd: 跨组向量化(按 chunk 分批), 1D 最近中心逐组独立"""
    n = len(w)
    pad = (-n) % group
    if pad:
        w = np.concatenate([w, np.zeros(pad, np.float32)])
    G = len(w) // group
    B = w.reshape(G, group)
    wq = np.empty_like(B)
    CH = 4096  # 每批组数, 控制中间张量内存
    for s in range(0, G, CH):
        b = B[s:s + CH]  # (g, group)
        qs = np.quantile(b, np.linspace(0.02, 0.98, k), axis=1)  # (k, g)
        qs = np.sort(qs.T, axis=1).astype(np.float32)  # (g, k)
        for _ in range(iters):
            mid = (qs[:, :-1] + qs[:, 1:]) * np.float32(0.5)  # (g, k-1)
            idx = (b[:, :, None] > mid[:, None, :]).sum(axis=2)  # (g, group)
            new = qs.copy()
            for j in range(k):
                m = idx == j
                cnt = m.sum(axis=1)
                sm = np.where(m, b, np.float32(0)).sum(axis=1)
                new[:, j] = np.where(cnt > 0, sm / np.maximum(cnt, 1), qs[:, j])
            qs = np.sort(new, axis=1)
        mid = (qs[:, :-1] + qs[:, 1:]) * np.float32(0.5)
        idx = (b[:, :, None] > mid[:, None, :]).sum(axis=2)
        wq[s:s + CH] = np.take_along_axis(qs, idx, axis=1)
    return wq.reshape(-1)[:n]

CASES = [
    ('blk.0.ffn_down.weight',  'FFN.d0'),
    ('blk.10.ffn_down.weight', 'FFN.d10'),
    ('blk.0.attn_q.weight',    'ATT.q0'),
    ('token_embd.weight',      'EMBD'),
]
KS = [3, 4, 8, 16]
GROUPS = [256, 64]

print('%-8s %4s %5s %10s %10s' % ('tensor', 'k', 'grp', 'rel_err', 'GB@2.5B'))
for name, tag in CASES:
    w = get_weight(name)
    base = np.linalg.norm(w.astype(np.float64))
    for k in KS:
        for g in GROUPS:
            wq = lloyd_k(w, k, g)
            err = np.linalg.norm((w - wq).astype(np.float64)) / base
            bits = np.log2(k) + 16.0 / g
            gb = 2.5e9 * bits / 8 / 1e9
            print('%-8s %4d %5d %10.4f %10.2f' % (tag, k, g, err, gb))
    print()
