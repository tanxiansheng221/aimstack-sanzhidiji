# -*- coding: utf-8 -*-
# 收敛性验证: iters 8 vs 30, 看共享点阵能否逼近 Lloyd-Max 理论界 (~0.058 @ k16)
import sys, os
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

def lloyd_shared(w, k, group, iters):
    n = len(w)
    pad = (-n) % group
    if pad:
        w = np.concatenate([w, np.zeros(pad, np.float32)])
    G = len(w) // group
    B = w.reshape(G, group)
    amax = np.abs(B).max(axis=1) + 1e-12
    bn = B / amax[:, None]
    qs = np.sort(np.quantile(bn, np.linspace(0.02, 0.98, k))).astype(np.float32)
    for _ in range(iters):
        mid = ((qs[:-1] + qs[1:]) * np.float32(0.5))
        idx = (bn[:, :, None] > mid).sum(axis=2)
        new = qs.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new[j] = np.float32(bn[m].mean())
        qs = np.sort(new)
    mid = ((qs[:-1] + qs[1:]) * np.float32(0.5))
    idx = (bn[:, :, None] > mid).sum(axis=2)
    return (qs[idx] * amax[:, None]).reshape(-1)[:n]

print('%-8s %4s %5s %6s %10s' % ('tensor', 'k', 'grp', 'iters', 'rel_err'))
for name, tag in [('blk.0.ffn_down.weight', 'FFN.d0'), ('blk.0.attn_q.weight', 'ATT.q0')]:
    w = get_weight(name)
    base = np.linalg.norm(w.astype(np.float64))
    for iters in (8, 30):
        for k in (8, 16):
            wq = lloyd_shared(w, k, 256, iters)
            err = np.linalg.norm((w - wq).astype(np.float64)) / base
            print('%-8s %4d %5d %6d %10.4f' % (tag, k, 256, iters, err))
    print()
