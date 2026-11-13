# -*- coding: utf-8 -*-
# AIM-W v2 部署形态实测: 全张量共享归一化点阵(全局 Lloyd) + 每组一个 scale
# 对照: lloyd_research.py 的逐组自由点阵(质量上限) 与 Q4_K_M(业界 ~0.06-0.08)
# EMBD 截断前 800 万元素加速(误差统计已够)
import sys, os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8

print('loading...')
kvs, tensors, data_start, raw = read_gguf('data/m2b_q8_0.gguf')

def get_weight(name, cap=None):
    for nm, dims, qt, off in tensors:
        if nm == name:
            ne0 = dims[0]; ne1 = dims[1] if len(dims) > 1 else 1
            n = ne0 * ne1
            if cap: n = min(n, cap)
            raw.seek(data_start + off)
            buf = raw.read((n + 31) // 32 * 34)
            return deq_q8(buf, n)
    raise KeyError(name)

def lloyd_shared(w, k, group, iters=8):
    """部署形态: 归一化域全局共享 k 点点阵, 每组只存 amax(f16)
    解码: w = amax_g * C[idx]  -> C 是每张量 k 个 f16, 每组 2B scale"""
    n = len(w)
    pad = (-n) % group
    if pad:
        w = np.concatenate([w, np.zeros(pad, np.float32)])
    G = len(w) // group
    B = w.reshape(G, group)
    amax = np.abs(B).max(axis=1) + 1e-12  # (G,)
    bn = B / amax[:, None]                # 归一化到 [-1,1]
    qs = np.quantile(bn, np.linspace(0.02, 0.98, k))  # (k,)
    qs = np.sort(qs).astype(np.float32)
    for _ in range(iters):
        mid = ((qs[:-1] + qs[1:]) * np.float32(0.5))      # (k-1,)
        idx = (bn[:, :, None] > mid).sum(axis=2)          # (G, group)
        new = qs.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new[j] = np.float32(bn[m].mean())
        qs = np.sort(new)
    mid = ((qs[:-1] + qs[1:]) * np.float32(0.5))
    idx = (bn[:, :, None] > mid).sum(axis=2)
    wq = (qs[idx] * amax[:, None]).reshape(-1)[:n]
    return wq

CASES = [
    ('blk.0.ffn_down.weight',  'FFN.d0',  None),
    ('blk.10.ffn_down.weight', 'FFN.d10', None),
    ('blk.0.attn_q.weight',    'ATT.q0',  None),
    ('token_embd.weight',      'EMBD',    8_000_000),
]
KS = [8, 16]
GROUPS = [256, 64]

print('%-8s %4s %5s %10s %10s' % ('tensor', 'k', 'grp', 'rel_err', 'GB@2.5B'))
for name, tag, cap in CASES:
    w = get_weight(name, cap)
    base = np.linalg.norm(w.astype(np.float64))
    for k in KS:
        for g in GROUPS:
            wq = lloyd_shared(w, k, g)
            err = np.linalg.norm((w - wq).astype(np.float64)) / base
            bits = np.log2(k) + 16.0 / g
            gb = 2.5e9 * bits / 8 / 1e9
            print('%-8s %4d %5d %10.4f %10.2f' % (tag, k, g, err, gb))
    print()
