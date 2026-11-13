# -*- coding: utf-8 -*-
# AIM-W v2 决定性实验: Q4_K 容器(32子块仿射 d/m + 6bit sc)不变, 均匀 16 级网格
# 换成全张量 Lloyd 学习的非均匀点阵. 同码长严格不劣于 Q4_K(uniform 是其特例).
# 对照组: uniform16(=Q4_K 4.50bpw) / lloyd16(4.50bpw) / lloyd12(4.09bpw) / lloyd8(3.50bpw)
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

def lloyd_1d(z, k, iters=12):
    """全局 1D Lloyd 点阵: 返回排序后的 k 个中心"""
    qs = np.sort(np.quantile(z, np.linspace(0.01, 0.99, k))).astype(np.float32)
    CH = 1 << 22
    for _ in range(iters):
        mid = ((qs[:-1] + qs[1:]) * np.float32(0.5))
        new = np.zeros_like(qs)
        cnt = np.zeros(k, np.int64)
        for s in range(0, len(z), CH):
            b = z[s:s + CH]
            idx = np.searchsorted(mid, b).clip(0, k - 1)
            cnt += np.bincount(idx, minlength=k)
            new += np.bincount(idx, weights=b, minlength=k)
        qs = np.sort(np.where(cnt > 0, new / np.maximum(cnt, 1), qs))
    return qs

def q4k_cb(w, levels, iters=12):
    """Q4_K 仿射容器 + 指定点阵 levels(均匀=Q4_K 原版; 学习=AIM-W v2)"""
    n = len(w)
    pad = (-n) % 256
    if pad:
        w = np.concatenate([w, np.zeros(pad, np.float32)])
    NB = len(w) // 256
    B = w.reshape(NB, 8, 32)
    wmax = B.max(axis=2)
    wmin = B.min(axis=2)
    eff = (wmax - wmin) / np.float32(15.0)
    offm = np.maximum(-wmin, np.float32(0))
    d = eff.max(axis=1, keepdims=True)
    dmin = offm.max(axis=1, keepdims=True)
    sc = np.clip(np.round(np.float32(63.0) * eff / np.maximum(d, 1e-12)), 0, 63)
    ms = np.clip(np.round(np.float32(63.0) * offm / np.maximum(dmin, 1e-12)), 0, 63)
    d1 = d * sc / np.float32(63.0)
    m1 = dmin * ms / np.float32(63.0)
    z = (B + m1[:, :, None]) / np.maximum(d1[:, :, None], 1e-12)  # 归一化层级域
    zc = z.reshape(-1)
    if levels == 'lloyd':
        c = lloyd_1d(zc, k, iters)
    else:
        c = np.arange(k, dtype=np.float32)
    mid = ((c[:-1] + c[1:]) * np.float32(0.5))
    idx = np.clip(np.searchsorted(mid, zc), 0, k - 1).reshape(NB, 8, 32)
    deq = (d1[:, :, None] * c[idx] - m1[:, :, None]).reshape(-1)[:n]
    return deq

CASES = [
    ('blk.0.ffn_down.weight',  'FFN.d0',  None),
    ('blk.10.ffn_down.weight', 'FFN.d10', None),
    ('blk.0.attn_q.weight',    'ATT.q0',  None),
    ('token_embd.weight',      'EMBD',    8_000_000),
]
print('%-8s %-12s %6s %10s %10s' % ('tensor', 'scheme', 'bpw', 'rel_err', 'vs_Q4K'))
for name, tag, cap in CASES:
    w = get_weight(name, cap)
    base = np.linalg.norm(w.astype(np.float64))
    rows = []
    for scheme, k in [('uniform16', 16), ('lloyd16', 16), ('lloyd12', 12), ('lloyd8', 8)]:
        wq = q4k_cb(w, 'lloyd' if scheme.startswith('lloyd') else 'uniform', k)
        err = np.linalg.norm((w - wq).astype(np.float64)) / base
        bpw = np.log2(k) + 0.375 + 0.125  # 层级位 + scales(96b/256) + d/dmin(32b/256)
        rows.append((scheme, bpw, err))
    ref = rows[0][2]
    for scheme, bpw, err in rows:
        print('%-8s %-12s %6.2f %10.4f %+9.1f%%' % (tag, scheme, bpw, err, (err - ref) / ref * 100))
    print()
