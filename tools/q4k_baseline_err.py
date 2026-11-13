# -*- coding: utf-8 -*-
# Q4_K 语义基线 rel_err 实测(与 v2 Lloyd 点阵同批张量、同口径对账)
# Q4_K: 4.5 bits/weight (144B/256); v2 k16: 4.0625 bits (g256) / 4.25 bits (g64)
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

def q4k_deq(w):
    """Q4_K min-max 线量化反推值(向量化, llama.cpp 官方语义):
    d>0, dmin>0 均为正幅度; w = d1*n - m1, n∈[0,15], m1 = dmin*m >= 0
    每子块32: eff=(wmax-wmin)/15, off_mag=-wmin; super-block256: d=max(eff), dmin=max(off_mag)
    sc/m=round(63*比值) 6bit 截断; nq=clip(round((w+m1)/d1),0,15); deq=d1*nq-m1"""
    n = len(w)
    pad = (-n) % 256
    if pad:
        w = np.concatenate([w, np.zeros(pad, np.float32)])
    NB = len(w) // 256
    B = w.reshape(NB, 8, 32)
    wmax = B.max(axis=2)  # (NB,8)
    wmin = B.min(axis=2)
    eff = (wmax - wmin) / np.float32(15.0)
    offm = np.maximum(-wmin, np.float32(0))  # 正幅度
    d = eff.max(axis=1, keepdims=True)    # (NB,1)
    dmin = offm.max(axis=1, keepdims=True)
    sc = np.clip(np.round(np.float32(63.0) * eff / np.maximum(d, 1e-12)), 0, 63)
    ms = np.clip(np.round(np.float32(63.0) * offm / np.maximum(dmin, 1e-12)), 0, 63)
    d1 = d * sc / np.float32(63.0)   # sc/63 是分数 (0,1]; 6bit 系数语义
    m1 = dmin * ms / np.float32(63.0)
    nq = np.clip(np.round((B + m1[:, :, None]) / np.maximum(d1[:, :, None], 1e-12)), 0, 15)
    deq = (d1[:, :, None] * nq - m1[:, :, None]).reshape(-1)
    return deq[:n]

CASES = [
    ('blk.0.ffn_down.weight',  'FFN.d0',  None),
    ('blk.10.ffn_down.weight', 'FFN.d10', None),
    ('blk.0.attn_q.weight',    'ATT.q0',  None),
    ('token_embd.weight',      'EMBD',    8_000_000),
]
if __name__ == '__main__':
    print('%-8s %10s' % ('tensor', 'rel_err'))
    for name, tag, cap in CASES:
        w = get_weight(name, cap)
        base = np.linalg.norm(w.astype(np.float64))
        wq = q4k_deq(w)
        err = np.linalg.norm((w - wq).astype(np.float64)) / base
        print('%-8s %10.4f' % (tag, err))
