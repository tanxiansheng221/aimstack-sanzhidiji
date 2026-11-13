# -*- coding: utf-8 -*-
# hdc_debug_weights2.py — 解剖 W2 哈希键: 信号 c 是否对齐? 竞品噪声分布长什么样?
import numpy as np
import sys, zlib
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(37)
D = 16384

def rphase(n):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, D)))

files = [f for f in sorted(Path(__file__).parent.glob("*.py")) if f.name.startswith("hdc_")]
corpus = b"".join(f.read_bytes() for f in files)[:16000]
train = np.frombuffer(corpus, dtype=np.uint8).astype(np.int64)
P = rphase(256)

_keycache = {}
def hkey(ctxbytes):
    v = _keycache.get(ctxbytes)
    if v is None:
        r = np.random.default_rng(zlib.crc32(ctxbytes))
        v = np.exp(1j * r.uniform(0, 2 * np.pi, D))
        _keycache[ctxbytes] = v
    return v

CH = 256
M = np.zeros(D, dtype=complex)
c3 = defaultdict(Counter)
for s in range(3, len(train), CH):
    e = min(s + CH, len(train))
    for t in range(s, e):
        c3[bytes(train[t-3:t])][int(train[t])] += 1
    C = np.stack([hkey(bytes(train[t-3:t])) for t in range(s, e)])
    T = C.conj()
    T *= P[train[s:e]]
    M += T.sum(axis=0)
Nw = len(train) - 3
print("语料 %d, Nw=%d, 独立上下文 %d" % (len(train), Nw, len(_keycache)))

# 取唯一续写、不同计数的上下文, 训练位置自回忆
uniq = {k: v for k, v in c3.items() if len(v) == 1}
picks = []
for k, v in uniq.items():
    c = sum(v.values())
    if c in (5, 6, 7, 9, 12, 20, 30, 60, 100, 150, 200, 300):
        picks.append((c, k, v))
picks.sort()
seen = set()
print("\n[训练位置自回忆, 哈希键, 唯一续写]")
print("%6s %8s %10s %10s %10s" % ("c", "corr真实", "竞品std", "理论std", "竞品max"))
for c, k, v in picks[:12]:
    ts = np.arange(3, len(train))
    kb = bytes(k)
    # 找该上下文的一个训练位置
    idx = None
    for t in range(3, len(train)):
        if bytes(train[t-3:t]) == kb:
            idx = t; break
    if idx is None: continue
    C = hkey(kb)
    R = M * C
    corr = np.abs(P @ R.conj()) / D
    true = list(v.keys())[0]
    others = np.delete(corr, true)
    ub = int(np.argsort(others)[::-1][0])
    print("%6d %8.2f %10.3f %10.3f %10.3f   (max竞品字节=%d)"
          % (c, corr[true], float(np.std(others)), np.sqrt(Nw/(2*D)), float(others.max()), ub))

# 逐写入解剖: 单独写一条, 看它的串扰贡献
print("\n[单写入串扰解剖] 只写 1 条 (ctx0->v0), 再用别的 ctx 查询:")
k0 = hkey(b'abc')
v0 = P[100]
M1x = (k0.conj() * v0)
for probe in [b'xyz', b'abc', b'abd']:
    Cq = hkey(probe)
    R = M1x * Cq
    corr = np.abs(P @ R.conj()) / D
    print("  查询 %s: corr[100]=%.4f, 竞品max=%.4f, 竞品std=%.4f (理论 per-write 1/sqrt(2D)=%.4f)"
          % (probe, corr[100], float(np.delete(corr, 100).max()), float(np.std(np.delete(corr, 100))), 1/np.sqrt(2*D)))
