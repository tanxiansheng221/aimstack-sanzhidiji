# -*- coding: utf-8 -*-
# hdc_debug_weights3.py — 受控注入: 隔离「键对齐 / 值码本复用 / 语料结构」三变量
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

Nw = len(train) - 3
K0 = hkey(b'\x01\x02\x03')       # 探针上下文 (语料中不存在)
v0 = P[100]

def build(extra_kind):
    """探针 5 次同键对齐写入 + Nw 条背景写入 (种类由 extra_kind 控制)"""
    M = np.zeros(D, dtype=complex)
    for _ in range(5):
        M += K0.conj() * v0
    if extra_kind == "none":
        pass
    elif extra_kind == "rand_key_rand_val":
        CH = 512
        for s in range(0, Nw, CH):
            e = min(s + CH, Nw)
            Ks = rphase(e - s)
            Vs = P[rng.integers(0, 256, e - s)]
            M += (Ks.conj() * Vs).sum(axis=0)
    elif extra_kind == "corpus":
        CH = 256
        for s in range(3, len(train), CH):
            e = min(s + CH, len(train))
            C = np.stack([hkey(bytes(train[t-3:t])) for t in range(s, e)])
            T = C.conj()
            T *= P[train[s:e]]
            M += T.sum(axis=0)
    return M

for kind in ["none", "rand_key_rand_val", "corpus"]:
    M = build(kind)
    R = M * K0
    corr = np.abs(P @ R.conj()) / D
    others = np.delete(corr, 100)
    print("背景=%-18s corr[探针值]=%6.3f (真值5) | 竞品std=%7.3f (理论%.3f) | 竞品max=%7.3f (理论%.3f)"
          % (kind, corr[100], float(np.std(others)), np.sqrt(Nw/(2*D)),
             float(others.max()), np.sqrt(Nw/(2*D))*np.sqrt(2*np.log(255))))

# 语料写入的键复用检查: 每个键平均写几次
cnt = Counter()
for t in range(3, len(train)):
    cnt[bytes(train[t-3:t])] += 1
print("\n键复用: 独立上下文 %d, 总写入 %d, 平均 %.2f 次/键, 最大 %d 次/键"
      % (len(cnt), Nw, Nw/len(cnt), max(cnt.values())))
