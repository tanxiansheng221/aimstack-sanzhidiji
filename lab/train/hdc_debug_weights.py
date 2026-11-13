# -*- coding: utf-8 -*-
# hdc_debug_weights.py — 解剖实验1背离: 信号 c 是否真的对齐? 噪声底在哪?
import numpy as np
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(37)
D = 16384

def rphase(n):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, D)))

files = [f for f in sorted(Path(__file__).parent.glob("*.py")) if f.name != "hdc_weights.py"]
corpus = b"".join(f.read_bytes() for f in files)[:8000]
train = np.frombuffer(corpus, dtype=np.uint8).astype(np.int64)
P = rphase(256)
Q3 = rphase(3)

def ctx3_rows(src, ts):
    return (P[src[ts - 3]] * Q3[0]) * (P[src[ts - 2]] * Q3[1]) * (P[src[ts - 1]] * Q3[2])

CH = 64
def write_super(src, t_lo, t_hi):
    M = np.zeros(D, dtype=complex)
    for s in range(t_lo, t_hi, CH):
        e = min(s + CH, t_hi)
        ts = np.arange(s, e)
        C = ctx3_rows(src, ts)
        T = C.conj()
        T *= P[src[ts]]
        M += T.sum(axis=0)
    return M

M3 = write_super(train, 3, len(train))
Nw = len(train) - 3
print("小语料 %d 字节, Nw=%d" % (len(train), Nw))

# ---- 对照组 0: 玩具随机键管线 (hdc_dialog 同款) ----
K = 600
keys = rphase(K)
cb = rphase(8)
vidx = rng.integers(0, 8, size=K)
Mt = (keys.conj() * cb[vidx]).sum(axis=0)
ok = 0
for i in rng.integers(0, K, size=50):
    R = Mt * keys[i]
    ok += int(np.argmax(np.abs(cb @ R.conj()))) == vidx[i]
print("[对照0 玩具随机键] %d/50 (应≈50)" % ok)

# ---- 计数表 ----
c3 = defaultdict(Counter)
for t in range(3, len(train)):
    c3[(train[t-3], train[t-2], train[t-1])][train[t]] += 1

top = sorted(c3.items(), key=lambda kv: -sum(kv[1].values()))[:6]
print("\n[对照组1: 训练位置自回忆, top 计数上下文]")
for ctx, cnt in top:
    c = sum(cnt.values())
    ts = np.arange(3, len(train))
    mask = (train[ts-3] == ctx[0]) & (train[ts-2] == ctx[1]) & (train[ts-1] == ctx[2])
    t = int(ts[mask][0])
    C = ctx3_rows(train, np.array([t]))[0]
    R = M3 * C
    corr = np.abs(P @ R.conj()) / D
    true = int(train[t])
    order = np.argsort(corr)[::-1]
    rank = int(np.where(order == true)[0][0])
    # 分解: R = c_major·V_major + ...
    maj_byte, maj_c = cnt.most_common(1)[0]
    Vm = P[maj_byte]
    Rn = R - maj_c * Vm
    print("  ctx=%s c=%d 值分布=%s" % (repr(bytes(ctx)), c, dict(cnt)))
    print("    corr[true]=%.3f (rank %d), corr_top=%.3f, 竞品corr std=%.3f, 理论噪声std=%.3f"
          % (corr[true], rank, corr[order[0]] if order[0] != true else corr[order[1]],
             float(np.std(np.delete(corr, true))), np.sqrt(Nw / (2.0 * D))))
    print("    |R - c_maj*V_maj| 每维均值=%.3f (理论 sqrt(n/2)=%.1f), |c_maj*V_maj| 每维=%.1f"
          % (float(np.mean(np.abs(Rn))), np.sqrt(Nw / 2), float(np.mean(np.abs(maj_c * Vm)))))
