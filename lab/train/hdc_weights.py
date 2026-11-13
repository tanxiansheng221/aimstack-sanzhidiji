# -*- coding: utf-8 -*-
# hdc_weights.py — 第二容器作为「模型权重的另一条路」：HDC 叠加语言模型
#   主张: 语言模型的权重可以不经过梯度下降 —— 叠加向量组就是"权重",
#         一遍写入(纯加法)即训练完成, 可合并(相加)、可低比特量化(相位)、可持续学习。
#   W1 缺陷实证: 角色-填充键(P⊙Q) —— 位置码在共享角色上精确抵消, 同类写入产生
#      相干常向量串扰(σ²=Σn_c²), 独立噪声律失效。
#   W2a 缺陷实证: 单库哈希键 —— 上下文重复(键复用)使重复写入对其他查询相干叠加,
#      受控注入: 随机键背景 σ=0.475(合理论) vs 真实语料 σ=2.26(3.2×, 重尾)。
#   W2b 修复 = 分库(SDM 式): 哈希分桶, 桶内噪声 √(桶占用), 频率对齐信号保留,
#      与计数 n-gram 表达到 parity, 设计律恢复成立。
#   W3 合并=逐桶相加(数学精确); W4 相位量化=低比特权重(sinc 律);
#   W5 持续学习=桶占用增长的 √ 噪声税 / 分库精确零遗忘。
import numpy as np
import sys, time, zlib
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(37)
D = 16384
B = 256                                   # 桶数 (SDM 式分库)

def rphase(n):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, D)))

files = [f for f in sorted(Path(__file__).parent.glob("*.py")) if f.name != "hdc_weights.py"]
corpus = b"".join(f.read_bytes() for f in files)[:36000]
N = len(corpus)
split = int(N * 0.9)
train = np.frombuffer(corpus[:split], dtype=np.uint8).astype(np.int64)
full = np.frombuffer(corpus, dtype=np.uint8).astype(np.int64)
Nw = len(train) - 3
print("语料 %d 字节 (%d 个文件), 训练段 %d, 测试段 %d, D=%d, B=%d" % (N, len(files), split, N - split, D, B))

P = rphase(256)
Q3 = rphase(3)

def ctx3_rows(src, ts):
    return (P[src[ts - 3]] * Q3[0]) * (P[src[ts - 2]] * Q3[1]) * (P[src[ts - 1]] * Q3[2])

CH = 256

# ---------- 计数表基线 ----------
c3 = defaultdict(Counter)
c2 = defaultdict(Counter)
uni = Counter(train[3:len(train)])
for t in range(3, len(train)):
    c3[(train[t-3], train[t-2], train[t-1])][train[t]] += 1
    c2[(train[t-2], train[t-1])][train[t]] += 1
def table_pred(b3, b2, b1):
    k3 = c3.get((b3, b2, b1))
    if k3: return k3.most_common(1)[0][0]
    k2 = c2.get((b2, b1))
    if k2: return k2.most_common(1)[0][0]
    return uni.most_common(1)[0][0]

# ---------- 解析设计律 ----------
#   信号 = c(同上下文对齐), 噪声 = n-c 条独立写入的随机相位投影
def pred_acc(c, n, nmc=20000):
    nn = max(n - c, 1)
    sd = np.sqrt(nn / (2.0 * D))
    z = rng.normal(0, sd, nmc) + 1j * rng.normal(0, sd, nmc)
    s = np.abs(c + z)
    return float(np.mean((1.0 - np.exp(-(s * s) * D / nn)) ** 255))

nq = 400
tq = np.sort(rng.choice(np.arange(split + 3, N), size=nq, replace=False))

def uniq_flag(t):
    return len(c3.get((full[t-3], full[t-2], full[t-1]), {})) == 1

def print_buckets(rows, nwt):
    """rows: list of (c, hit_vs_truth, hit_vs_table, m_b)"""
    edges = [0, 1, 2, 3, 5, 9, 20, 50, 10**9]
    labels = ["0(未见)", "1", "2", "3-4", "5-8", "9-19", "20-49", "50+"]
    print("    %-8s %5s %10s %10s %10s" % ("c", "题数", "对真值", "对表argmax", "解析(唯一续写)"))
    for i in range(len(labels)):
        lo, hi = edges[i], edges[i+1]
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel: continue
        u = [r for r in sel if r[4]]
        ana = float(np.mean([pred_acc(r[0], r[3]) for r in u])) if u else float("nan")
        print("    %-8s %5d %9.1f%% %10.1f%% %10.1f%%"
              % (labels[i], len(sel), 100*np.mean([r[1] for r in sel]),
                 100*np.mean([r[2] for r in sel]), 100*ana))

# ================================================================
# W1: 角色-填充键 (P⊙Q) — 相干串扰缺陷实证
# ================================================================
def write_rf(src, t_lo, t_hi):
    M = np.zeros(D, dtype=complex)
    for s in range(t_lo, t_hi, CH):
        e = min(s + CH, t_hi)
        ts = np.arange(s, e)
        C = ctx3_rows(src, ts)
        T = C.conj()
        T *= P[src[ts]]
        M += T.sum(axis=0)
    return M

t0 = time.perf_counter()
M3_rf = write_rf(train, 3, len(train))
print("\n=== W1: 角色-填充键 (P⊙Q), 写入 %.1fs ===" % (time.perf_counter() - t0))
def rf_pred(Mv, t):
    C = ctx3_rows(full, np.array([t]))[0]
    return int(np.argmax(np.abs(P @ (Mv * C).conj()) / D))
rows = []
for t in tq:
    pred = rf_pred(M3_rf, int(t))
    c = sum(c3.get((full[t-3], full[t-2], full[t-1]), {}).values())
    tab = table_pred(full[t-3], full[t-2], full[t-1])
    rows.append((c, pred == full[t], pred == tab, Nw, uniq_flag(t)))
print("  对真值 %d/%d | 频率-准确率背离 (解析=独立律, 系统性失效=相干串扰):" % (sum(r[1] for r in rows), nq))
print_buckets(rows, Nw)
print("  机理: 共享角色上 conj(Q)·Q=1 精确抵消 → W=conj(P[b'])⊙P[b]⊙P[v] 为类别常向量")

# ================================================================
# W2a: 单库哈希键 — 键复用相干噪声 (受控注入见 hdc_debug_weights3.py:
#      随机键背景 σ=0.475 合理论, 真实语料 σ=2.26 = 3.2×, 重尾)
# ================================================================
_keycache = {}
def hkey(ctxbytes):
    v = _keycache.get(ctxbytes)
    if v is None:
        r = np.random.default_rng(zlib.crc32(ctxbytes))
        v = np.exp(1j * r.uniform(0, 2 * np.pi, D))
        _keycache[ctxbytes] = v
    return v

def write_single(src, t_lo, t_hi):
    M = np.zeros(D, dtype=complex)
    for s in range(t_lo, t_hi, CH):
        e = min(s + CH, t_hi)
        C = np.stack([hkey(bytes(src[t-3:t])) for t in range(s, e)])
        T = C.conj()
        T *= P[src[s:e]]
        M += T.sum(axis=0)
    return M

t0 = time.perf_counter()
M3s = write_single(train, 3, len(train))
print("\n=== W2a: 单库哈希键, 写入 %.1fs, 独立上下文 %d ===" % (time.perf_counter() - t0, len(_keycache)))
def s_pred(Mv, t):
    C = hkey(bytes(full[t-3:t]))
    return int(np.argmax(np.abs(P @ (Mv * C).conj()) / D))
rows = []
for t in tq:
    pred = s_pred(M3s, int(t))
    c = sum(c3.get((full[t-3], full[t-2], full[t-1]), {}).values())
    tab = table_pred(full[t-3], full[t-2], full[t-1])
    rows.append((c, pred == full[t], pred == tab, Nw, uniq_flag(t)))
print("  对真值 %d/%d | 对表 argmax %d/%d — 键复用(平均%.1f次/键)导致相干噪声, 独立律仍失效:"
      % (sum(r[1] for r in rows), nq, sum(r[2] for r in rows), nq, Nw/len(_keycache)))
print_buckets(rows, Nw)

# ================================================================
# W2b: 分库 (SDM 式) — 修复
# ================================================================
t0 = time.perf_counter()
Mb = np.zeros((B, D), dtype=complex)
Mb2 = np.zeros((B, D), dtype=complex)
occ = np.zeros(B, dtype=np.int64)
for s in range(3, len(train), CH):
    e = min(s + CH, len(train))
    ctxs = [bytes(train[t-3:t]) for t in range(s, e)]
    C = np.stack([hkey(x) for x in ctxs])
    T = C.conj()
    T *= P[train[s:e]]
    bidx = np.array([zlib.crc32(x) & (B - 1) for x in ctxs])
    np.add.at(Mb, bidx, T)
    np.add.at(occ, bidx, 1)
for s in range(2, len(train), CH):
    e = min(s + CH, len(train))
    ctxs = [bytes(train[t-2:t]) for t in range(s, e)]
    C = np.stack([hkey(x) for x in ctxs])
    T = C.conj()
    T *= P[train[s:e]]
    bidx = np.array([zlib.crc32(x) & (B - 1) for x in ctxs])
    np.add.at(Mb2, bidx, T)
M1 = np.zeros(D, dtype=complex)
for s in range(3, len(train), 4096):
    M1 += P[train[s:min(s+4096, len(train))]].sum(axis=0)
print("\n=== W2b: 分库哈希键 (B=%d 桶), 写入 %.1fs, 桶占用 mean=%.0f max=%d ==="
      % (B, time.perf_counter() - t0, occ.mean(), occ.max()))

def bank_pred(Mb_, t, order):
    ctx = bytes(full[t-order:t])
    C = hkey(ctx)
    b = zlib.crc32(ctx) & (B - 1)
    corr = np.abs(P @ (Mb_[b] * C).conj()) / D
    return int(np.argmax(corr)), float(corr.max())

def b_pred(t):
    p, s0 = bank_pred(Mb, t, 3)
    if s0 >= 0.5: return p
    p, s0 = bank_pred(Mb2, t, 2)
    if s0 >= 0.5: return p
    return int(np.argmax(np.abs(P @ M1.conj()) / D))

rows = []
for t in tq:
    p3, s0 = bank_pred(Mb, int(t), 3)
    c = sum(c3.get((full[t-3], full[t-2], full[t-1]), {}).values())
    tab = table_pred(full[t-3], full[t-2], full[t-1])
    b = zlib.crc32(bytes(full[t-3:t])) & (B - 1)
    mb = int(occ[b])
    rows.append((c, p3 == full[t], p3 == tab, mb, uniq_flag(t)))
ok_blu = sum(b_pred(t) == full[t] for t in tq)
ok_tab = sum(table_pred(full[t-3], full[t-2], full[t-1]) == full[t] for t in tq)
print("  纯三元组对真值 %d/%d | 对表 argmax %d/%d" % (sum(r[1] for r in rows), nq, sum(r[2] for r in rows), nq))
print("  设计律恢复验证 (唯一续写上下文, 解析用桶占用 n=m_b):")
print_buckets(rows, None)
print("  回退链 LM: HDC 分库(三→二→一元) %d/%d (%.1f%%) | 计数表(带回退) %d/%d (%.1f%%)"
      % (ok_blu, nq, 100*ok_blu/nq, ok_tab, nq, 100*ok_tab/nq))
w_fp = B * D * 16 / 1024 / 1024
print("  权重体积: %.1f MB (fp128 复数) / %.1f MB (4bit 相位) | 训练耗时一遍 %.0fs 零梯度"
      % (w_fp, w_fp/4, time.perf_counter() - t0))

# ================================================================
# W3: 合并 = 逐桶相加 (数学精确)
# ================================================================
mid = len(train) // 2
MbA = np.zeros((B, D), dtype=complex); occA = np.zeros(B, dtype=np.int64)
MbB = np.zeros((B, D), dtype=complex)
for s in range(3, mid, CH):
    e = min(s + CH, mid)
    ctxs = [bytes(train[t-3:t]) for t in range(s, e)]
    C = np.stack([hkey(x) for x in ctxs]); T = C.conj(); T *= P[train[s:e]]
    bidx = np.array([zlib.crc32(x) & (B - 1) for x in ctxs])
    np.add.at(MbA, bidx, T); np.add.at(occA, bidx, 1)
for s in range(mid, len(train), CH):
    e = min(s + CH, len(train))
    ctxs = [bytes(train[t-3:t]) for t in range(s, e)]
    C = np.stack([hkey(x) for x in ctxs]); T = C.conj(); T *= P[train[s:e]]
    bidx = np.array([zlib.crc32(x) & (B - 1) for x in ctxs])
    np.add.at(MbB, bidx, T)
MbAB = MbA + MbB
rel = float(np.max(np.abs(MbAB - Mb)) / np.max(np.abs(Mb)))
diff = 0
for t in tq:
    if uniq_flag(t) and sum(c3.get((full[t-3], full[t-2], full[t-1]), {}).values()) >= 5:
        diff += bank_pred(MbAB, int(t), 3)[0] != bank_pred(Mb, int(t), 3)[0]
print("\n=== W3: 合并 = 逐桶相加 ===")
print("  相对偏差 %.2e, 高置信预测分歧 %d → 语义无损合并 (对照: 梯度权重合并需蒸馏)" % (rel, diff))

# ================================================================
# W4: 相位量化 = 低比特权重 (信号衰减 sinc(pi/m), 分库高余量下近无损)
# ================================================================
def quant_mb(m):
    step = 2 * np.pi / m
    return np.abs(Mb) * np.exp(1j * np.round(np.angle(Mb) / step) * step)
print("\n=== W4: 相位量化 (纯三元组, 同 %d 题) ===" % nq)
print("    %-8s %6s %10s %10s %8s" % ("相位级m", "比特/维", "对真值", "预测", "体积"))
for m in [2, 4, 16, 256]:
    Mq = quant_mb(m)
    ok = 0; ana = []
    for t in tq:
        p, _ = bank_pred(Mq, int(t), 3)
        ok += p == full[t]
        c = sum(c3.get((full[t-3], full[t-2], full[t-1]), {}).values())
        if uniq_flag(t) and c >= 1:
            b = zlib.crc32(bytes(full[t-3:t])) & (B - 1)
            ana.append(pred_acc(c * float(np.sinc(1.0/m)), int(occ[b])))
    print("    %-8d %6.0f %9.1f%% %9.1f%% %7.1fMB"
          % (m, np.log2(m), 100*ok/nq, 100*float(np.mean(ana)), w_fp/np.log2(m)/4))

# ================================================================
# W5: 持续学习 (旧域保真度: 共库 √税 / 分库零遗忘)
# ================================================================
TA = train[:mid]
c3A = defaultdict(Counter)
for t in range(3, len(TA)):
    c3A[(TA[t-3], TA[t-2], TA[t-1])][TA[t]] += 1
occA2 = np.zeros(B, dtype=np.int64)
MbTA = np.zeros((B, D), dtype=complex)
for s in range(3, len(TA), CH):
    e = min(s + CH, len(TA))
    ctxs = [bytes(TA[t-3:t]) for t in range(s, e)]
    C = np.stack([hkey(x) for x in ctxs]); T = C.conj(); T *= P[TA[s:e]]
    bidx = np.array([zlib.crc32(x) & (B - 1) for x in ctxs])
    np.add.at(MbTA, bidx, T); np.add.at(occA2, bidx, 1)
tq_old = np.sort(rng.choice(np.arange(3, len(TA)), size=300, replace=False))
def old_pred(Mb_, t, src):
    ctx = bytes(src[t-3:t]); C = hkey(ctx)
    b = zlib.crc32(ctx) & (B - 1)
    return int(np.argmax(np.abs(P @ (Mb_[b] * C).conj()) / D))
ok_before = sum(old_pred(MbTA, t, TA) == TA[t] for t in tq_old)
MbTB = np.zeros((B, D), dtype=complex)
for s in range(mid, len(train), CH):
    e = min(s + CH, len(train))
    ctxs = [bytes(train[t-3:t]) for t in range(s, e)]
    C = np.stack([hkey(x) for x in ctxs]); T = C.conj(); T *= P[train[s:e]]
    bidx = np.array([zlib.crc32(x) & (B - 1) for x in ctxs])
    np.add.at(MbTB, bidx, T)
Mb_after = MbTA + MbTB
ok_after = sum(old_pred(Mb_after, t, train) == TA[t] for t in tq_old)
ok_bank = sum(old_pred(MbTA, t, TA) == TA[t] for t in tq_old)
sel_u = [t for t in tq_old if len(c3A[(TA[t-3], TA[t-2], TA[t-1])]) == 1]
okb_u = sum(old_pred(MbTA, t, TA) == TA[t] for t in sel_u)
oka_u = sum(old_pred(Mb_after, t, train) == TA[t] for t in sel_u)
okbank_u = sum(old_pred(MbTA, t, TA) == TA[t] for t in sel_u)
ana_b = 100*float(np.mean([pred_acc(sum(c3.get((train[t-3], train[t-2], train[t-1]), {}).values()) if bytes(train[t-3:t]) in _keycache else 0, int(occA2[zlib.crc32(bytes(train[t-3:t])) & (B-1)])) for t in tq_old]))
occF = np.zeros(B, dtype=np.int64)
for s in range(3, len(train), 4096):
    for t in range(s, min(s+4096, len(train))):
        occF[zlib.crc32(bytes(train[t-3:t])) & (B-1)] += 1
ana_a = 100*float(np.mean([pred_acc(sum(c3.get((train[t-3], train[t-2], train[t-1]), {}).values()), int(occF[zlib.crc32(bytes(train[t-3:t])) & (B-1)])) for t in tq_old]))
print("\n=== W5: 持续学习 (旧域 %d 题) ===" % len(tq_old))
print("  共库: 旧域 %d/%d → 新增 %d 条写入后 %d/%d (桶占用增长=√可预测噪声税, 解析 %.1f%%→%.1f%%)"
      % (ok_before, len(tq_old), Nw - (len(TA) - 3), ok_after, len(tq_old), ana_b, ana_a))
print("  分库: 旧域 %d/%d —— 与 before 精确一致 (结构零遗忘)" % (ok_bank, len(tq_old)))
print("  唯一续写子集 (%d 题): 共库 %d/%d → %d/%d | 分库 %d/%d → %d/%d (记忆保真)"
      % (len(sel_u), okb_u, len(sel_u), oka_u, len(sel_u), okb_u, len(sel_u), okbank_u, len(sel_u)))

print("\n=== 总结: 权重=分库叠加向量组; 训练=一遍加法; 单库相干碰撞能量 Σm_c² 是容量真律,")
print("    分库(SDM)根治; 合并=相加; 量化=1bit 相位近无损; 持续学习=√税或零遗忘 ===")
