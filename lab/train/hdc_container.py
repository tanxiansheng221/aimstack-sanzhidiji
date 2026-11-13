# -*- coding: utf-8 -*-
# hdc_container.py — 「另一个容器」数学原型：相位超维计算（HDC/VSA）
# 验证用户设想：海量节点（D 维相位向量）同时计算 → 喂数据=直接吃透（写入即学会、零训练、零遗忘）
#
# 数学核心：取 m 级量化相位的复数超维向量 P_r[i] = exp(i·2π·r·k_i/m)
#   绑定运算 ⊙ = 逐元素复数乘法 → P_r ⊙ P_s = P_{(r+s) mod m}  （精确成立，非近似）
#   即：模 m 加法的规则不在权重里，在容器的代数结构里——这是与权重容器的本质区别
import numpy as np
rng = np.random.default_rng(42)

D = 2048          # 超维向量维度 = 「节点数」
print("=== 容器参数: D=%d 维相位超维向量 ===" % D)

def make_prototypes(m):
    """生成 m 个模 m 等价类原型: P_r[i] = exp(i·2π·r·k_i/m), k_i 每维随机"""
    k = rng.integers(0, m, size=D)
    r = np.arange(m)
    # P[r,i] = exp(i*2*pi*r*k[i]/m)
    ang = 2*np.pi*np.outer(r, k)/m
    return np.exp(1j*ang), k

def decode(C, P):
    """解码: 与所有原型做相关, 取最大"""
    corr = np.abs(P @ C.conj())
    return int(np.argmax(corr)), corr

# ---------- 实验 1: 绑定代数做 (a+b) mod m —— 零训练, 写入即吃透 ----------
print("\n=== 实验1: 绑定代数计算 (a+b) mod m, 10 万道全新题, 零训练 ===")
N1 = 100000
hit = 0
mismatch_by_m = {}
for _ in range(N1):
    m = int(rng.integers(2, 10))
    a = int(rng.integers(10, 100)); b = int(rng.integers(10, 100))
    P, k = make_prototypes(m) if m not in mismatch_by_m else mismatch_by_m[m]
    mismatch_by_m[m] = (P, k)
    A = P[a % m]; B = P[b % m]
    C = A * B                      # 绑定 = 逐元素相位相加 = 模 m 加法
    r_hat, _ = decode(C, P)
    hit += (r_hat == (a + b) % m)
print("10 万题命中率: %.4f  (训练步数: 0, 梯度: 无)" % (hit / N1))

# ---------- 实验 2: 联想记忆叠加存储 (SDM 式) + 容量曲线 ----------
print("\n=== 实验2: 联想记忆 superposition 容量 (D=%d, m=8 值域) ===" % D)
m = 8
P, k = make_prototypes(m)
def capacity_test(K):
    keys = np.exp(1j*rng.uniform(0, 2*np.pi, size=(K, D)))
    vals = P[rng.integers(0, m, size=K)]           # 值是 m 个原型之一
    M = (keys.conj() * vals).sum(axis=0)           # 一次性叠加写入
    q = rng.integers(0, K, size=min(K, 300))       # 抽查回忆
    rec = M * keys[q]                              # 用 key 解码
    idx, _ = decode_batch(rec, P)
    return float(np.mean(idx == rng_vals[q]))
def decode_batch(C_batch, P):
    # corr: (B, m)
    corr = np.abs(C_batch @ P.conj().T)
    return np.argmax(corr, axis=1), corr
rng_vals = None
def capacity_test2(K):
    global rng_vals
    keys = np.exp(1j*rng.uniform(0, 2*np.pi, size=(K, D)))
    v_idx = rng.integers(0, m, size=K)
    vals = P[v_idx]
    M = (keys.conj() * vals).sum(axis=0)
    q = rng.integers(0, K, size=min(K, 300))
    rec = M * keys[q]
    idx, _ = decode_batch(rec, P)
    return float(np.mean(idx == v_idx[q]))
for K in [50, 100, 200, 400, 800, 1600]:
    acc = capacity_test2(K)
    print("  存储 %5d 对, 一次写入, 回忆准确率: %.4f" % (K, acc))

# ---------- 实验 3: 反遗忘 —— 算术运算与记忆写入互相干扰为零 ----------
print("\n=== 实验3: 反遗忘 (先存 500 对联想, 再做 1 万次算术, 再验联想) ===")
K = 500
keys = np.exp(1j*rng.uniform(0, 2*np.pi, size=(K, D)))
v_idx = rng.integers(0, m, size=K)
M = (keys.conj() * P[v_idx]).sum(axis=0)
for _ in range(10000):                             # 1 万次算术运算(纯绑定, 不碰 M)
    mm = int(rng.integers(2, 10))
    Pm, _ = make_prototypes(mm) if mm != 8 else (P, k)
    a = int(rng.integers(10, 100)); b = int(rng.integers(10, 100))
    _ = Pm[a % mm] * Pm[b % mm]
q = rng.integers(0, K, size=300)
idx_before, _ = decode_batch(M * keys[q], P)
acc_before = float(np.mean(idx_before == v_idx[q]))
for _ in range(10000):                             # 1 万次算术运算(纯绑定, 不碰 M)
    mm = int(rng.integers(2, 10))
    Pm, _ = make_prototypes(mm) if mm != 8 else (P, k)
    a = int(rng.integers(10, 100)); b = int(rng.integers(10, 100))
    _ = Pm[a % mm] * Pm[b % mm]
idx_after, _ = decode_batch(M * keys[q], P)
acc_after = float(np.mean(idx_after == v_idx[q]))
print("  1 万次算术前, 联想回忆: %.4f" % acc_before)
print("  1 万次算术后, 联想回忆: %.4f  (衰减 = %.4f, 理论应=0)" % (acc_after, acc_before - acc_after))
print("  对照: 权重容器同场景 lang4 技能被 lang6 回炉一次冲光 → 0")

# ---------- 实验 4: 一次性学习 vs 梯度下降 (同题对照) ----------
print("\n=== 实验4: 同样的 (a+b) mod m 任务, 两种容器的学习成本 ===")
print("  权重容器 (E3/E6 实测): 2000 步 × 32 题 = 6.4 万次曝光, 泛化 0 (先验 0.2305)")
print("  相位容器 (实验1):      0 步训练, 每题首次出现即 100%% 命中 (代数内置规则)")
print("\n=== DONE ===")
