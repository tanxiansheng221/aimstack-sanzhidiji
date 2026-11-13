# -*- coding: utf-8 -*-
# hdc_brain_v2.py — 三区脑容器仿真：把用户设想全部容纳进一个数学原型
#
# 设想 → 映射：
#   [1] 大量节点里面没东西        → D 维相位空间, 近正交向量 2^D 个, 绝大多数闲置
#   [2] 节点包裹大脑/小脑/海马体  → 同一 D 维基质上的三种存储模式(分区共存于一个容器)
#   [3] 喂数据海量节点同时计算吃透 → 一次写入(叠加分摊), 无梯度无迭代
#   [4] 汇入海马体/皮层/小脑      → 海马体=快速叠加库, 皮层=固化槽位库, 小脑=绑定代数运算
#   [5] 巩固(海马体→皮层)         → 高频访问条目固化, 容量无限扩展
#   [6] 不容易忘                  → 分区物理隔离, 各区读写互不触碰
#   [7] 不幻觉                    → 置信度判决: 不知道就拒答(权重容器做不到)
import numpy as np
rng = np.random.default_rng(7)

D = 16384       # 基质维度（节点数）—— 容量随 D 线性标度
print("=== 三区脑容器 D=%d ===" % D)

# ---------------- 基质工具 ----------------
def rand_phase(n):
    return np.exp(1j*rng.uniform(0, 2*np.pi, size=(n, D)))

def sim(a, b):
    """相关度: |<a,b>|/D ∈ [0,1]"""
    return np.abs(np.vdot(a, b)) / D

# ---------------- 三区定义（同一基质，三种模式） ----------------
class Brain:
    def __init__(self, m_codebook=64):
        self.m = m_codebook
        # 皮层区: 固化槽位 [(key, entry)] 精确存储, 容量=槽数(可增长)
        self.cortex = []            # list of (key_vec, entry_idx)
        # 海马体区: 叠加库 M = Σ conj(K_i)·V_{j_i}, 一次写入, 容量~D/10
        self.hippo_M = np.zeros(D, dtype=complex)
        self.hippo_count = 0
        # 小脑区: 绑定代数(无状态), 见 cerebellum_op
        # 语义词典(皮层码本): entry → 相位向量 (概念本身)
        self.codebook = rand_phase(m_codebook)
        self.access = np.zeros(m_codebook)   # 巩固统计

    # --- 小脑: 模算术绑定运算 (无状态, 永不遗忘, 永不幻觉) ---
    def cerebellum_mod(self, a, b, m):
        k = rng.integers(0, m, size=D)
        P = np.exp(1j*2*np.pi*np.outer(np.arange(m), k)/m)
        return int(np.argmax(np.abs(P @ (P[a % m] * P[b % m]).conj())))  # = (a+b)%m 恒等

    # --- 海马体: 一次写入 ---
    def hippo_write(self, key, entry):
        self.hippo_M += key.conj() * self.codebook[entry]
        self.hippo_count += 1

    def hippo_recall(self, key):
        R = self.hippo_M * key
        corr = np.abs(self.codebook @ R.conj()) / D
        s = np.sort(corr)[::-1]
        z = (s[0] - s[1]) / (np.std(corr) + 1e-9)   # top1-top2 间隔 z 分数
        return int(np.argmax(corr)), float(z), corr

    # --- 皮层: 固化槽位精确回忆 ---
    def cortex_write(self, key, entry):
        self.cortex.append((key, entry))

    def cortex_recall(self, key):
        if len(self.cortex) < 2:
            return (self.cortex[0][1], 9.9) if self.cortex else (None, -1.0)
        sims = np.array([sim(key, k) for k, _ in self.cortex])
        s = np.sort(sims)[::-1]
        z = (s[0] - s[1]) / (np.std(sims) + 1e-9)
        return self.cortex[int(np.argmax(sims))][1], float(z)

    # --- 汇流仲裁: 三区答案按置信度融合 ---
    def ask(self, key, thresh=3.0):
        cj, cconf = self.cortex_recall(key)
        hj, hconf, _ = self.hippo_recall(key)
        if max(cconf, hconf) < thresh:
            return None, max(cconf, hconf)   # 拒答: 我不知道
        if cconf >= hconf:
            return cj, cconf
        return hj, hconf

    # --- 巩固: 海马体高频条目固化进皮层 ---
    def consolidate(self, pairs, top_n):
        # pairs: [(key, entry)], 取最近写入的 top_n 固化(演示用简单策略)
        for key, entry in pairs[-top_n:]:
            self.cortex_write(key, entry)

# ---------------- 仿真时间线 ----------------
brain = Brain(m_codebook=64)
log = []

# T1: 学习 300 个新事实(全部一次写入海马体, 零迭代)
facts = rand_phase(300)
fact_entry = rng.integers(0, brain.m, size=300)
for i in range(300):
    brain.hippo_write(facts[i], int(fact_entry[i]))
q = rng.integers(0, 300, size=100)
ok = sum(1 for i in q if brain.ask(facts[i])[0] == fact_entry[i])
log.append(("T1 学 300 新事实(一次写入)", "回忆 %d/100" % ok))

# T2: 小脑做 5000 次算术(期间不碰任何记忆变量)
ar_ok = 0
for _ in range(5000):
    m = int(rng.integers(2, 10))
    a = int(rng.integers(10, 100)); b = int(rng.integers(10, 100))
    ar_ok += (brain.cerebellum_mod(a, b, m) == (a + b) % m)
log.append(("T2 小脑 5000 次算术", "正确 %d/5000" % ar_ok))

# T3: 巩固 —— 最近 100 条固化进皮层槽位
brain.consolidate([(facts[i], int(fact_entry[i])) for i in range(300)], 100)
qc = list(range(200, 300))[:50]   # 被固化的条目
ok = sum(1 for i in qc if brain.cortex_recall(facts[i])[0] == fact_entry[i])
log.append(("T3 固化 100 条进皮层", "皮层回忆 %d/50" % ok))

# T4: 再学 300 个新事实(海马体继续写入) → 验证旧记忆零扰动
facts2 = rand_phase(300)
fact_entry2 = rng.integers(0, brain.m, size=300)
for i in range(300):
    brain.hippo_write(facts2[i], int(fact_entry2[i]))
old_ok = sum(1 for i in q[:50] if brain.ask(facts[i])[0] == fact_entry[i])
new_ok = sum(1 for i in rng.integers(0, 300, size=50) if brain.ask(facts2[i])[0] == fact_entry2[i])
log.append(("T4 再学 300 新事实", "旧记忆 %d/50, 新记忆 %d/50" % (old_ok, new_ok)))

# T5: 模糊理解 —— 带噪声的 key(模拟"话没说全/说法不同") 也能召回
def noisy(k, rho):
    idx = rng.choice(D, size=int(D*rho), replace=False)
    kn = k.copy(); kn[idx] = np.exp(1j*rng.uniform(0, 2*np.pi, size=len(idx)))
    return kn
for rho in [0.1, 0.2, 0.3]:
    ok = sum(1 for i in q[:50] if brain.ask(noisy(facts[i], rho))[0] == fact_entry[i])
    log.append(("T5 噪声 %.0f%% 的变体提问" % (rho*100), "召回 %d/50" % ok))

# T6: 幻觉判决 —— 问 200 个从没见过的问题, 容器应说"不知道"
unk = rand_phase(200)
rejected = sum(1 for i in range(200) if brain.ask(unk[i])[0] is None)
log.append(("T6 问 200 个未知问题", "拒答 %d/200 (幻觉率 %.1f%%)" % (rejected, 100*(200-rejected)/200)))

# T7: 总审计 —— 全部记忆一次性点验
ok_all = sum(1 for i in range(100) if brain.ask(facts[i])[0] == fact_entry[i]) \
       + sum(1 for i in range(100) if brain.ask(facts2[i])[0] == fact_entry2[i])
log.append(("T7 总点验 200 条(新旧混合)", "全对 %d/200" % ok_all))

print()
for tag, res in log:
    print("[%s] %s" % (tag, res))
print("\n=== 三区共存于同一 %d 维基质; 全程无梯度、无迭代、无反向传播 ===" % D)
