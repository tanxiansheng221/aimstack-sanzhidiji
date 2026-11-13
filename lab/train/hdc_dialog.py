# -*- coding: utf-8 -*-
# hdc_dialog.py — 第二容器的「对话与推理」：多轮记忆 / 二跳问答 / 整机对话流
#   实验 D: 多轮对话事实记忆 —— 边聊边记, 容量律验算 (噪声 ∝ √K/D)
#   实验 E: 二跳推理问答 —— 知识图谱叠加存储, 验算理论噪声预测
#   实验 F: 六轮整机对话流 —— 陈述/提问/算术/指代/拒答/多跳 混合脚本逐轮计分
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(23)
D = 16384

def rphase(n):
    return np.exp(1j*rng.uniform(0, 2*np.pi, size=(n, D)))

def sim(a, b):
    return float(np.abs(np.vdot(a, b)) / D)

def noisy(v, rho):
    idx = rng.choice(D, size=int(D*rho), replace=False)
    vn = v.copy(); vn[idx] = np.exp(1j*rng.uniform(0, 2*np.pi, size=len(idx)))
    return vn

# ================= 实验 D: 多轮对话事实记忆 (边聊边记) =================
print("=== 实验 D: 聊了 N 轮之后, 第 1 轮说的事还记得吗 (D=%d) ===" % D)
ED = 400                                  # 实体词典 (人名/物品/颜色...)
E = rphase(ED)
def ask_mem(M, key, codebook, thresh=0.5):
    """codebook 回忆 + 相对间隔置信度(尺度无关; z 分数在 8 候选小集上会被信号抬高 std 反噬)"""
    R = M * key
    corr = np.abs(codebook @ R.conj()) / D
    s = np.sort(corr)[::-1]
    conf = (s[0] - s[1]) / (s[0] + 1e-9)
    if conf < thresh:
        return None
    return int(np.argmax(corr))
def cb_index(codebook, v):
    return int(np.argmax(np.abs(codebook @ v.conj())))
for K in [100, 300, 600, 1000]:
    # 组1: 唯一键 (纯容量律)
    keys_u = rphase(K)
    codebook = rphase(8)                            # 8 种"偏好值"
    vidx = rng.integers(0, 8, size=K)
    vals = codebook[vidx]
    Mu = (keys_u.conj() * vals).sum(axis=0)         # 每轮一句, 一次写入
    q = rng.integers(0, K, size=100)
    ok_u = sum(1 for i in q if ask_mem(Mu, keys_u[i], codebook) == vidx[i])
    # 组2: 词典键 (400 池有放回 — 键碰撞=真实歧义)
    keys_d = E[rng.integers(0, ED, size=K)]
    vidx2 = rng.integers(0, 8, size=K)
    Md = (keys_d.conj() * codebook[vidx2]).sum(axis=0)
    ok_d = sum(1 for i in q[:100] if ask_mem(Md, keys_d[i], codebook) == vidx2[i])
    noise_pred = np.sqrt(K) * 1.25 / np.sqrt(D)     # 理论噪声下限 √K/√D
    print("  聊 %5d 轮: 唯一键回忆 %d/100 | 词典键(400池) %d/100 | 理论噪声底 %.3f" % (K, ok_u, ok_d, noise_pred))

# ================= 实验 E: 二跳推理问答 (知识图谱叠加) =================
print("\n=== 实验 E: 二跳问答 A --r1--> B --r2--> C, 容量律验算 (实体无放回, 消歧义) ===")
NE = 4096                                 # 实体字典(≥最大边数, 保证头实体唯一)
Edict = rphase(NE)
def build_kb(K):
    """K 条 r1 边 + K 条 r2 边; 头/中/尾均无放回 → 无键歧义, 纯噪声容量"""
    perm1 = rng.permutation(NE)[:2*K]
    heads1, mids = perm1[:K], perm1[K:]
    tails = rng.permutation(NE)[:K]
    M1 = (Edict[heads1].conj() * Edict[mids]).sum(axis=0)
    M2 = (Edict[mids].conj() * Edict[tails]).sum(axis=0)
    return M1, M2, heads1, mids, tails
def decode_ent(v):
    corr = np.abs(Edict @ v.conj()) / D
    return int(np.argmax(corr))
for K in [50, 200, 800, 1600]:
    M1, M2, h, m, t = build_kb(K)
    q = rng.integers(0, K, size=100)
    ok1 = ok2 = 0
    for i in q:
        R1 = M1 * Edict[h[i]]
        m_hat = decode_ent(R1)
        ok1 += (m_hat == m[i])
        R2 = M2 * Edict[m[i]]                      # 用真中间体(排除一跳误差传递)
        ok2 += (decode_ent(R2) == t[i])
    noise_pred = np.sqrt(K) * 1.25 / np.sqrt(D)
    print("  每关系 %5d 条边: 一跳 %d/100, 二跳 %d/100 | 理论噪声底 %.3f" % (K, ok1, ok2, noise_pred))

# ================= 实验 F: 六轮整机对话流 =================
print("\n=== 实验 F: 六轮整机对话 (陈述/算术/提问/指代/拒答/多跳) ===")
codebook = rphase(8)
PREF = {"蓝": 0, "红": 1, "绿": 2, "黑": 3}
col_vec = {k: codebook[v] for k, v in PREF.items()}
M_facts = np.zeros(D, dtype=complex)             # 海马体: 事实叠加库
S_speaker = rphase(1)[0]                         # 工作寄存器: 当前说话人(指代解析用)
score = []

# 轮 1: 陈述 "我喜欢蓝色" → 事实写入
key = S_speaker * rphase(1)[0]                   # 说话人⊙话题(颜色)
fav = col_vec["蓝"]
M_facts += key.conj() * fav
score.append(("轮1 陈述+写入", "写入完成"))

# 轮 2: 算术 "13加29等于多少" → 小脑
k = rng.integers(0, 10, size=D)
m_mod = 100
P = np.exp(1j*2*np.pi*np.outer(np.arange(m_mod), k)/m_mod)
ans = int(np.argmax(np.abs(P @ (P[42] * P[29]).conj())))   # 42+29=71
score.append(("轮2 算术 42+29", "%d (真值 71)" % ans if ans == 71 else "%d 错" % ans))

# 轮 3: 提问 "我喜欢什么颜色" → 情景回忆
qkey = key                                        # 同 key 结构(指代解析: 我=说话人寄存器)
R = M_facts * qkey
corr = np.abs(codebook @ R.conj()) / D
pred = PREF["蓝"] == int(np.argmax(corr))
score.append(("轮3 回忆偏好", "答蓝 %s" % ("✓" if pred else "✗")))

# 轮 4: 问没说过的 "我有几个孩子" → 应拒答
Mz = np.zeros(D, dtype=complex)
kz = S_speaker * rphase(1)[0]
Rz = M_facts * kz
corr = np.abs(codebook @ Rz.conj()) / D
s = np.sort(corr)[::-1]
z = (s[0] - s[1]) / (np.std(corr) + 1e-9)
score.append(("轮4 未知问题", "拒答 %s (z=%.2f)" % ("✓" if z < 3.0 else "✗ 幻觉", z)))

# 轮 5-6: 二跳 "我朋友的朋友喜欢什么" — 图谱已在实验 E 构建(独立库)
M1, M2, h, m, t = build_kb(200)
i = int(rng.integers(0, 200))
R1 = M1 * Edict[h[i]]
m_mid = decode_ent(R1)
R2 = M2 * Edict[m_mid]
score.append(("轮5 二跳推理 A→B→C", "%s (中转 %s)" % ("✓" if decode_ent(R2) == t[i] else "✗", "✓" if m_mid == m[i] else "✗")))
score.append(("轮6 记忆无损点验", "轮1 事实仍可召回 %s" % ("✓" if (lambda c: PREF["蓝"] == int(np.argmax(c)))(np.abs(codebook @ (M_facts*qkey).conj())/D) else "✗")))

for tag, res in score:
    print("  [%s] %s" % (tag, res))
print("\n=== 对话=状态寄存器+叠加事实库; 推理=图谱绑定展开; 拒答=z 间隔; 全程零梯度 ===")
