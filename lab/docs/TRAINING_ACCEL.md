# 训练加速专项（81.13）— 自研快核路线

目标：1.5B × 十三科全弹药 × 10 万步，2-3 天训完。质量零砍（所有杠杆数学等价/无损）。

## 杠杆栈（目标合计 ~32×）
| 杠杆 | 状态 | 预期 | 质量影响 |
|---|---|---|---|
| 1. 分块线性注意力 | ✅ GEMM 形式已实现+CPU 对账 PASS（worst 1.91e-06） | 块内 FLOPs 3·T·D²→2·T·C·D+T·D²（D2048 下 ~10×） | 数学等价 |
| 2. fp16 AMP | Kaggle T4 实测中；DCU fp16 待验 | 2× | 无损（GradScaler） |
| 3. 打包+长度分桶 | 待做（数据层） | 1.4× | 无损（同数据去 pad 浪费） |
| 4. torch.compile | 待实测 | 1.3× | 无损 |
| 5. 4 卡数据并行（昆山 ¥2.53/卡时） | 需 RCCL DDP 验证 | 3.5× | 无损 |
| 6. Wq/Wk/Wv 三投影融合单 matmul | 待做 | 1.1-1.2× | 等价 |
| 7. 课程数据预分词缓存（uint8 静态数组） | 待做 | 消除每 epoch 重复 tokenize | 无损 |
| 8. 自研三值训练核（长期，C 内核线） | 规划 | 8-16×（终版武器） | STE 对账 |

## FLOPs 硬账
1.5B ≈ 442 TFLOP/步（B64×T256，训练≈3×前向）。10 万步 = 4.4e19 FLOPs。
- 昆山 4 卡 fp16（4×47 TFLOPS × 30% MFU ≈ 56 有效）：~220h ≈ 880 卡时 ≈ ¥2200（中性）
- 叠满杠杆 32× 后：单步 1.2-2.5s → **2-3 天达成**
- Kaggle T4×2 免费 30h/周：实验+中模型产线用，1.5B 显存不够（AdamW 状态 24GB>16GB）

## 分阶段验收
1. ✅ CPU 等价对账（aim_lm_fast.py test）
2. 🔥 Kaggle T4 GPU 实测（bench-linattn-fast kernel v5）
3. DCU fp16 + chunked 移植实测（DTK 注意：bf16 SDPA 已知坏，fp16 待验）
4. 昆山 4 卡 DDP 冒烟
5. 终训排期

## T4 实测战报（2026-10-29，kernel v4/v5）
**显存链（v1→v4 OOM 逐城攻破，每条都是 1.5B 终训的必踩坑）**：
1. naive B64 cum_kv 4.3GB+反向图 > 14.56GB → naive 必须 B/4 折算（本身即证据）
2. chunked 预物化 kv（12.9GB）→ einsum 移进块循环
3. autograd 为 einsum 反向保存每块 cum_local（1GB×4块×3层=12.9GB）→ **块级梯度检查点**（use_reentrant=False，省显存花 ~30% 重算税）
4. D2048 单块 cum_local = B8×C64×D2048²×4B = **8GB** → v5 改 B2/chunk16/fp16（268MB/块）

**速度读数（T4，诚实口径：T4 只证可行性，斜率以 DCU 为准）**：
- 等价性：T4 GPU worst 4.3e-05 ✅（fp32 精度内，CPU 1.67e-06）
- naive 在 30M 档（D512/L6/B32）直接 OOM = chunkwise 必需品第二实证
- T4 无 bf16，fp16 autocast 反而更慢（cast 税 > 收益，访存瓶颈型算子不吃 tensor core）——AMP 判决要等 DCU fp16
- 1.5B 形状（D2048/L12/B2/chunk16/fp16）fwd+bwd 装下：**峰值 10.75GB，全链跑通**——显存红线解除，DCU 64GB 无压力
- **但 cumsum 分块形式吞吐极低**（35.8 tok/s，~1.5% MFU）：真凶=块内 cumsum 是访存型+batched matvec 不吃张量核 → 催生 GEMM 形式（v6）

**GEMM 形式（v6，flash-linear-attention 正宗姿势）**：
- 块内因果和 = `(q@kᵀ ⊙ Mtril)@v` + 块间 `q@S`，状态转移 `S += kᵀ@v`（bmm）
- 零 cumsum、零 (B,C,D,D) 中间张量，全部胖 GEMM（张量核友好）
- 块内 FLOPs：cumsum 形式 3·T·D² → GEMM 形式 2·T·C·D + T·D²（D2048/chunk16 下 ~10×↓）
- CPU 等价 PASS worst 1.91e-06；T4 三方对比 v6 已实测
- 工程教训：因果和的正确 GEMM 形式是掩码作用在打分矩阵（逐元素 ⊙），`q@M@kᵀ` 是错的（对 query 维求和）；维度报错救了等价性

## T4 v6 终表（2026-10-29，同卡同配置三方对比）
| 配置 | cumsum 分块 | GEMM 分块 | GEMM+fp16 | GEMM 加速比 |
|---|---|---|---|---|
| 1.69M（D256/L3/B64） | 1538ms | 84.5ms | **47.1ms** | **32.7×** |
| 30M（D512/L6/B32） | 6459ms | 388ms | **200ms** | **32×** |
| 1.5B 形状（D2048/L12/B2） | 14308ms | — | **532ms** | **27×** |

- GEMM+fp16 双杠杆叠加后 AMP 终于生效（84.5→47.1ms = 1.8×）：cumsum 形式下 AMP 反而更慢，GEMM 形式下 fp16 真吃张量核
- 1.5B 形状单 T4：962 tok/s（v5 cumsum 形式仅 35.8，**27×**），峰值显存 9.21GB，T4 有效算力 ~40% MFU（健康训练核读数）
- **外推已被实测替换，见下节**

## DCU 单卡真实斜率实测（2026-10-29，同日收官，纸面外推全部作废）
脚本 `dcu_fast_bench.py`（build-tools），结果落 `dcu_harvest/dcu_fast_results.json`：
- 等价性 fp32：2.92e-06 ✅；**DTK fp16 autocast 直接可用**（bf16 SDPA 仍禁用——fp16 定案）
- 804M（D2048/L24）：**7365 tok/s，40.2 TFLOPS，21.6GB**
- **1.5B（D2048/L44，14.8 亿参数）：3994 tok/s，39.9 TFLOPS，39.4GB**
- 40 TFLOPS ≈ 50% MFU——GEMM 核完全脱离访存瓶颈；39.4GB+AdamW 11.8GB≈51GB < 64GB，**1.5B 全参单卡装得下**

**单卡训练时间硬账（实测替换外推）**：
| 阶段 | Token 量 | 单卡耗时 |
|---|---|---|
| v1 首训（37.6K 十科） | ~2000 万 | **1.4h** |
| v2 全量（推理+情绪+创意+安全） | ~3 亿 | **~21h** |
| 用户原目标 | 1.5B 两三天 | **被击穿到一天以内** |

**通往真训的剩余清单**：
1. GEMM 核替换进正式训练脚本（lab 的 aim_lm_dcu 系）
2. 数据预分词缓存（uint8 静态数组）——4K tok/s 的喂速不能用每 epoch 重 tokenize
3. lr 3e-4 配方已定版（本日 SCNet 扫描，16M 档）
4. 4 卡 DDP 变成「可选加速 3.5×」而非必需——单卡已达标
5. T4 v7（compile+批量战役）仍在跑，只影响 T4 产线，不阻塞 DCU 终训

## 终训排期（2026-10-29 定稿，用户拍板「单卡 1.5B 压在两三天」→ 已击穿至 ~30h）

**前置准备实况（当晚）**：
- ✅ 混料器 `build_v1_mix.py`：v1_mix.jsonl = **430,942 行 ≈ 1.46 亿 token**（十科×2 主粮 + 情绪 13 万 + 创意 9.6 万 + 拒绝 6.5 万，任务路由前缀 [情绪:/创意:/安全:] 分科，每源 1% 抽 val=8376）
- ✅ 真训脚本 `aim_train_real.py`：字节预分词缓存（npz 138MB 秒载）+ GEMM 分块注意力块级 checkpoint + 回答区 CE + AdamW(0.9,0.95)+warmup+cosine + AMP fp16 GradScaler + grad clip + 断点续训（AIM_* 环境变量一码两跑 DCU/本地）
- ✅ **①微型冒烟 PASS**（本地 CPU）：0.25M 档 150 步全管线跑通，loss 2.74→2.07 正常下降，checkpoint 往返正常，CPU 4454 tok/s；T=256 小档丢弃长样本属预期，T=768 覆盖 94.4%
- ⏭️ ②Kaggle T4 804M（cache 138MB 传 Kaggle dataset + kernel）→ ③开机（804M DCU 实弹 + 1.5B 双 lr）
- **v1 首训时间账更新**：混料 1.46 亿 token ÷ 3994 tok/s ≈ **10h**（原估 1.4h 是按 2000 万 token——开源语料全量进场后 v1 即第一轮 epoch，v2 从它 warm-start 续训；v1 10h + v2 ~25h ≈ 35 卡时，额度内但偏紧，可选 v1 抽样 7000 万 token 减半）

**两级训练 × 三级平台阶梯**（免费平台吃 bug，卡时只花 DTK 特有验证）：

| 阶段 | 模型/数据 | 平台 | 时长 | 目的 |
|---|---|---|---|---|
| ① 微型冒烟 | 1.6M（D256/L3）× 真管线 | 本地 CPU（7.8GB 内存上限→微型档） | ~1h | 全管线正确性：真数据+交叉熵+AdamW+checkpoint |
| ② 804M 短跑 | 804M（D2048/L24）× 500 万 token | Kaggle T4（免费）+ 8-bit Adam | ~2h | 规模化+优化器内存策略 |
| ③ 开机① | 804M DCU 实弹 + **1.5B 双 lr 短跑**（2e-4 vs 3e-4） | SCNet 单卡 | ~4h GPU | DTK 真训 fp16/显存/喂速实测；**1.5B 档 lr 配方定版**（16M 档 3e-4 不能直接抄） |
| ④ 训练窗口期 | 推理科程序化扩产 50-100 万（merge_curriculum 放大，本地 CPU 近零成本） | 本地 | 与③后并行 | 终训弹药大头 |
| ⑤ **v2 终训** | **1.5B（D2048/L44/H1=4096）× ~3 亿 token × 定版配方，v1 权重 warm-start** | SCNet 单卡一发 | **25-32h** | 终版权重 |

**首训≠终训**：v1（~2000 万 token，十科+存量）只定配方；v2 全量（推理 50-100 万+情绪 50 万+创意 5 万+拒绝 4800）才是正式，量级差 15×。v1 产物不扔，作 v2 初始化。

**关键路径=语料**：情绪 7.3 万/50 万 + 创意 2.2 万/5 万 + 拒绝 1537/4800 在跑（~3 天追平）；十科 37.6K 已定版。全程预计 5-7 天终版落地，总卡时 ~40（免费额度内）。

## 语料补给终版（2026-10-29 下午，用户拍板「不充值，靠开源扒+过闸」）

**背景**：zhipu 余额耗尽（上午 429 风暴=限流+额度双重根因），kimi 剩 ¥5 保全。蒸馏线全停，改走开源语料 mapper/过闸管线（零 API 成本）。

**hf-mirror 六矿已在库**（open_corpus/，455MB，今晨下载）：PsyQA、Aya、OASST1/2、emotional-support-zh、Opus-WritingPrompts、ROCStories；另补下载 prosocial-dialog（allenai）117MB。

**mapper/过闸战果**（open_corpus_mapper.py / mapper2.py）：
| 源 | → 库 | 入库 | 闸门 |
|---|---|---|---|
| PsyQA train.json | emp2 情绪（zh） | **15,550** | 关键词→十情绪词表映射 + 长度 + CJK 纯度 + 广告/联系信息封禁 + 去重 |
| Opus-WritingPrompts | creative（en） | 2,288 | 长度 500-6000 + 英文纯度 + 去重 |
| ROCStories | creative（en） | **67,744** | 5 句短篇 150-1200 字符 |
| prosocial-dialog（needs_caution/intervention） | refusal（en） | **61,952** | context→response 对 + 长度 + 英文纯度 + 去重 |
| OASST2/Aya | —— | 0（跳过） | 通用指令对话，与情绪十词表 schema 不匹配，诚实弃用 |
| emotional-support-zh | —— | 0（跳过） | 6253 条只有 5 个会话标题，多样性不及格 |

**三库终版盘面**：
- 情绪 emp2：**93,173**（7.7 万蒸馏 + 1.56 万 PsyQA）
- 创意：**95,537**（2.55 万蒸馏 + 7 万开源，ROC 占比 71%）
- 拒绝：**64,806**（2854 蒸馏 10 语种 + 6.2 万 prosocial en 谨慎/干预回应）
- 十科：37.6K（定版）+ 推理科 v2 训练期程序化扩产 50-100 万（零成本，待排产）

**工程教训**：①THUDM/Safety-Prompts 镜像 401（gated），PKU-SafeRLHF-30K 镜像空仓——zh 安全刮取此路不通，zh 拒绝靠蒸馏存量；②prosocial 的回应是「谨慎质询」不是硬拒绝，与蒸馏的礼貌拒绝互补，训练配比控制占比；③情绪 keyword 映射有噪声（如自伤情境误标愤怒），v2 混训占比控制。

**血泪钉子**：SCNet 容器出网被禁，relay 上传自始静默失败——收割一律走 SFTP（_scnet_harvest.py），31 文件已落 dcu_harvest/（commit 8e106e6）。

**lr 配方定版战报（同日 SCNet，scale16M/D256/L3/30K步/同数据）**：
- lr 3e-4：best val_exact **0.1904** / 终局 ans 0.6165 / 9393s ✅ 定版
- lr 5e-4：best val_exact 0.1680 / 终局 ans 0.6035 / 9732s
- 3e-4 相对高 13%，两指标全胜 → 16M 档基准配方 = lr 3e-4
