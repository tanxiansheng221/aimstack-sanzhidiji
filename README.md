# aimstack

> Fully self-built local AI stack — from scratch: ternary C inference engine (zero third-party runtime), plugin bus, multi-engine web crawler with quality gates, FTS knowledge base, and the complete training pipeline + research logs.

**中文：一个从零开始的全自研本地 AI 栈。** 没有框架、没有 PyTorch 依赖（训练对账用完即弃，产物零外部运行时）、没有现成引擎——C 推理核、插件总线、爬虫、知识库、训练器，每一层都是手写的。开发机是一台 2018 年买的双核笔记本，至今没换。

🎬 **UI 全链演示 (6:59，在线播放)** —— 对话直答 / 知识库引答 / 联网搜索-爬取-入库-引答 / 插件总线直调，全程 2018 年双核笔记本实机录屏：



https://github.com/user-attachments/assets/d4f897d0-001c-4dc6-b773-324c69b16a02


```
你说话
  │
  ▼
UI (server.py, 纯 stdlib)          ← 本地 127.0.0.1:8600
  │
  ▼
C 引擎 (engine/, 纯 C, 零第三方运行时)
  ├─ 三值推理核 (BitNet b1.58 式 2-bit 打包 + SIMD 快路径)
  ├─ 自研分词器 (字节级 + BPE)
  └─ 插件总线 (约百行 C: caps.json 注册表 → 拉起 Python 插件 → JSON 信封)
        │
        ▼
Python 插件 (纯 stdlib)
  ├─ kb.search / kb.add     ← 外置知识库 (FTS+向量混合检索, 查重闸)
  ├─ web.search / web.ingest← 多引擎搜索 (bing/百度/360 并行 + Edge 渲染级联 + 质量五信号闸)
  └─ demo.time              ← 最小冒烟插件
```

模型不感知插件的存在：路由（确定性代码）决定调什么插件，插件把料喂给模型，模型只负责读料写答案。**模型 = 读者，插件 = 图书馆员。**

## 快速开始

```bash
git clone https://github.com/tanxiansheng221/aimstack-sanzhidiji.git
cd aimstack-sanzhidiji

# 1. 编译引擎 (Windows, 双架构见 docs/engine.md)
cd engine && build.cmd && cd ..

# 2. 自建知识库 (中文维基预切块 -> FTS 库, 见 docs/kb.md)
python lab/kb/build_wiki_base.py <切块目录>

# 3. 微软 BitNet 权重: 自行下载 gguf (hf.co/microsoft/bitnet-b1.58-2B-4T, MIT,
#    国内可走 hf-mirror.com) -> 转换为 AIM-W + 分词器头, 见 docs/weights.md
python tools/bitnet2aimw.py <权重.gguf> data/m2b_tern.aimw
python tools/make_bnet_head.py

# 4. 起服务
python ui/server.py        # http://127.0.0.1:8600
```

详细说明：[docs/quickstart.md](docs/quickstart.md)

## 目录

| 目录 | 内容 | 说明书 |
|---|---|---|
| `engine/` | C 推理引擎 + 三值核 + 分词器 + 插件总线 | [docs/engine.md](docs/engine.md) |
| `plugins/` | 5 个 Python 插件 + caps.json 注册表 | [docs/bus.md](docs/bus.md) |
| `ui/` | 本地 Web UI | [docs/quickstart.md](docs/quickstart.md) |
| `lab/creeper/` | 多引擎搜索 + 渲染级联 + 质量闸 | [docs/creeper.md](docs/creeper.md) |
| `lab/kb/` | 知识库引擎 (FTS 索引/查重闸/入库) | [docs/kb.md](docs/kb.md) |
| `lab/train/` | 完整训练管线 (无自动微分训练栈 + 十科课程生成器 + 语料构建器) | [docs/training.md](docs/training.md) |
| `lab/seeds/` | 实验种子 (op18 双种子 / cot_mech 复现) | [docs/training.md](docs/training.md) |
| `lab/results/` | 实验结果 JSON (物证) | — |
| `lab/docs/` | 117 章 CPU 原生推演全记录 + E6 负面结果档案 | 先看这个再改代码 |
| `tools/` | 探针/基准测试脚本 | — |

## 硬件宣言：2018 年的笔记本兜住了下限

整个栈的开发机是一台 **2018 年买的戴尔笔记本**，至今（2026）没换过：

| 部件 | 规格 |
|---|---|
| CPU | Intel Core i7-7500U @ 2.70GHz（**2 核 4 线程**，低压 U） |
| 内存 | 8GB DDR4-2133 单通道 |
| GPU | GeForce 940MX（太弱，弃用——整套栈纯 CPU，不碰显卡） |
| 硬盘 | 512GB SSD + 1TB HDD |

这台机器实测兜住的数字：

- 三值核 SIMD 快路径：**85.53 tok/s**（从 2.08 优化 41 倍，C/NumPy 逐位对账保持绿）
- BitNet b1.58 2B（AIM-W 0.87GB）：**3.20 tok/s** 双工人（串行 1.99，引擎内自校准护栏）
- 本地训练甜点区：**5M-30M 参数**；152M 全量本机约 35 天（所以训练上免费云容器，栈本身不设硬件门槛）

**下限是我兜住了，上限需要大家来测**——这套栈没写一行硬件特定代码，多核、AVX-512、大内存的表现我们没有机会验证，欢迎真机数据。

## 训练资产（全部公开）

- 无自动微分训练栈（手写反传，selftest 29 参数对账 rel 5.7e-7）
- 十科课程生成器 + 语料构建器全套（模运算/因果/约束/图/逻辑/数学/规划/序列/状态机/代码）
- 语料验尸纪律、回放比例、三配方负结果、四臂算术对照零泛化的完整档案
- 详见 [docs/training.md](docs/training.md) 与 [lab/docs/](lab/docs/)

**正面结果和负面结果一起发**——负面结果档案（E6_CAMPAIGN_AND_RECIPE_DOSSIER）可能是这个仓库里最值钱的部分。

## 权重归属声明（必读）

- **BitNet b1.58 2B 权重 © Microsoft Corporation，MIT 许可**。本仓库的 AIM-W 格式（`.aimw`）只是该权重的**转换层/打包格式**，不是我们训练的模型。转换工具在 `tools/bitnet2aimw.py`，权重本体请从微软官方渠道（或 hf-mirror.com 镜像）下载。
- 自研模型（152M 等）的权重**不在本仓库发布**，本仓库只发布训练它的完整管线。
- 知识库自建自中文维基 dump，**CC-BY-SA**，署名维基百科贡献者。见 [NOTICE.md](NOTICE.md)。

## 协议矩阵

| 资产 | 协议 |
|---|---|
| 全部代码 | **AGPL-3.0**——把它做进你的联网产品？要么开源你的衍生，要么来谈商业授权 |
| `lab/docs/` 研究文档 | CC-BY-NC 4.0（署名-非商业） |
| 微软权重（外部，自行下载） | MIT，归 Microsoft |
| 维基内容（自建库，源自 dump） | CC-BY-SA，归维基百科贡献者 |

## 纪律

- 爬虫遵守 robots.txt，拒绝不级联渲染；搜索页仅作查询接口，正文一律来自第三方页面
- 质量闸 fail-closed：查重 + 五信号（残文长度/壳密度/句子结构/信息密度/菜单味），全不过不入库
- 知识库引擎、插件、爬虫全部纯 stdlib，零第三方依赖

## 已知边界（诚实清单）

- 2B 模型理解力是天花板：管线供料没问题，读多深是模型的事
- support.mozilla.org 等 JS 环境检测站抓不了（技术墙，非 robots）
- gov.cn 部分站点 SSL 证书链异常，按安全纪律拒绝而非跳过校验
- Windows 优先（Edge 渲染级联用系统自带浏览器）；Linux 编译过，渲染级联需装 Chromium

## 联系 / Contact

- 商务合作 / 商业授权 / 问题反馈：**t8070824207@qq.com**
- 运营主体：深圳市罗湖区清水河街道锋爪数创数字技术工作室（个体工商户）
- 统一社会信用代码：92440300MAKDFBKUXT
