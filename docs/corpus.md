# 语料构建器说明书

所有语料**生成器**开源、可复现；数据集本体不入库（各自有上游许可，见 NOTICE.md），用脚本从官方渠道拉取。

## 课程类（十科，确定性生成器）

| 生成器 | 科目 |
|---|---|
| `curriculum_math_lab.py` | 数学 |
| `curriculum_mod_arith`(在 math 内) | 模运算 |
| `curriculum_causal_lab.py` | 因果 |
| `curriculum_csp_lab.py` | 约束满足 |
| `curriculum_fsm_lab.py` | 状态机 |
| `curriculum_graph_lab.py` | 图 |
| `curriculum_logic_lab.py` | 逻辑 |
| `curriculum_plan_lab.py` | 规划 |
| `curriculum_progexec_lab.py` | 代码执行 |
| `curriculum_seqind_lab.py` | 序列归纳 |

全套走「题面自包含 + 定向毒必翻 + 生成器内嵌通道 B 复核 + 退化 cap + 多通道对账」五条入场纪律——生成即验证，题题带标准答案与可复核链路。

打包：`build_subject_curriculum.py` 汇总成训练包；`build_segment_caches.py` 分科缓存。

## 语言/对话类

| 脚本 | 产物 | 上游 |
|---|---|---|
| `build_lang1_corpus.py` + `compress_lang1.py` | 语言打底 | 维基 + LCCC（`convert_lccc.py`） |
| `build_lang3b_corpus.py` / `build_lang3c_corpus.py` | 中文对白 | CPED（arXiv:2205.14727） |
| `build_lang4_slot.py` | 槽位计算 `[C:算式]=结果` | 自产（Calc-X 式，全求值验证） |
| `build_lang5_refusal.py` / `build_lang5_pack.py` | 中文拒绝语料（五类：暴/黄/隐私/违法/核心资产） | 自产合成 |
| `fetch_cped.py` / `fetch_esconv.py`（tools 遗产） | 上游拉取 | ESConv（thu-coai） |

## 混合池类

`build_v1_mix.py` / `build_804m_*.py` / `build_1p5b_*.py` / `build_pack_*.py` / `build_spec_pool.py` / `build_fmt*_corpus.py`（`fmt2_generators.py`/`fmt5_generators.py`）：多源按比例混合 + 打包缓存（npz）。配方细节与各版本差异见 `lab/docs/CORPUS_PLAN_LANG7.md`。

## 语料验尸（上卡前必跑）

```bash
python lab/train/peek_lang_corpus.py <语料文件或目录>
```

输出：汉字字节占比 / ASCII 占比 / 语种计数 / 窗口统计。**铁律**：小模型喂单语种，多语种留档不给小模型（事故记录见 training.md）。

## 上游协议速查

| 数据集 | 许可要点 |
|---|---|
| 维基 dump | CC-BY-SA（署名+相同方式共享） |
| CPED | 学术用途，见原仓库 |
| ESConv | 原仓库许可 |
| ROCStories / Opus-WritingPrompts | 原发布渠道条款 |
| prosocial-dialog | Apache-2.0（原仓库） |

凡上游协议不明确的，宁可跳过（有「诚实跳过」记录：OASST2 schema 不匹配、部分镜像 401——不硬闯）。
