# NOTICE / 第三方内容归属

本仓库包含或引用的第三方内容，按来源列明：

## 1. Microsoft BitNet 权重（外部下载，不入库）

- 模型：[microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- 版权：© Microsoft Corporation，**MIT 许可**
- 分发说明：本仓库 Releases（tag `weights-v1`）分发该权重的 **AIM-W 打包格式转换版**（`m2b_tern.aimw`）与配套分词头。MIT 许可允许再分发，本仓库保留微软版权归属声明、不修改权重数值；也可从微软官方渠道下载原始 gguf 自行转换（`tools/bitnet2aimw.py`、`tools/gguf2aimw.py`）。
- "BitNet" "b1.58" 为 Microsoft 的研究成果名称，本文档中的引用仅为技术描述需要。

## 2. 维基百科知识库成品（GitHub Releases 分发）

- 内容来源：中文维基百科 dump
- 许可：**CC-BY-SA 3.0 / 4.0**（维基百科文本许可）
- 署名：中文维基百科贡献者（每张卡的 `source` 字段保留原文链接可溯源）
- 再分发条件：同以 CC-BY-SA 许可、保留署名与原文链接。本仓库的成品库文件（cards.jsonl + FTS 索引）满足上述条件。

## 3. 语料上游数据集（通过脚本获取，不入库再分发）

以下数据集由 `lab/train/` 中的获取脚本从官方渠道拉取，本仓库不重新分发其内容，使用前请遵守各自许可：

| 数据集 | 来源 | 用途 |
|---|---|---|
| CPED | scut-bds/CPED (arXiv:2205.14727) | 中文多轮对白 |
| ESConv | thu-coai/Emotional-Support-Conversation | 情绪支持对话 |
| PsyQA | 志愿者公开数据 | 中文心理问答 |
| ROCStories / Opus-WritingPrompts | 官方发布 | 创意写作 |
| prosocial-dialog | allenai | 谨慎回应/干预语料 |
| LCCC | thu-coai | 中文对话打底 |
| 中文维基 dump | Wikimedia Downloads | 知识库打底 |

## 4. 推理内核参考

三值推理内核的实现参考了 BitNet b1.58 论文（Frostig et al., Microsoft Research）公开描述的 1.58-bit 三值权重计算方法，代码为独立实现，无 Microsoft 代码复制。

## 5. 渲染级联

Edge 渲染使用系统自带 Microsoft Edge 浏览器（Playwright `channel=msedge`），不随仓库分发任何浏览器二进制。
