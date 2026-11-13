# 微软权重转换说明书

**归属声明（先读）**：BitNet b1.58 2B 权重 © Microsoft Corporation（MIT）。本仓库只提供 AIM-W 打包格式的**转换层**与推理内核，不训练、不修改、不随仓库分发权重本体。

## 拿到权重

1. 从官方渠道下载：`huggingface.co/microsoft/bitnet-b1.58-2B-4T`（MIT，选 GGUF 格式）
2. 校验文件完整性

## 转换成 AIM-W

```bash
# 主转换器 (bitnet gguf -> .aimw 三值 2-bit 打包)
python tools/bitnet2aimw.py <下载的权重.gguf> data/m2b_tern.aimw

# 通用 gguf 转换 (fp 权重或 q4k 等)
python tools/gguf2aimw.py <in.gguf> <out.aimw>
python tools/gguf2aimw_v2.py <in.gguf> <out.aimw>     # v2 格式

# 转换后校验
python tools/validate_aimw.py data/m2b_tern.aimw
python tools/peek_aimw_full.py data/m2b_tern.aimw     # 结构预览
```

转换产物 ~0.87GB（2-bit 三值打包的收益：原 gguf 的 1/4-1/2 体积）。

## 对账工具（换权重后建议跑）

| 工具 | 用途 |
|---|---|
| `tools/bnet_oracle.py` | NumPy 侧 oracle 实现，与 C 引擎输出逐位对账 |
| `tools/make_bnet_head.py` / `make_mini_bitnet.py` | 小测试权重构造（秒级冒烟，不用 2B 大权重） |
| `tools/check_tok.py` / `test_tokrt.py` | 分词器对账 |

对账纪律：**任何内核改动必须 C/NumPy 逐位对账过绿再谈速度**（引擎说明书里的 41 倍优化就是在对账契约保持的前提下做的）。

## AIM-W 格式

- 定义与读写实现在 `engine/aimw.c/.h`
- 三值权重 {-1,0,+1} 打包进 2-bit，per-block scale
- 头部含架构/层数/维度元信息，`peek_aimw_full.py` 可打印

## 自研模型权重说明

自研 152M / 微核等模型**不在本仓库发布**。`lab/train/` 提供了完整训练管线，你可以用十科课程生成器（`lab/train/curriculum_*_lab.py`，模运算/因果/约束/图/逻辑/数学/规划/序列/状态机/代码十科）+ 语料构建器自己从零训一个小模型——这正是本仓库「训练资产全公开」的部分。

## 常见问题

- **转换报 shape 不符**：确认下载的是 b1.58 2B 的 gguf 且未被量化工具二次修改
- **推理乱码**：先跑 `tools/check_tok.py` 排除分词器对账问题，再查 gguf→aimw 转换日志
- **速度不对**：确认 `AIM_WORKERS=2` 与自校准输出（`AIM_PAR_DBG=1`）；沙箱环境 EcoQoS 会拖慢一个数量级
