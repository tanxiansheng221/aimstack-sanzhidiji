# Linux 多线程 + AVX-512 VNNI 路径 (可选) 与基准测试

本补丁为 `engine/forward.c` 增加了 POSIX 工作线程池和更快的三值/注意力路径, 全部通过**环境变量开启** (默认关闭)。不设任何变量时, 原有代码路径保持不变。它补上了 `engine.zh.md` 中 "POSIX 多 worker 需要 pthread 移植 (尚未完成)" 这一项。

测试环境: Ryzen 9 7945HX (16 核 / 32 线程, AVX-512 + VNNI), Linux, gcc 16, 用 `tools/bitnet2aimw.py` 转换的 `microsoft/bitnet-b1.58-2B-4T`。

## 编译

```bash
gcc -O3 -march=native -pthread -o aimp engine/aimw.c engine/forward.c engine/gguf.c engine/main.c engine/tokenizer.c engine/bus.c -lm
```

注意: 文档中的 `gcc -O2` (不带 `-march`) 会把 `__AVX2__` 路径编译掉, 本机解码只有约 1.6 tok/s; 未改动的引擎用 `-O3 -march=native` 为约 6.9 tok/s。

## 开关 (默认全部关闭)

| 环境变量 | 作用 |
|---|---|
| `AIM_WORKERS=N` | POSIX 线程池 (pthreads + C11 原子 + futex), N 个额外 worker; 按物理核优先绑核。另有 `AIM_MIN_ROWS` (16 线程建议 512)、`AIM_SPIN`、`AIM_PIN` |
| `AIM_VNNI=1` | int8 激活的三值内核 (逐向量 absmax, 即 BitNet W1.58A8), 使用 AVX-512 VNNI `dpbusd`; 整数累加, 所以多线程与串行结果完全一致 |
| `AIM_PREFILL_BATCH=k` | 批量内核 (每块最多 8 列, 权重只解包一次) 与批量预填充; 除最后一个 token 外跳过输出头 |
| `AIM_Q8V=1` | Q8 输出层的 int8 VNNI 内核 |
| `AIM_SPEC=k` | 精确的推测解码 (最长后缀 n-gram 起草, 草稿长度 1+3*匹配长度); 输出与普通贪心完全一致。原起草器存在位置错位一位的问题 |
| `AIM_LM4=1` | **有损**: 单 token logits 使用输出层的 4-bit 副本。文本会变化, 质量未测, 且与 `AIM_SPEC` 不兼容 |

另外: 按 (token, head) 并行的 AVX-512 注意力 (HD=128), 以及激活缓冲区填充 (2 的幂步长在 L1 中别名冲突, 占用约 40% 的 matvec 时间)。

示例 (16 物理核):

```bash
AIM_WORKERS=15 AIM_MIN_ROWS=512 AIM_VNNI=1 AIM_Q8V=1 AIM_PREFILL_BATCH=32 AIM_SPEC=8 \
  ./aimp data/m2b_tern.aimw data/m2b_llama3_head.gguf --gt "prompt" 64
```

## 结果 (BitNet b1.58 2B-4T, 仅 CPU, tok/s)

| | 预填充 | 解码 |
|---|---|---|
| 原版, `gcc -O2` | n/a | 1.6 |
| 原版, `-O3 -march=native`, 1 线程 | 7.1 | 6.9 |
| 本补丁, 1 线程 | 65 | 24 (`AIM_Q8V=1` 时 34) |
| 本补丁, 4 线程 | 239 | 53 |
| 本补丁, 8 线程 | 420 | 53 |
| 本补丁, 16 线程 (绑核) | 623-660 | 58-60 |
| llama.cpp b12110 (CPU, TQ2_0), 1 线程 | 47 | 23.4 |
| llama.cpp, 16 线程 | 490 | 42.6 |

叠加推测解码 (16 线程, 输出精确一致): 大量复制的文本 57 -> 230-360 tok/s, JSON 58 -> 81, 摘要 51 -> 85, 自由故事无收益 (58 -> 58, 约 2% 开销)。

说明:
- 解码受内存带宽限制 (约 54 GB/s); Q8 输出层约占每 token 字节数的 39%。
- 本补丁与 llama.cpp 都在 16 线程 (=物理核数) 达到峰值; SMT 兄弟线程没有收益, 32 线程略差。
- 引擎的 `TIME` 使用 `clock()` (各线程 CPU 时间之和), 比较多线程运行时请用墙钟时间。

## 正确性

7 个测试提示中有 6 个在 48 个 token 内与原浮点引擎逐 token 一致 (包括一个 308 token 的提示)。一个提示 (俳句) 在后段出现分歧: int8 激活会使近似并列的候选翻转。所有输出都流畅且正确。尚未测量困惑度, 这一项仍待完成。

## 对比中发现的其他问题

- README/快速上手建议使用 GGUF, 但 `tools/bitnet2aimw.py` 实际读取的是 `microsoft/bitnet-b1.58-2B-4T` 的打包 `model.safetensors`。
- 与本 PR 无关, 但若与 llama.cpp 对比会用到: 原版 llama.cpp (b12110) 在 BitNet 2B-4T 上输出乱码, 因为其计算图用 SiLU 而模型需要平方 ReLU (`src/models/bitnet.cpp`, `LLM_FFN_SILU` -> `LLM_FFN_RELU_SQR`)。部分社区 TQ2_0 GGUF 即使打了该修复仍然乱码; 可用的是 `Synapticode` 的版本。

## 未完成

VNNI 路径的 AVX2 回退 (目前需要 AVX-512 VNNI 硬件)、推测解码下的 int8/4-bit 输出层、困惑度检查、ARM64/NEON。
