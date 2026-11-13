# C 引擎说明书

纯 C、零第三方运行时的本地推理引擎。所有数字都来自 2018 年双核 8GB 笔记本实测（见 README 硬件宣言）。

## 文件地图

| 文件 | 职责 |
|---|---|
| `main.c` | CLI 入口：`--cap`（插件直调）/ `--ask`（检索引答）/ `--gt`（给定文本生成）/ `--enc` `--dec`（编解码） |
| `forward.c` | 三值推理核：2-bit 打包、SIMD 快路径、线程池、自校准护栏 |
| `tokenizer.c/.h` | 自研分词器：字节级 + BPE |
| `aimw.c/.h` | AIM-W 权重打包格式（三值 2-bit）读写 |
| `gguf.c/.h` | GGUF 读取（仅作权重来源格式） |
| `bus.c/.h` | 插件总线调度器（见 bus.md） |
| `build.cmd` / `build.ps1` | Windows 构建脚本 |

## 构建

```cmd
cd engine
build.cmd          # 默认 x64
```

验收标准是「双架构编译」：x64 与 ARM64 都应能编译通过（真机验证见研究日志）。

## 三值核

权重为三值 {-1, 0, +1}，打包进 2-bit（AIM-W 格式，`aimw.c`）。推理是查表 + 整数累加，无浮点权重乘法。

性能账（同一台双核机）：

| 阶段 | 速度 |
|---|---|
| 基线串行 | 2.08 tok/s |
| SIMD v1（查表符号化） | **85.53 tok/s**（41×） |
| BitNet b1.58 2B 全模型 | 串行 1.99 → 双工人 **3.20 tok/s** |

## 线程池与自校准护栏

`forward.c` 内置 Windows 线程池（最多 15 工人，动态块窃取）：

- **默认串行**。`AIM_WORKERS=n` 环境变量显式开启
- 启动时自动做一次**自校准**：同一负载并行 vs 串行实测，并行不比串行快 25% 以上就自动降级回串行——防止弱核/带宽受限机器上多线程负收益
- 另有 `AIM_PAR_DBG=1` 输出校准详情
- 启动时主动关闭进程级 EcoQoS 电源节流（沙箱环境曾实测 worker 被冻结 5 秒不唤醒）

实测参考：2 核机上 W2 是甜点（+60%），W4 反而降速。4 核以上机器请自行 bench（`tools/` 下有基准脚本）。

## CLI 直调

```bash
# 插件直调 (不加载模型, 秒回)
aimp_probe x --cap kb.search "算力" 4

# 检索引答 (内部先调 kb.search 插件拿 top3 再拼 prompt)
aimp_probe <weights.aimw> <tokenizer.gguf> --ask "什么是注意力机制" 32

# 给定文本生成
aimp_probe <weights.aimw> <tokenizer.gguf> --gt "以下是资料:...\n问题:..." 32
```

`--ask` 的关键点：**模型不知道插件存在**。检索、拼 prompt 全是确定性代码，模型只做文本进文本出。

## 已知边界

- 940MX 级别的独显无加速价值，引擎纯 CPU，不写 CUDA
- `forward.c` 的 SIMD 路径按 x86 SSE/AVX 写，ARM64 走通用路径（双架构编译保证不炸，速度另测）
- 长上下文/大 batch 未经系统调优——这套核为「单机单请求」设计
