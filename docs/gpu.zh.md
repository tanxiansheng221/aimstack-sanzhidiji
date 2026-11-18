> [English](gpu.md)

# GPU 推理后端说明书（可选 PyTorch 路径）

产品链（UI → C 引擎 → 插件总线 → 知识库 → 爬虫）是纯 C + stdlib，**零第三方运行时**。本后端是可选旁路：同一个 AIM-W 文件，由自写的 PyTorch 加载器在 GPU 上跑。PyTorch/ROCm/CUDA 是全仓唯一第三方运行时，且只活在这里。

## 它是什么

- `tools/gpu_backend.py`：AIM-W 解析器（与 `aimw.c` 同款格式纪律——未知 version/dtype/码 3 一律拒绝，禁止猜测）、反量化器、以及镜像 `forward.c` 语义的前向（RMSNorm、half-split RoPE、SubLN×2、GQA、tied lm_head）
- 只做贪心解码，不采样——对齐是契约

## 用法

```bash
# GPU（auto 自动探测 cuda/mps，也可显式 cpu）
python tools/gpu_backend.py --aimw data/m2b_tern.aimw --prompt-ids 128000,791,7438,315,2324,374 --steps 256 --device cuda --compute fp32

# bf16 快速路径（2.4 倍，但没拿到对齐豁免——见下）
python tools/gpu_backend.py --aimw data/m2b_tern.aimw --prompt-ids ... --steps 256 --device cuda --compute bf16

# 从文本出发（需要官方 tokenizer.json 放 data/bitnet2b/）
python tools/gpu_backend.py --aimw ... --prompt "The capital of France is" --steps 64 --tok data/bitnet2b/tokenizer.json
```

| 参数 | 取值 | 含义 |
|---|---|---|
| `--device` | auto / cuda / mps / cpu | `auto` 探测到 cuda 就用 GPU |
| `--compute` | fp32 / bf16 | fp32 = 对齐路径；bf16 = 快速路径 |
| `--prompt-ids` / `--prompt` | 逗号 ids / 文本 | 二选一 |
| `--steps` | N | 贪心生成 token 数 |

## 对齐纪律（整个存在的意义）

C 引擎是真值。后端只有贪心逐 token 与 C 引擎一致，才有资格写对齐声明。AMD Radeon Pro W7900（gfx1100，48GB，ROCm 7.2.1）实测，权重 MD5 与 C 引擎验收过的文件逐字节一致：

| 路径 | 对 C 引擎 parity | 速度 |
|---|---|---|
| **fp32 计算** | **256/256 token 逐位一致**（三题） | 13.7 tok/s（≈ 本机 C 引擎 8×） |
| bf16 计算 | 随 prompt 而变：有的题全对，近并列 logit 处 argmax 翻转（观测 t30 / t57） | 32.9 tok/s（2.4×） |

三引擎 × 三问题 × 三测试矩阵（C-Linux gcc13 / torch fp32 / torch bf16）在 `lab/results/engine_matrix_gpu_results.json`：17/18 EXACT，唯一 miss 就是上面有据可查的 bf16 近并列翻转。**只有 fp32 路径拿了对齐豁免**；bf16 数字如实报告，不藏。

## 选哪个引擎

| 场景 | 用 |
|---|---|
| 纯 CPU 机器，守住诚实下限 | C 引擎（`aimp_probe`）——永远 CPU |
| GPU 机器，要速度且要精确 | `gpu_backend.py --device cuda --compute fp32` |
| GPU 机器，快速草稿不要求精确 | `--compute bf16` |

UI 服务目前只驱动 C 引擎（`AIM_ENGINE_EXE`）；GPU 后端的 CLI 协议（GREEDY id 数组）与服务端的 chat 协议（GEN_TEXT）是两套。
