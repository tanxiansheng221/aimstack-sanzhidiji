# GPU Backend Manual (optional PyTorch path)

The product chain (UI → C engine → plugin bus → KB → crawler) is pure C + stdlib with **zero third-party runtime**. This backend is an optional side door: the same AIM-W file, loaded and run by a self-written PyTorch loader, for machines that have a GPU. PyTorch/ROCm/CUDA is the only third-party runtime in the whole repo, and it lives only here.

## What it is

- `tools/gpu_backend.py`: AIM-W parser (same format discipline as `aimw.c` — unknown version/dtype/code-3 is rejected, never guessed), dequantizer, and a forward pass mirroring `forward.c` semantics (RMSNorm, half-split RoPE, SubLN×2, GQA, tied lm_head)
- Greedy decode only, no sampling — parity is the contract

## Usage

```bash
# GPU (auto-detects cuda/mps; explicit cpu also works)
python tools/gpu_backend.py --aimw data/m2b_tern.aimw --prompt-ids 128000,791,7438,315,2324,374 --steps 256 --device cuda --compute fp32

# bf16 fast path (2.4x, NOT parity-blessed — see below)
python tools/gpu_backend.py --aimw data/m2b_tern.aimw --prompt-ids ... --steps 256 --device cuda --compute bf16

# from text (needs the official tokenizer.json under data/bitnet2b/)
python tools/gpu_backend.py --aimw ... --prompt "The capital of France is" --steps 64 --tok data/bitnet2b/tokenizer.json
```

| Flag | Values | Meaning |
|---|---|---|
| `--device` | auto / cuda / mps / cpu | `auto` picks cuda when available |
| `--compute` | fp32 / bf16 | fp32 = parity path; bf16 = fast path |
| `--prompt-ids` / `--prompt` | csv ids / text | one of the two |
| `--steps` | N | greedy tokens to generate |

## The parity discipline (the whole point)

The C engine is the ground truth. A backend earns a parity claim only by matching it greedy-token-for-greedy-token. Measured on AMD Radeon Pro W7900 (gfx1100, 48GB, ROCm 7.2.1), weights MD5-identical to the C-validated file:

| Path | Parity vs C engine | Speed |
|---|---|---|
| **fp32 compute** | **256/256 tokens bit-identical** (3 prompts) | 13.7 tok/s (~8× laptop C engine) |
| bf16 compute | prompt-dependent: exact on some prompts, argmax flips on near-tie logits elsewhere (token 30 / token 57 observed) | 32.9 tok/s (2.4×) |

A 3-engine × 3-question × 3-test matrix (C-Linux gcc13 / torch fp32 / torch bf16) lives in `lab/results/engine_matrix_gpu_results.json`: 17/18 EXACT; the single miss is the documented bf16 near-tie divergence. **Only the fp32 path is parity-blessed**; bf16 numbers are reported, not hidden.

## Which engine should you use

| Situation | Use |
|---|---|
| CPU-only box, the honest floor | C engine (`aimp_probe`) — always CPU |
| GPU box, need speed with exactness | `gpu_backend.py --device cuda --compute fp32` |
| GPU box, fast draft where exactness is not required | `--compute bf16` |

The UI server currently drives the C engine only (`AIM_ENGINE_EXE`); the GPU backend's CLI protocol (GREEDY id arrays) is separate from the server's chat protocol (GEN_TEXT).
