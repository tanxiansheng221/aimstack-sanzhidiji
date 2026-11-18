# Linux multithreading + AVX-512 VNNI path (opt-in) and benchmark

This adds a POSIX worker pool and a faster ternary/attention path to `engine/forward.c`, all **opt-in via environment variables**. With none of them set the original code path runs unchanged. It closes the "Multi-worker on POSIX would be a pthread port (not yet done)" note in `engine.md`.

Tested on: Ryzen 9 7945HX (16 cores / 32 threads, AVX-512 + VNNI), Linux, gcc 16, `microsoft/bitnet-b1.58-2B-4T` converted with `tools/bitnet2aimw.py`.

## Build

```bash
gcc -O3 -march=native -pthread -o aimp engine/aimw.c engine/forward.c engine/gguf.c engine/main.c engine/tokenizer.c engine/bus.c -lm
```

Note: the documented `gcc -O2` line (no `-march`) compiles the `__AVX2__` path out and decodes at ~1.6 tok/s on this CPU; `-O3 -march=native` gives ~6.9 tok/s on the unmodified engine.

## Knobs (all default off)

| env | effect |
|---|---|
| `AIM_WORKERS=N` | POSIX thread pool (pthreads + C11 atomics + futex), N extra workers; core-aware pinning (one per physical core first). `AIM_MIN_ROWS` (512 is good for 16 threads), `AIM_SPIN`, `AIM_PIN` |
| `AIM_VNNI=1` | int8-activation ternary kernel (per-vector absmax, BitNet W1.58A8) using AVX-512 VNNI `dpbusd`; integer accumulation so threaded == serial exactly |
| `AIM_PREFILL_BATCH=k` | batched kernel (up to 8 columns per tile, weights unpacked once) and batched prefill; skips the output head except for the last token |
| `AIM_Q8V=1` | int8 VNNI kernel for the Q8 output layer |
| `AIM_SPEC=k` | exact speculative decoding (longest-suffix n-gram drafter, draft length 1+3*matchlen); output identical to plain greedy. The previous drafter was off by one position |
| `AIM_LM4=1` | **lossy**, 4-bit copy of the output layer for single-token logits. Text changes; quality not measured; incompatible with `AIM_SPEC` |

Also: parallel AVX-512 attention (HD=128) over (token, head), and padded activation buffers (power-of-two strides aliased in L1 and cost ~40% of matvec time).

Example (16 physical cores):

```bash
AIM_WORKERS=15 AIM_MIN_ROWS=512 AIM_VNNI=1 AIM_Q8V=1 AIM_PREFILL_BATCH=32 AIM_SPEC=8 \
  ./aimp data/m2b_tern.aimw data/m2b_llama3_head.gguf --gt "prompt" 64
```

## Results (BitNet b1.58 2B-4T, CPU only, tok/s)

| | prefill | decode |
|---|---|---|
| as shipped, `gcc -O2` | n/a | 1.6 |
| as shipped, `-O3 -march=native`, 1 thread | 7.1 | 6.9 |
| this patch, 1 thread | 65 | 24 (34 with `AIM_Q8V=1`) |
| this patch, 4 threads | 239 | 53 |
| this patch, 8 threads | 420 | 53 |
| this patch, 16 threads (pinned) | 623-660 | 58-60 |
| llama.cpp b12110 (CPU, TQ2_0), 1 thread | 47 | 23.4 |
| llama.cpp, 16 threads | 490 | 42.6 |

Speculative decoding on top (16 threads, exact output): copy-heavy text 57 -> 230-360 tok/s, JSON 58 -> 81, summaries 51 -> 85, free-form story no gain (58 -> 58, ~2% overhead).

Notes:
- Decode is memory-bandwidth bound (~54 GB/s); the Q8 output layer is ~39% of the bytes per token.
- 16 threads (= physical cores) is the peak for both this patch and llama.cpp; SMT siblings add nothing and 32 threads is slightly worse.
- The engine's `TIME` line uses `clock()` (CPU time summed over threads), so use wall time when comparing threaded runs.

## Correctness

6 of 7 test prompts are token-identical to the original float engine over 48 tokens (including a 308-token prompt). One prompt (a haiku) diverged late; the int8 activations make near-ties flip. All outputs were fluent and correct. No perplexity measurement was done, so that is still open.

## Other findings from the comparison

- `README`/quickstart suggest a GGUF, but `tools/bitnet2aimw.py` reads the packed `model.safetensors` from `microsoft/bitnet-b1.58-2B-4T`.
- Not part of this PR, but useful if you compare against llama.cpp: stock llama.cpp (b12110) produces garbage on BitNet 2B-4T because its graph uses SiLU where the model needs squared ReLU (`src/models/bitnet.cpp`, `LLM_FFN_SILU` -> `LLM_FFN_RELU_SQR`). Some community TQ2_0 GGUFs stay garbage even with that fix; one that works is `Synapticode`'s.

## Not done

AVX2 fallback for the VNNI path (needs AVX-512 VNNI hardware today), int8/4-bit output layer under speculative decoding, perplexity check, ARM64/NEON.
