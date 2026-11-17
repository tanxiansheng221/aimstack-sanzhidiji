> [中文版](engine.zh.md)

# C Engine Manual

A local inference engine in pure C with zero third-party runtime. All numbers below come from real measurements on a 2018 dual-core 8GB laptop (see the Hardware Statement in the README).

## File map

| File | Responsibility |
|---|---|
| `main.c` | CLI entry: `--cap` (direct plugin call) / `--ask` (retrieval-grounded answer) / `--gt` (generation from given text) / `--enc` `--dec` (encode/decode) |
| `forward.c` | Ternary inference core: 2-bit packing, SIMD fast path, thread pool, self-calibration guard |
| `tokenizer.c/.h` | Self-built tokenizer: byte-level + BPE |
| `aimw.c/.h` | AIM-W weight packing format (ternary 2-bit) read/write |
| `gguf.c/.h` | GGUF reading (weight source format only) |
| `bus.c/.h` | Plugin bus scheduler (see bus.md) |
| `build.cmd` / `build.ps1` | Windows build scripts |

## Build

```cmd
cd engine
build.cmd          # x64 by default
```

The acceptance criterion is "dual-architecture compilation": both x64 and ARM64 should compile (real-device verification in the research logs).

## Ternary core

Weights are ternary {-1, 0, +1}, packed into 2 bits (AIM-W format, `aimw.c`). Inference is table lookup + integer accumulation — no floating-point weight multiplication.

Performance ledger (same dual-core machine):

| Stage | Speed |
|---|---|
| Baseline serial | 2.08 tok/s |
| SIMD v1 (symbol-table lookup) | **85.53 tok/s** (41×) |
| BitNet b1.58 2B full model | serial 1.99 → dual-worker **3.20 tok/s** |

## Thread pool and self-calibration guard

`forward.c` ships a Windows thread pool (up to 15 workers, dynamic work stealing):

- **Serial by default.** Set the `AIM_WORKERS=n` environment variable to enable it explicitly
- A **self-calibration** runs at startup: the same load is measured in parallel vs serial; if parallel is not more than 25% faster, it automatically falls back to serial — protecting weak-core / bandwidth-limited machines from negative multi-threading returns
- `AIM_PAR_DBG=1` prints calibration details
- Process-level EcoQoS power throttling is actively disabled at startup (in sandboxed environments, workers were observed frozen for 5 seconds without waking)

Measured reference: on 2 cores, W2 is the sweet spot (+60%); W4 actually slows down. For 4+ core machines, bench it yourself (benchmark scripts under `tools/`).

## Direct CLI calls

```bash
# Direct plugin call (no model loaded, instant)
aimp_probe x --cap kb.search "compute" 4

# Retrieval-grounded answer (internally calls the kb.search plugin for top-3, then assembles the prompt)
aimp_probe <weights.aimw> <tokenizer.gguf> --ask "what is attention" 32

# Generation from given text
aimp_probe <weights.aimw> <tokenizer.gguf> --gt "Reference:...\nQuestion:..." 32
```

The key to `--ask`: **the model does not know plugins exist**. Retrieval and prompt assembly are deterministic code; the model only does text-in, text-out.

## Known limits

- A 940MX-class discrete GPU offers no acceleration value — the engine is pure CPU, no CUDA
- The SIMD path in `forward.c` targets x86 SSE/AVX; ARM64 uses the generic path (dual-architecture compilation guarantees it runs; speed measured separately)
- Long context / large batch are not tuned — this core is designed for "one machine, one request"
