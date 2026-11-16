# Microsoft Weight Conversion Manual

**Attribution notice (read first)**: BitNet b1.58 2B weights © Microsoft Corporation (MIT). This repository only provides the **conversion layer** for the AIM-W packing format and the inference kernel — it does not train, modify, or distribute the weights themselves.

## Getting the weights

1. Download from the official channel: `huggingface.co/microsoft/bitnet-b1.58-2B-4T` (MIT, GGUF format)
2. Verify file integrity

## Converting to AIM-W

```bash
# Main converter (bitnet gguf -> .aimw ternary 2-bit packing)
python tools/bitnet2aimw.py <downloaded.gguf> data/m2b_tern.aimw

# Generic gguf conversion (fp weights, q4k, etc.)
python tools/gguf2aimw.py <in.gguf> <out.aimw>
python tools/gguf2aimw_v2.py <in.gguf> <out.aimw>     # v2 format

# Post-conversion validation
python tools/validate_aimw.py data/m2b_tern.aimw
python tools/peek_aimw_full.py data/m2b_tern.aimw     # structure preview
```

Conversion output is ~0.87GB (the payoff of 2-bit ternary packing: 1/4 to 1/2 of the original gguf size).

## Reconciliation tools (recommended after any weight swap)

| Tool | Purpose |
|---|---|
| `tools/bnet_oracle.py` | NumPy-side oracle implementation, reconciled bit-by-bit against the C engine |
| `tools/make_bnet_head.py` / `make_mini_bitnet.py` | Tiny test-weight construction (second-level smoke tests, no 2B weights needed) |
| `tools/check_tok.py` / `test_tokrt.py` | Tokenizer reconciliation |

Reconciliation discipline: **any kernel change must pass the C/NumPy bit-exact reconciliation before speed is even discussed** (the 41× optimization in the engine manual was done entirely under that contract).

## AIM-W format

- Definition and read/write implementation in `engine/aimw.c/.h`
- Ternary weights {-1,0,+1} packed into 2 bits, per-block scale
- Header carries architecture/layer/dimension metadata; `peek_aimw_full.py` can print it

## Self-trained model weights

Self-trained 152M / micro-core models are **not distributed in this repository**. `lab/train/` provides the complete training pipeline — you can train a small model from scratch with the ten-subject curriculum generators (`lab/train/curriculum_*_lab.py`: modular arithmetic / causal / constraint / graph / logic / math / planning / sequences / state machines / code) plus the corpus builders. That is exactly the "training assets fully open" part of this repo.

## FAQ

- **Shape mismatch during conversion**: confirm you downloaded the b1.58 2B gguf and it wasn't re-quantized by another tool
- **Garbled inference output**: run `tools/check_tok.py` first to rule out tokenizer reconciliation issues, then check the gguf→aimw conversion log
- **Wrong speed**: confirm `AIM_WORKERS=2` and check self-calibration output (`AIM_PAR_DBG=1`); EcoQoS in sandboxed environments costs an order of magnitude
