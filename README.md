# aimstack

> Fully self-built local AI stack — from scratch: ternary C inference engine (zero third-party runtime), plugin bus, multi-engine web crawler with quality gates, FTS knowledge base, and the complete training pipeline + research logs.

No frameworks, no PyTorch dependency (used only for reconciliation during development, zero external runtime in the product), no off-the-shelf engine — the C inference core, plugin bus, crawler, knowledge base, and trainer are all hand-written. The dev machine is a dual-core laptop bought in 2018, never replaced.

🎬 **Full-chain UI demo (6:59, online)** — direct chat answers / KB-grounded answers / web search-crawl-ingest-answer / direct plugin calls, all screen-recorded on the 2018 dual-core laptop:

https://github.com/user-attachments/assets/d4f897d0-001c-4dc6-b773-324c69b16a02

```
you speak
  │
  ▼
UI (server.py, pure stdlib)        ← local 127.0.0.1:8600
  │
  ▼
C engine (engine/, pure C, zero third-party runtime)
  ├─ ternary inference core (BitNet b1.58-style 2-bit packing + SIMD fast path)
  ├─ tokenizer (self-built: byte-level + BPE; third-party weights ship with the official Microsoft tokenizer, see below)
  └─ plugin bus (~100 lines of C: caps.json registry → spawns Python plugins → JSON envelopes)
        │
        ▼
Python plugins (pure stdlib)
  ├─ kb.search / kb.add     ← external knowledge base (FTS retrieval, dedup gate)
  ├─ web.search / web.ingest← multi-engine search (bing/Baidu/360 in parallel + Edge render cascade + quality gates)
  └─ demo.time              ← minimal smoke-test plugin
```

The model is unaware of the plugins: routing (deterministic code) decides what to call, plugins feed material to the model, and the model only reads and writes answers. **Model = reader, plugins = librarians.**

## Quick start

```bash
git clone https://github.com/tanxiansheng221/aimstack-sanzhidiji.git
cd aimstack-sanzhidiji

# 1. Build the engine (Windows, dual-architecture see docs/engine.md)
cd engine && build.cmd && cd ..

# 2. Build the knowledge base (Chinese wiki pre-chunks -> FTS store, see docs/kb.md)
python lab/kb/build_wiki_base.py <chunk directory>

# 3. Microsoft BitNet weights: download the gguf yourself (MIT;
#    hf-mirror.com works in China) -> convert to AIM-W + tokenizer head, see docs/weights.md
python tools/bitnet2aimw.py <weights.gguf> data/m2b_tern.aimw
python tools/make_bnet_head.py

# 4. Start the server
python ui/server.py        # http://127.0.0.1:8600
```

Details: [docs/quickstart.md](docs/quickstart.md)

> **Which tokenizer to use (important)**
> - For **third-party weights** (e.g., Microsoft BitNet b1.58-2B-4T), use the **official Microsoft tokenizer**: `tools/llama3_tok.py`. It is a pure-stdlib Llama-3 BPE implementation reading the official `tokenizer.json` (download from hf.co/microsoft/bitnet-b1.58-2B-4T, hf-mirror.com works in China, place at `data/bitnet2b/tokenizer.json`); encoding/decoding aligns with the official output.
> - The engine's built-in **self-built C tokenizer** (`engine/tokenizer.c`, byte-level + BPE) has only been reconciled token-by-token against a reference on a limited sample set (`tools/check_tok.py`) and is **not fully verified yet**. For third-party weights, the official tokenizer is authoritative; the self-built tokenizer serves the experiment loop for self-trained models (152M etc.).

## Directory

| Directory | Contents | Manual |
|---|---|---|
| `engine/` | C inference engine + ternary core + tokenizer + plugin bus | [docs/engine.md](docs/engine.md) |
| `plugins/` | 5 Python plugins + caps.json registry | [docs/bus.md](docs/bus.md) |
| `ui/` | Local web UI | [docs/quickstart.md](docs/quickstart.md) |
| `lab/creeper/` | Multi-engine search + render cascade + quality gates | [docs/creeper.md](docs/creeper.md) |
| `lab/kb/` | Knowledge base engine (FTS index / dedup gate / ingest) | [docs/kb.md](docs/kb.md) |
| `lab/train/` | Complete training pipeline (no-autograd stack + ten-subject curriculum generators + corpus builders) | [docs/training.md](docs/training.md) · [corpus](docs/corpus.md) |
| `lab/seeds/` | Experiment seeds (op18 dual-seed / cot_mech replication) | [docs/training.md](docs/training.md) |
| `lab/results/` | Experiment result JSONs (evidence) | — |
| `lab/docs/` | 117-chapter CPU-native research log + E6 negative-results dossier | Read this before touching the code |
| `tools/` | Probes / benchmark scripts | — |

## Hardware statement: a 2018 laptop holds the floor

The dev machine for the entire stack is a **Dell laptop bought in 2018**, never replaced (as of 2026):

| Part | Spec |
|---|---|
| CPU | Intel Core i7-7500U @ 2.70GHz (**2 cores / 4 threads**, low-power U) |
| RAM | 8GB DDR4-2133 single channel |
| GPU | GeForce 940MX (too weak, unused — the whole stack is pure CPU, no CUDA) |
| Storage | 512GB SSD + 1TB HDD |

Numbers this machine actually held:

- Ternary-core SIMD fast path: **85.53 tok/s** (optimized 41× from 2.08, C/NumPy bit-exact reconciliation kept green)
- BitNet b1.58 2B (AIM-W 0.87GB): **3.20 tok/s** dual-worker (serial 1.99, in-engine self-calibration guard)
- Local training sweet spot (this machine measured): **5M-30M params on 2 cores / 8GB**; a 152M full run would take ~35 days locally — the sweet spot scales with CPU and RAM (double the cores and memory and the model grows with it); training goes to free cloud containers, the stack itself sets no hardware floor

**The floor is held by this machine; the ceiling is yours to test** — the stack contains zero hardware-specific code. Multi-core, AVX-512, and large-memory machines are welcome to report real numbers.

## Training assets (fully open)

- No-autograd training stack (hand-written backprop, 29-parameter selftest reconciliation, rel 5.7e-7)
- Ten-subject curriculum generators + corpus builder suite (modular arithmetic / causal / constraint / graph / logic / math / planning / sequences / state machines / code)
- Full archive: corpus-autopsy discipline, replay ratios, three negative-return continuation recipes, four-arm arithmetic comparison with zero generalization
- See [docs/training.md](docs/training.md) and [lab/docs/](lab/docs/)

**Positive and negative results ship together** — the negative-results dossier (E6_CAMPAIGN_AND_RECIPE_DOSSIER) may be the most valuable part of this repo.

## The research log (117 chapters, written as it happened)

The entire stack has a complete first-person build journal: [lab/docs/CPU_NATIVE_DEDUCTION.md](lab/docs/CPU_NATIVE_DEDUCTION.md) — every design decision, every measured number, every failed idea, in real time. It is the primary source behind every claim in this README: the SIMD bring-up and the 0.9 → 3.2 tok/s campaign, the bit-exact reconciliation discipline, the speculative-decoding register-budget analysis, and the negative-results archive.

**English readers**: start with the [Research Log Overview & Chapter Map](lab/docs/RESEARCH_LOG_OVERVIEW.en.md) — headline results, a chapter map ("if you want to know X, read Y"), and why the original ships in Chinese. The log itself is CC-BY-NC 4.0.

## Weight attribution notice (must read)

- **BitNet b1.58 2B weights © Microsoft Corporation, MIT license**. The AIM-W format (`.aimw`) in this repo is only a **conversion layer / packing format** for those weights, not a model we trained. The converter is `tools/bitnet2aimw.py`; download the weights from Microsoft's official channel (or the hf-mirror.com mirror).
- Self-trained model weights (152M etc.) are **not released in this repo**; the repo releases the complete pipeline that trains them.
- The knowledge base is self-built from Chinese Wikipedia dumps, **CC-BY-SA**, crediting Wikipedia contributors. See [NOTICE.md](NOTICE.md).

## License matrix

| Asset | License |
|---|---|
| All code | **AGPL-3.0** — want to embed it in your online product? Either open-source your derivative or talk business licensing |
| `lab/docs/` research documents | CC-BY-NC 4.0 (attribution, non-commercial) |
| Microsoft weights (external, self-downloaded) | MIT, owned by Microsoft |
| Wiki content (self-built store, from dumps) | CC-BY-SA, owned by Wikipedia contributors |

## Discipline

- The crawler obeys robots.txt and refuses to bypass; search pages are query interfaces only, page bodies always come from third-party sites
- Quality gates are fail-closed: dedup + five signals (residual length / shell density / sentence structure / information density / menu smell); failing any gate means the content never enters the store
- Knowledge base engine, plugins, and crawler are all pure stdlib, zero third-party dependencies

## Known limits (honest list)

- The edge-side parameter ceiling is set by **memory and core count**, not by the ternary kernel: 2B is the comfortable ceiling on this 8GB dual-core test machine; machines with more RAM/cores go straight to bigger models (zero hardware-specific code in the stack). The pipeline's feeding is fine — how deep the model reads is the model's business
- support.mozilla.org and similar JS-environment-detection sites cannot be crawled (technical wall, not robots)
- Some gov.cn sites have broken SSL certificate chains — rejected per security discipline rather than skipping verification
- Windows-first (the render cascade uses Edge); Linux compiles, the render cascade needs Chromium

## Contact

- Business / licensing / feedback: **t8070824207@qq.com**
- Operating entity: Fengzhua Digital Technology Studio (锋爪数创数字技术工作室), Luohu District, Shenzhen, China
- Unified Social Credit Code: 92440300MAKDFBKUXT
