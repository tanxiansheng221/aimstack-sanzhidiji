# Quick Start

Goal: from clone to running server within 10 minutes. Building the knowledge base and converting weights are two independent one-time preparations (about 1 hour each, depending on bandwidth and machine).
Impatient? Feed a few local md/html documents through the UI's "Document Ingest" into the dynamic store — chat and full web-search chain work without the base library.

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10+ | UI / Edge render cascade first; Linux can compile the engine (render cascade needs Chromium installed) |
| Python 3.8+ | All plugins/services are pure stdlib — **no pip install needed** |
| C compiler | For building the engine (MSVC or MinGW, see engine.md) |
| ~4GB disk | Wiki dump + 1.5GB base library + 0.9GB Microsoft weights + code |

## Four steps to a running stack

```bash
# 1. Build the engine
cd engine && build.cmd && cd ..

# 2. Build the knowledge base (Chinese wiki pre-chunks -> ~776k-card FTS library)
#    Put wiki_zh_*.jsonl.gz (each line {"title":..., "text":...}) in the chunk directory;
#    defaults to data/wiki/ if no argument; or skip this step and ingest a few
#    local md/html files via the UI's "Document Ingest"
python lab/kb/build_wiki_base.py <chunk directory>
# See docs/kb.md

# 3. Microsoft BitNet weights: download the gguf yourself (MIT) -> convert to AIM-W + tokenizer head
#    hf.co/microsoft/bitnet-b1.58-2B-4T, hf-mirror.com works in China
python tools/bitnet2aimw.py <weights.gguf> data/m2b_tern.aimw
# tokenizer.json is in that HF repo; place it under data/bitnet2b/ then:
python tools/make_bnet_head.py          # -> data/m2b_llama3_head.gguf
# See docs/weights.md

# 4. Start the server
python ui/server.py                 # http://127.0.0.1:8600
```

Open `http://127.0.0.1:8600` in your browser.

## Acceptance checklist (30 seconds each)

| # | Action | Expected |
|---|---|---|
| 1 | Type `在吗` (hello) in the chat box | Direct chat answer (no KB, no web) |
| 2 | Type `什么是注意力机制` (what is attention) | Path = KB-grounded answer with a source card (base library hit) |
| 3 | Type `帮我搜索一下 firefox` (search firefox for me) | Path = web-grounded answer: search → crawl → gate → ingest → answer (1-3 minutes) |
| 4 | Pick `demo.time` from the plugin dropdown, type anything, click 🔌 direct call | Bubble echoes JSON (`cap: demo.time`) |
| 5 | Sidebar "Store statistics" | base ≈ 776k cards / dyn store counts |

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `AIM_WORKERS` | 2 (built into server.py) | Inference engine thread pool workers; the self-calibration guard falls back to serial when parallel isn't faster |
| `AIM_KB_BASE` | `<repo>/kb_store` | Fixed base library (finished wiki store) path |
| `AIM_KB_DYN_ROOT` | `<repo>/kb_dyn_store` | Dynamic store (user uploads + crawler increments) path |
| `AIM_MODEL_W` | `<repo>/data/m2b_tern.aimw` | AIM-W ternary weights (step-3 conversion output) |
| `AIM_MODEL_TOK` | `<repo>/data/m2b_llama3_head.gguf` | Tokenizer head gguf |
| `AIM_ENGINE_EXE` | `<repo>/engine/build/aimp_probe.exe` | Engine executable |

> Note: some dbg/test scripts under `tools/` are dev-time debug tools and a few contain dev-machine absolute paths (for reading reference only, not part of the quick-start chain). The four-step quick-start chain uses repo-relative paths throughout — works right after clone.

## FAQ

- **Port 8600 occupied**: change `PORT` at the top of `ui/server.py`
- **Chat is slow**: local CPU inference + the full web chain taking 1-3 minutes is normal; small talk should be near-instant (direct-answer path)
- **Crawled pages never enter the store**: the fail-closed quality gate is a feature, not a bug — check per-site rejection reasons in the "Web Ingest" panel (lowqual:xxx / robots_disallow / fetch_error)
- **Low generation speed**: confirm `AIM_WORKERS=2`; EcoQoS throttling in sandboxes/cloud VMs can cost an order of magnitude, real machines are unaffected
