# External Knowledge Base Manual

Two-layer store design: a **fixed base** (finished wiki store, 776k cards) + a **dynamic store** (user uploads + crawler increments, swappable as a whole at any time). Retrieval queries both stores in parallel and merges results sorted by score.

## Where the knowledge base comes from (self-built)

This repository **does not distribute the store itself** — the pipeline is fully open-source, with two paths:

**① Build the base from a Wikipedia dump** (recommended starting point, ~776k cards):

```bash
# Download the Chinese Wikipedia dump (Wikimedia Downloads or a mirror)
python lab/kb/build_wiki_base.py <zhwiki-dump.xml.bz2>   # parse -> chunk -> dedup -> ingest -> FTS
```

**② Increments from the crawler**: the `web.ingest` plugin (see creeper.md) searches by topic → crawls → gates → ingests, stacking increments on top of the base — this is the designed usage of this stack: the knowledge base is alive, not a dead snapshot.

Either way, the output is two files: `kb_store/cards.jsonl` + `kb_store/kb_fts.db`. The directory can be moved as a whole.

## Build / increment

| File | Purpose |
|---|---|
| `lab/kb/build_wiki_base.py` | Build the base library from a wiki dump (heavy job; using the finished store is fine) |
| `lab/kb/kb_core.py` | Core store engine: card append (O(1)), SHA-256 content-addressed dedup gate |
| `lab/kb/fts_index.py` | FTS5 index wrapper (incremental — only new cards are indexed, no full rescan) |
| `lab/kb/ingest.py` | Rich-format parsing: txt/md/html/docx/pdf → plain text |
| `lab/kb/kb_add.py` (plugin) | Ingest pipeline: load_any → chunk → dedup (fail-closed) → cards.jsonl → FTS |
| `lab/kb/cli.py` | CLI direct call |

House rules: **the dedup gate blocks duplicates; the quality gate blocks low quality** (the crawler's five quality signals are in creeper.md). Placeholder files (<30 chars after comment stripping) are skipped outright.

## Retrieval

```bash
# CLI direct call
aimp_probe x --cap kb.search "linear attention" 5

# Or via the UI "Knowledge Base Retrieval" panel
```

`kb_search.py` (plugin) implements merged retrieval over base+dyn stores, dedup and sort by `score` descending. Each card returns `title/text/source/score/store(base|dyn)`.

## Swapping the dynamic store

The `kb_dyn_store/` directory is self-contained (cards + index); swapping stores = swapping directories + changing the `AIM_KB_DYN_ROOT` environment variable. That is the core of the "externalized knowledge base" design: the base stays, the dynamic layer restarts anytime.

## Notes

- cards.jsonl is append-only NDJSON — **do not hand-edit** (dedup relies on SHA-256 content addressing; manual edits cause index-card drift)
- The FTS index can be rebuilt anytime (`purge_base_fts.py` / rebuild tools); the cards themselves are the source of truth
- The card chunking strategy is tuned for prose pages: short-line shell text is cut by the quality gate before ingest
- 776k cards retrieve in seconds on the dual-core machine; partition larger stores yourself
