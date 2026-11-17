> [中文版](bus.zh.md)

# Plugin Bus Manual

A plugin system of ~100 lines of C plus pure-stdlib Python. Design principle: **the model does not perceive plugins; the router decides what gets called**.

## Protocol

```
Caller (C engine / UI / your script)
  │  aim_bus_call("cap.name", '{"query":"...","topk":5}', ...)
  ▼
bus.c: read the caps.json registry → write the query to a temp JSON file → spawn the subprocess from the template
  ▼
Python plugin: argv[1] = query file path → process → print one JSON object to stdout
  ▼
Caller receives the JSON from stdout ({"cap":"cap.name", ...})
```

## Registry `plugins/caps.json`

```json
{
  "kb.search":  "python H:/path/plugins/kb_search.py %s",
  "web.search": "python H:/path/plugins/web_search.py %s",
  "kb.add":     "python H:/path/plugins/kb_add.py %s",
  "web.ingest": "python H:/path/plugins/web_ingest.py %s",
  "demo.time":  "python H:/path/plugins/demo_time.py %s"
}
```

`%s` = path to the query JSON file. The C side does a micro-scan of caps.json (`"cap" : "cmd"` key-value pairs) — no full JSON parser needed.

## Plugin contract (three rules)

1. `argv[1]` receives the query file path, content like `{"query":"...","topk":5}`
2. stdout must contain **exactly one JSON object** (extra prints pollute parsing; debug info goes to stderr or a log file)
3. The output object carries a `"cap": "<its own name>"` field

The minimal example is `plugins/demo_time.py` (20 lines).

## Adding a new plugin (two steps, zero C changes)

```python
# plugins/my_cap.py
import io, json, sys
req = json.load(io.open(sys.argv[1], encoding="utf-8"))
json.dump({"cap": "my.cap", "echo": req.get("query", "")}, sys.stdout, ensure_ascii=False)
```

```json
// add one line to caps.json
"my.cap": "python H:/path/plugins/my_cap.py %s"
```

Verify with `aimp_probe x --cap my.cap hello`, or pick `my.cap` in the UI chat dropdown and click 🔌 direct call.

## The five built-in plugins

| Plugin | Input | Output essentials |
|---|---|---|
| `kb.search` | query, topk | `results[]`: merged retrieval over base+dyn stores, sorted by score |
| `kb.add` | file path | `added/rejected/fts_indexed`: parse → chunk → dedup → incremental FTS index |
| `web.search` | query, topk | `results[]`: multi-engine parallel SERP (with snippets) |
| `web.ingest` | query, topn | `crawled/rejected/details[]`: parallel fetch → render cascade → quality gate → ingest, see creeper.md |
| `demo.time` | anything | echo + timestamp, smoke test for the bus protocol |

## Who calls what in the business chain

Chat auto mode (the router in `ui/server.py` — deterministic regex, not the model):

```
small-talk whitelist hit ──────────────► model direct answer
knowledge question ──► [kb.search] hit ──► --ask retrieval-grounded answer
not found ──► [web.search] ──► [web.ingest] ──► [kb.search] ──► --ask / summary fallback / honest refusal
```

Every step is visible in the UI's run-log panel.
