# Search Engine / Crawler Manual

Self-built crawler line: multi-engine search entry → parallel fetch → Edge render cascade → five-signal quality gate → ingest. Pure stdlib throughout (the render cascade uses the system Edge — no browser downloads).

## File map

| File | Responsibility |
|---|---|
| `serp.py` | Four-engine SERP adapters (bing/Baidu/360/Sogou) + multi-engine parallel + block-level snippet extraction |
| `engine.py` | FetchKernel (robots check + per-domain rate limit + size cap) + fetch_generic + MediaWiki official API adapter |
| `render.py` | Edge render cascade (Playwright `channel=msedge`, zero browser downloads) |
| `quality.py` | Five-signal quality gate, fail-closed |
| `crawl_urls.py` | Batch crawl orchestration |
| `github_adapter.py` | GitHub official API adapter (never crawls GitHub) |

## Search layer (serp.py)

- Three engines fire **in parallel via thread pool** (no queueing), interleaved merge and dedup
- Block-level result extraction: bing's `b_algo`+`b_caption` blocks; Baidu/360/Sogou split by `<h3>`, snippets with HTML entity decoding
- Shell pages automatically escalate to the render cascade (raw fetch <8KB or missing result-block markers → re-fetch via Edge)
- Anti-bot shell pages (Sogou IP-level risk control) honestly return empty — no brute-forcing

robots.txt findings (written into code comments): the search entry paths of **all four** engines are robots-disallowed (`/search`, `/s?`, `/s`, `/web?`) — industry standard. This stack treats SERPs as query interfaces (small volume, low frequency) and **fetches page bodies only from third-party pages, checking robots per site**; a robots refusal means no cascade, not a workaround.

## Fetch layer (web_ingest plugin orchestration)

```
SERP results → domain-diversified topN (quality sources whitelisted first: MDN/wiki/official docs)
  → parallel raw fetch (4 worker threads)
  → serial render cascade (thin-shell pages / timeout sites, robots checked first)
  → five-signal quality gate (fail-closed)
  → kb.add(origin=crawl) → incremental FTS index
```

## Five-signal quality gate (quality.py)

| Signal | Blocks |
|---|---|
| residual_too_short | residual text <80 chars after shell stripping |
| shell_ratio | login/follow/nav shell-word ratio too high |
| too_few_sentences | no sentence structure (non-prose page) |
| low_density | low information density (max of CJK-diversity / latin-word-diversity channels) |
| menu_smell | menu-list smell (many ultra-short lines) |

All fail-closed: any failed signal rejects with a reason. The density signal is dual-channel max — technical docs mixing code/terms dilute the CJK ratio; a pure-CJK metric would false-kill them, already fixed.

## Render cascade (render.py)

- `render_html` (keeps HTML, for SERP) / `render_fetch` (returns body text)
- Playwright `channel=msedge`: uses the system Edge, **zero browser downloads**
- Only cascades for "thin-shell pages / raw-fetch timeout" sites, robots checked first
- Known un-crawlable: JS environment-detection sites (e.g., support.mozilla.org's integrity check) — honestly skipped

## Adding a search engine

1. Write `search_xxx(query, n)` in `serp.py`, returning `[{"title","url","snippet"}]` (filter the engine's own domain pages — promo pages do leak through in practice)
2. Register it in the `ENGINES` dict
3. Re-verify the robots stance and write the conclusion into comments (current status of the four: all disallow search pages, see above)
4. Probe first (`probe_engines.py`-style), then wire in

## Adding quality-source whitelist entries

The `PRIORITY` tuple in `web_ingest.py` lists quality domains in order (sites with proven body quality); matches jump to the front of the candidate queue, preserving SERP order within the same tier.
