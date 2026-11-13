# -*- coding: utf-8 -*-
"""Full loop smoke: query -> multi-engine search -> crawl top-N diverse -> KB gate -> search back."""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creeper.serp import multi_search
from creeper.engine import FetchKernel, fetch_generic, ingest_doc
from creeper.render import render_fetch
from creeper.quality import quality
from kb.kb_core import KB

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QUERY = sys.argv[1] if len(sys.argv) > 1 else "线性注意力 大语言模型"
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 3

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
kb = KB(root, build_index=False)  # huge-store mode: gate only
from kb.fts_index import FtsIndex
fts = FtsIndex(root)
kernel = FetchKernel(min_interval=1.5)

print("Q:", QUERY)
hits = multi_search(QUERY, engines=("bing", "sogou"), per_engine=6)
print("search hits:", len(hits))

# prefer diverse domains
picked, domains = [], set()
for h in hits:
    dom = h["url"].split("/")[2]
    if dom not in domains:
        domains.add(dom); picked.append(h)
    if len(picked) >= TOPN:
        break

for h in picked:
    doc, st = fetch_generic(kernel, h["url"], title=h["title"])
    if not doc and st == "too_little_text":
        doc, st = render_fetch(h["url"])  # JS shell -> real browser (robots already checked)
    if not doc:
        print("  SKIP %-50s (%s)" % (h["url"][:50], st))
        continue
    ok, sc, rs = quality(doc["text"])  # gate 2: quality before ingest
    if not ok:
        print("  LOWQUAL %-46s %s" % (h["url"][:46], rs))
        continue
    added, total = ingest_doc(kb, doc, source=h["url"])
    print("  CRAWLED %-50s +%d cards (gate %d | q%.2f)" % (h["url"][:50], added, total - added, sc))

print("kb:", kb.stats())
a, n = fts.add_jsonl(os.path.join(root, "cards.jsonl"))  # incremental: new cards only
print("fts +%d new cards indexed" % a)
print("--- verify: FTS5 search KB for the query ---")
for r in fts.search(QUERY, topk=4):
    print("  %.1f [%s] %s | %s" % (r["score"], r["origin"], r["title"][:40], r["text"][:60]))
