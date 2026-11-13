# -*- coding: utf-8 -*-
"""Juejin via SERP + render kernel (SPA pages need real browser)."""
import io, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creeper.serp import multi_search
from creeper.render import render_fetch
from creeper.quality import quality
from kb.kb_core import KB
from kb.fts_index import FtsIndex

sys.stdout.reconfigure(encoding="utf-8")

QUERY = sys.argv[1] if len(sys.argv) > 1 else "大模型 训练 原理"
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 4

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
kb = KB(root, build_index=False)
fts = FtsIndex(root)

hits = multi_search("site:juejin.cn " + QUERY, engines=("bing", "sogou"), per_engine=8)
urls = [h["url"] for h in hits if re.match(r"https://juejin\.cn/post/\d+", h["url"])]
# dedupe keep order
seen, urls = set(), [u for u in urls if not (u in seen or seen.add(u))]
print("juejin post urls found:", len(urls))

added_total = 0
for u in urls[:TOPN]:
    doc, st = render_fetch(u)
    if not doc:
        print("  SKIP %-45s (%s)" % (u[:45], st)); continue
    ok, sc, rs = quality(doc["text"])
    if not ok:
        print("  LOWQUAL %-38s %s" % (doc["title"][:38], rs)); continue
    res = kb.add(doc["title"] + " (掘金)", doc["text"], source=u, origin="crawl",
                 tags=["juejin", "tech"])
    added = sum(1 for _, st in res if st == "added")
    added_total += added
    print("  OK %-45s +%d cards (q%.2f)" % (doc["title"][:45], added, sc))

a, _ = fts.add_jsonl(os.path.join(root, "cards.jsonl"))
print("kb +%d cards | fts +%d | %s" % (added_total, a, kb.stats()))
if added_total:
    for r in fts.search(QUERY, topk=2):
        print("  VERIFY %.1f [%s] %s | %s" % (r["score"], r["origin"], r["title"][:35], r["text"][:55]))
