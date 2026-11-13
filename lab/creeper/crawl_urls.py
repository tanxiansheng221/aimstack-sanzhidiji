# -*- coding: utf-8 -*-
"""Crawl explicit good URLs through the full pipeline (robots->plain/render->quality->KB)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creeper.engine import FetchKernel, fetch_generic, ingest_doc
from creeper.render import render_fetch
from creeper.quality import quality
from kb.kb_core import KB
from kb.fts_index import FtsIndex

sys.stdout.reconfigure(encoding="utf-8")

URLS = [
    "https://meko1.github.io/llm-interview-guide/pretraining/llm-training-pipeline",
    "https://docs.swanlab.cn/examples/pretrain_llm.html",
    "https://developer.aliyun.com/article/1708555",
    "https://blog.csdn.net/Python_0011/article/details/139295470",
    "https://blog.csdn.net/2401_85375186/article/details/145845778",
]

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
kb = KB(root, build_index=False)
fts = FtsIndex(root)
kernel = FetchKernel(min_interval=2.0)

added_total = 0
for u in URLS:
    doc, st = fetch_generic(kernel, u)
    ok, sc, rs = (False, 0.0, ["no_doc:" + st]) if not doc else quality(doc["text"])
    if not ok and st != "robots_disallow":
        # cascade: plain text failed quality -> real browser render
        rdoc, rst = render_fetch(u)
        if rdoc:
            rok, rsc, rrs = quality(rdoc["text"])
            if rok or rsc > sc:  # render strictly better
                doc, ok, sc, rs = rdoc, rok, rsc, rrs
    if not doc:
        print("SKIP %-58s (%s)" % (u[:58], st)); continue
    if not ok:
        print("LOWQUAL %-53s %s" % (u[:53], rs)); continue
    res = kb.add(doc["title"], doc["text"], source=u, origin="crawl", tags=["web"])
    added = sum(1 for _, st in res if st == "added")
    added_total += added
    print("OK   %-58s +%d cards (q%.2f)" % (u[:58], added, sc))

a, _ = fts.add_jsonl(os.path.join(root, "cards.jsonl"))
print("kb +%d cards | fts +%d | %s" % (added_total, a, kb.stats()))
if added_total:
    for r in fts.search("大语言模型 训练", topk=3):
        print("  VERIFY %.1f [%s] %s | %s" % (r["score"], r["origin"], r["title"][:35], r["text"][:55]))
