# -*- coding: utf-8 -*-
"""Smoke mediawiki adapter against a GFW-reachable MediaWiki site."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creeper.engine import FetchKernel, fetch_mediawiki, ingest_doc
from kb.kb_core import KB

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
kb = KB(root)
kernel = FetchKernel(min_interval=1.0)
for api, title in [("https://zh.moegirl.org.cn/api.php", "初音未来")]:
    doc, st = fetch_mediawiki(kernel, api, title)
    if doc:
        added, total = ingest_doc(kb, doc, source=api)
        print("[mediawiki %s] %s | +%d/%d cards" % (api.split('/')[2], doc["title"][:30], added, total))
        for r in kb.search(title, topk=2):
            print("  %.1f %s | %s" % (r["score"], r["origin"], r["text"][:60]))
    else:
        print("[mediawiki] FAIL:", st)
