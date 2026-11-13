# -*- coding: utf-8 -*-
"""Show last crawled cards + verify one of their own words is retrievable."""
import io, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.fts_index import FtsIndex

sys.stdout.reconfigure(encoding="utf-8")

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
cards = [json.loads(l) for l in io.open(os.path.join(root, "cards.jsonl"), encoding="utf-8")]
new = [c for c in cards if c["origin"] == "crawl"][-4:]
for c in new:
    print("-", c["id"], c["title"][:50], "|", c["text"][:80].replace("\n", " "))

fts = FtsIndex(root)
if new:
    # take a distinctive token from the newest card
    for c in reversed(new):
        text = c["text"]
        for w in (c["title"].split() + text.split()):
            w = w.strip("，。：；、（）()[]【】 ")
            if len(w) >= 4:
                hits = fts.search(w, topk=2)
                if hits:
                    print("\nVERIFY query=%r" % w)
                    for h in hits:
                        print("  %.1f [%s] %s | %s" % (h["score"], h["origin"], h["title"][:30], h["text"][:60]))
                    sys.exit(0)
print("no verifiable token found")
