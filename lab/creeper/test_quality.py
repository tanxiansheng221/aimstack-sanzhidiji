# -*- coding: utf-8 -*-
"""Quality gate test: known junk must reject, known good must pass."""
import io, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creeper.quality import quality

sys.stdout.reconfigure(encoding="utf-8")

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
cards = [json.loads(l) for l in io.open(os.path.join(root, "cards.jsonl"), encoding="utf-8")]

junk = [c for c in cards if c["origin"] == "crawl" and ("QQ浏览器" in c["title"] or "思源学堂" in c["title"])]
good = [c for c in cards if c["origin"] == "wiki" and c["title"] == "线性代数"][:1] + \
       [c for c in cards if c["origin"] == "user"][:1]

fails = 0
for c in junk:
    ok, sc, rs = quality(c["text"])
    tag = "REJECT" if not ok else "PASS!!"
    if ok: fails += 1
    print("%s %.2f %s | %s" % (tag, sc, rs, c["title"][:40]))
for c in good:
    ok, sc, rs = quality(c["text"])
    tag = "pass" if ok else "REJECT!!"
    if not ok: fails += 1
    print("%s %.2f %s | %s" % (tag, sc, rs, c["title"][:40]))
print("RESULT:", "ALL CORRECT" if fails == 0 else "%d WRONG" % fails)
