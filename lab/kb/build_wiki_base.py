# -*- coding: utf-8 -*-
"""Build wiki base layer: convert kaggle chunks -> KB cards -> FTS5 index."""
import gzip, glob, hashlib, io, json, os, re, sys, time, unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.fts_index import FtsIndex

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 用法: python lab/kb/build_wiki_base.py [切块目录]  (目录内含 wiki_zh_*.jsonl.gz,
#   每行 {"title":..., "text":...}; 不传参默认 data/wiki/)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "data", "wiki")
ROOT = os.environ.get("AIM_KB_BASE", os.path.join(_ROOT, "kb_store"))
CARDS = os.path.join(ROOT, "cards.jsonl")
os.makedirs(ROOT, exist_ok=True)

def _norm(s):
    return re.sub(r"\s+", " ", s).strip()

def run():
    t0 = time.time()
    existing = set()
    if os.path.exists(CARDS):
        for l in io.open(CARDS, encoding="utf-8"):
            try:
                existing.add(json.loads(l)["id"])
            except Exception:
                pass
    print("existing cards:", len(existing))
    out = io.open(CARDS, "a", encoding="utf-8")
    n = added = skipped = 0
    for f in sorted(glob.glob(os.path.join(SRC, "wiki_zh_*.jsonl.gz"))):
        with io.TextIOWrapper(gzip.open(f), encoding="utf-8") as g:
            for line in g:
                n += 1
                try:
                    d = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                title = _norm(d["title"])
                text = _norm(d["text"])
                if len(text) < 40:
                    skipped += 1
                    continue
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                cid = "kb-" + h
                if cid in existing:
                    continue
                rec = {"id": cid, "title": title, "text": text,
                       "source": d.get("url") or "wikipedia-zh",
                       "origin": "wiki", "tags": ["wiki-base", "zh"],
                       "ts": int(time.time()), "sha256": h, "part": 0}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing.add(cid)
                added += 1
                if added % 100000 == 0:
                    print("  %dk converted (%.0fs)" % (added // 1000, time.time() - t0), flush=True)
    out.close()
    print("converted %d/%d (skip %d) in %.0fs" % (added, n, skipped, time.time() - t0))

    idx = FtsIndex(ROOT)
    a2, n2 = idx.add_jsonl(CARDS, batch=10000)
    print("FTS indexed +%d, total %d in %.0fs" % (a2, idx.count(), time.time() - t0))

if __name__ == "__main__":
    run()
