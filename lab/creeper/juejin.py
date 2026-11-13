# -*- coding: utf-8 -*-
"""Juejin adapter (official content API, markdown bodies). B-tier compliant.
List by category -> detail (markdown) -> quality gate -> KB.
Usage: python juejin.py [category_id] [max_articles]   (default category 6809637767543259144=后端)
Known category ids: 6809637767543259144 后端 | 6809637769959178254 前端 | 6809637773935378443 人工智能
"""
import io, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.kb_core import KB
from kb.fts_index import FtsIndex
from creeper.quality import quality

sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.juejin.cn"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"

def _post(path, payload):
    req = urllib.request.Request(API + path,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read(5_000_000).decode("utf-8"))

def list_articles(category_id, n=10):
    d = _post("/content_api/v1/article/query_list",
              {"id_type": 2, "sort_type": 2, "cursor": "0", "limit": n,
               "category_id": category_id})
    return [a["article_id"] for a in d.get("data", []) or []]

def fetch_article(article_id):
    d = _post("/content_api/v1/article/detail", {"article_id": article_id})
    info = (d.get("data") or {}).get("article_info") or {}
    title = info.get("title") or ""
    md = info.get("mark_content") or ""
    return {"id": article_id, "title": title, "md": md,
            "url": "https://juejin.cn/post/" + article_id}

def main():
    cat = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "6809637773935378443"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
    kb = KB(root, build_index=False)
    fts = FtsIndex(root)

    ids = list_articles(cat, n)
    print("juejin category %s -> %d article ids" % (cat, len(ids)))
    added_total = 0
    for aid in ids:
        try:
            art = fetch_article(aid)
        except Exception as e:
            print("  SKIP %s (%s)" % (aid, str(e)[:60])); continue
        if not art["md"]:
            print("  SKIP %s (no content)" % aid); continue
        ok, sc, rs = quality(art["md"])
        if not ok:
            print("  LOWQUAL %-30s %s" % (art["title"][:30], rs)); continue
        res = kb.add(art["title"] + " (掘金)", art["md"], source=art["url"],
                     origin="crawl", tags=["juejin", "tech"])
        added = sum(1 for _, st in res if st == "added")
        added_total += added
        print("  OK %-42s +%d cards (q%.2f)" % (art["title"][:42], added, sc))
        time.sleep(1.2)  # polite pacing
    a, _ = fts.add_jsonl(os.path.join(root, "cards.jsonl"))
    print("kb +%d cards | fts +%d | %s" % (added_total, a, kb.stats()))

if __name__ == "__main__":
    main()
