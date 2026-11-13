# -*- coding: utf-8 -*-
"""GitHub adapter (A-tier: official REST API, open-source content).
No token = 60 core/h, 10 search/min. Enough for KB seeding runs.
Usage: python github_adapter.py <search query> [max_repos]
"""
import io, json, os, sys, time, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.kb_core import KB
from creeper.quality import quality

sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.github.com"
UA = "aim-kb-creeper/0.1 (self-developed KB seeder)"

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read(5_000_000).decode("utf-8"))

def search_repos(query, n=5):
    d = _get(API + "/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "sort": "stars", "per_page": str(n)}))
    return [(r["full_name"], r.get("description") or "", r["stargazers_count"])
            for r in d.get("items", [])]

def fetch_readme(full_name):
    """Raw README via official API (base64 or raw by Accept header)."""
    req = urllib.request.Request(API + "/repos/%s/readme" % full_name,
                                 headers={"User-Agent": UA,
                                          "Accept": "application/vnd.github.raw"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(1_000_000).decode("utf-8", "ignore")

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "linear attention"
    max_repos = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
    kb = KB(root, build_index=False)
    from kb.fts_index import FtsIndex
    fts = FtsIndex(root)

    repos = search_repos(query, max_repos)
    print("search %r -> %d repos" % (query, len(repos)))
    total_added = 0
    for full, desc, stars in repos:
        try:
            md = fetch_readme(full)
        except Exception as e:
            print("  SKIP %-40s (%s)" % (full, str(e)[:60]))
            continue
        ok, sc, rs = quality(md)
        if not ok:
            print("  LOWQUAL %-40s %s" % (full[:40], rs))
            continue
        title = "%s (GitHub, ★%d)" % (full, stars)
        res = kb.add(title, md, source="https://github.com/" + full,
                     origin="crawl", tags=["github", "readme"])
        added = sum(1 for _, st in res if st == "added")
        total_added += added
        print("  OK    %-40s +%d cards (q%.2f)" % (full[:40], added, sc))
        time.sleep(1)  # polite pacing
    a, n = fts.add_jsonl(os.path.join(root, "cards.jsonl"))
    print("kb +%d cards | fts +%d indexed | stats %s" % (total_added, a, kb.stats()))
    if total_added:
        for r in fts.search(query, topk=3):
            print("  VERIFY %.1f [%s] %s | %s" % (r["score"], r["origin"], r["title"][:40], r["text"][:60]))

if __name__ == "__main__":
    main()
