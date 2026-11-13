# -*- coding: utf-8 -*-
"""Creeper: self-developed universal crawler engine (81.3). Stdlib only.

Layers: fetch kernel (urllib, robots-aware, rate-limited) -> site adapters
(mediawiki API / generic html) -> clean via kb.ingest -> KB gate (fail-closed).
"""
import io, json, os, re, sys, time, urllib.request, urllib.robotparser, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.kb_core import KB
from kb.ingest import html_to_text

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

UA = "AimCreeper/0.1 (+self-developed; respects robots.txt)"
MAX_BYTES = 2_000_000

class FetchKernel:
    """Layer 1: polite fetch. robots.txt + per-host rate limit + size cap."""
    def __init__(self, min_interval=2.0):
        self.min_interval = min_interval
        self._last = {}
        self._robots = {}

    def _robot_ok(self, url):
        host = urllib.parse.urlsplit(url).netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                req = urllib.request.Request("https://" + host + "/robots.txt",
                                             headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=10) as r:
                    rp.parse(r.read(200_000).decode("utf-8", "ignore").splitlines())
            except Exception:
                rp = None  # unreachable robots -> treat as allowed, fetch carefully
            self._robots[host] = rp
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(UA, url)

    def fetch(self, url):
        if not self._robot_ok(url):
            return (None, None), "robots_disallow"
        host = urllib.parse.urlsplit(url).netloc
        wait = self.min_interval - (time.time() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read(MAX_BYTES)
            return (data, ctype), "ok"
        except Exception as e:
            return (None, None), "fetch_error:" + str(e)[:120]

# ---------------- adapters (Layer 2) ----------------
def fetch_mediawiki(kernel, api, title, lang="zh"):
    """Official MediaWiki API plaintext extract (zero scrape, A-tier compliant)."""
    u = api + "?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1})
    (data, _), st = kernel.fetch(u)
    if st != "ok":
        return None, st
    d = json.loads(data.decode("utf-8", "ignore"))
    pages = d.get("query", {}).get("pages", {})
    for _, p in pages.items():
        ext = p.get("extract")
        if ext:
            return {"title": p.get("title", title), "text": ext}, "ok"
    return None, "no_extract"

def fetch_generic(kernel, url, title=None):
    """Any HTML page -> text (A/B-tier single page)."""
    (data, ctype), st = kernel.fetch(url)
    if st != "ok":
        return None, st
    if b"html" not in ctype.encode("latin-1", "ignore") and not data[:512].lstrip().startswith(b"<"):
        return {"title": title or url, "text": data.decode("utf-8", "ignore")}, "ok"
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            html = data.decode(enc)
            break
        except Exception:
            continue
    else:
        html = data.decode("utf-8", "ignore")
    t, text = html_to_text(html)
    if len(text) < 40:
        return None, "too_little_text"
    return {"title": t or title or url, "text": text}, "ok"

ADAPTERS = {
    "mediawiki": fetch_mediawiki,   # wikipedia etc (official API)
    "generic": fetch_generic,       # any page
}

# ---------------- ingest into KB (Layer 3+4) ----------------
def ingest_doc(kb, doc, origin="crawl", source=None):
    res = kb.add(doc["title"], doc["text"], source=source or "crawl",
                 origin=origin, tags=["creeper"])
    added = sum(1 for _, st in res if st == "added")
    return added, len(res)

if __name__ == "__main__":
    # smoke: crawl real pages -> KB -> search back
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")
    kb = KB(root)
    kernel = FetchKernel(min_interval=1.0)
    jobs = [
        ("mediawiki", {"api": "https://zh.wikipedia.org/w/api.php", "title": "线性代数"}),
        ("generic", {"url": "https://docs.python.org/3/tutorial/datastructures.html"}),
    ]
    for name, kw in jobs:
        doc, st = ADAPTERS[name](kernel, **kw)
        if doc:
            added, total = ingest_doc(kb, doc, source=kw.get("url") or kw.get("api"))
            print("[%s] %s | cards added %d/%d" % (name, doc["title"][:40], added, total))
        else:
            print("[%s] FAIL: %s" % (name, st))
    print("kb stats:", kb.stats())
    for q in ("线性代数 矩阵", "list methods"):
        print("Q:", q)
        for r in kb.search(q, topk=2):
            print("  %.1f %s | %s" % (r["score"], r["id"], r["text"][:70]))
