# -*- coding: utf-8 -*-
"""SERP adapters: search-entry layer for creeper (多源搜索的入口).
Engines reachable in CN: bing_cn (cn.bing.com), sogou. Returns [{title,url,snippet}].
Then generic adapter crawls each URL -> KB gate. Multi-source search loop:
query -> search -> multi-source fetch -> clean -> gate -> KB.
"""
import io, os, re, sys, time, urllib.parse, urllib.request
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

class _X(HTMLParser):
    """Extract anchor+title pairs from SERP html (heuristic: result blocks)."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._a = None; self._href = None; self._buf = []
        self._depth = 0
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "a" and d.get("href"):
            self._a = True; self._href = d["href"]; self._buf = []
    def handle_data(self, data):
        if self._a and data.strip():
            self._buf.append(data.strip())
    def handle_endtag(self, tag):
        if tag == "a" and self._a:
            text = " ".join(self._buf)
            if text and self._href and self._href.startswith("http") and len(text) > 8:
                if not re.search(r"\.(bing|sogou|microsoft)\.com", self._href):
                    self.results.append({"title": text[:120], "url": self._href})
            self._a = None; self._href = None

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read(1_500_000).decode("utf-8", "ignore")

def _strip_tags(s):
    import html as _h
    s = _h.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


RENDER_MIN = 8000   # 裸抓低于此字节数 = 反爬壳/残页 -> Edge 渲染级联

def _get_smart(url, result_marker=None):
    """裸抓 -> 太薄或缺结果块 -> Edge 渲染重抓 (81.10 级联: plain -> render)"""
    html = _get(url)
    ok = len(html) >= RENDER_MIN and (result_marker is None or result_marker in html)
    if ok:
        return html
    try:
        from creeper.render import render_html
        h2, _t = render_html(url)
        if h2 and len(h2) > len(html):
            return h2
    except Exception:
        pass
    return html


def search_bing_cn(query, n=8):
    """结果块级抽取: <li class=b_algo> 内含标题链接 + 摘要文本 (snippet); 壳页走渲染级联"""
    html = _get_smart("https://cn.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "count": str(n)}), result_marker="b_algo")
    out = []
    for blk in re.findall(r'<li class="b_algo".*?</li>', html, re.S):
        m = re.search(r'<h2[^>]*><a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        url, title = m.group(1), _strip_tags(m.group(2))
        m2 = re.search(r'<div class="b_caption"[^>]*><p[^>]*>(.*?)</p>', blk, re.S)
        snip = _strip_tags(m2.group(1)) if m2 else ""
        if (title and len(title) > 6 and len(snip) >= 20
                and not re.search(r"\.(bing|microsoft)\.com", url)):
            if url not in [r["url"] for r in out]:
                out.append({"title": title[:120], "url": url,
                            "snippet": snip[:300]})
        if len(out) >= n:
            break
    if not out:   # 兜底: 旧解析器 (只有标题)
        p = _X(); p.feed(html)
        return p.results[:n]
    return out

def search_sogou(query, n=8):
    """按 <h3> 切结果块, 抽标题 + 摘要; 反爬壳页自动走 Edge 渲染级联"""
    html = _get_smart("https://www.sogou.com/web?" + urllib.parse.urlencode(
        {"query": query}), result_marker="<h3")
    out = []
    for blk in re.split(r"<h3", html)[1:]:
        m = re.search(r'<a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        url, title = m.group(1), _strip_tags(m.group(2))
        snip = _strip_tags(re.sub(r"<a.*?</a>", " ", blk, count=1, flags=re.S))
        if title and len(title) > 6 and not re.search(r"\.(sogou|qq)\.com", url):
            if url not in [r["url"] for r in out]:
                out.append({"title": title[:120], "url": url,
                            "snippet": snip[:300]})
        if len(out) >= n:
            break
    if not out:   # 兜底: 旧解析器
        p = _X(); p.feed(html)
        return p.results[:n]
    return out

def search_baidu(query, n=8):
    """百度 SERP (h3 切块; 类名随机化, 走纯文本抽取; 链接是跳转链可跟)"""
    html = _get_smart("https://www.baidu.com/s?" + urllib.parse.urlencode(
        {"wd": query, "rn": "8"}), result_marker="<h3")
    out = []
    for blk in re.split(r"<h3", html)[1:]:
        m = re.search(r'<a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        url, title = m.group(1), _strip_tags(m.group(2))
        seg = blk.split("</h3>", 1)[-1] if "</h3>" in blk else blk
        snip = _strip_tags(seg)[:280]
        if title and len(title) > 5 and len(snip) >= 15:
            if url not in [r["url"] for r in out] and "baidu.com" not in url:
                out.append({"title": title[:120], "url": url,
                            "snippet": snip[:300]})
        if len(out) >= n:
            break
    return out


def search_so360(query, n=8):
    """360 搜索 (h3 切块, 同款抽取)"""
    html = _get_smart("https://www.so.com/s?" + urllib.parse.urlencode(
        {"q": query}), result_marker="res-list")
    out = []
    for blk in re.split(r"<h3", html)[1:]:
        m = re.search(r'<a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        url, title = m.group(1), _strip_tags(m.group(2))
        seg = blk.split("</h3>", 1)[-1] if "</h3>" in blk else blk
        snip = _strip_tags(seg)[:280]
        if title and len(title) > 5 and len(snip) >= 15:
            if url not in [r["url"] for r in out] and "so.com" not in url:
                out.append({"title": title[:120], "url": url,
                            "snippet": snip[:300]})
        if len(out) >= n:
            break
    return out


ENGINES = {"bing": search_bing_cn, "sogou": search_sogou,
           "baidu": search_baidu, "so360": search_so360}

def multi_search(query, engines=("bing", "baidu", "so360"), per_engine=6,
                 parallel=True):
    """多引擎并行 (线程池, 不排队) -> 交错合并去重"""
    pools = []
    live = [e for e in engines if e in ENGINES]
    if parallel and len(live) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(live)) as ex:
            futs = [(e, ex.submit(ENGINES[e], query, per_engine)) for e in live]
            for e, f in futs:
                try:
                    pools.append(f.result(timeout=90))
                except Exception as ex2:
                    print("engine %s fail: %s" % (e, str(ex2)[:80]), flush=True)
    else:
        for e in live:
            try:
                pools.append(ENGINES[e](query, per_engine))
            except Exception as ex2:
                print("engine %s fail: %s" % (e, str(ex2)[:80]), flush=True)
    merged, seen = [], set()
    for i in range(per_engine):
        for pool in pools:
            if i < len(pool) and pool[i]["url"] not in seen:
                seen.add(pool[i]["url"]); merged.append(pool[i])
    return merged

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "线性注意力 三值权重"
    for r in multi_search(q):
        print("-", r["title"][:60], "|", r["url"][:90])
