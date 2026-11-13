# -*- coding: utf-8 -*-
"""严格复测: 1) bing/cn.bing robots.txt 原文 + 我们 SERP 入口是否踩线
           2) 火狐真实文档页正文抓取 (非首页壳)"""
import sys, io, re, urllib.request, urllib.robotparser

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

def get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(2_000_000).decode("utf-8", "ignore")

print("=" * 30, "PART 1: BING ROBOTS", "=" * 30)
for host in ("https://www.bing.com/robots.txt", "https://cn.bing.com/robots.txt"):
    print("\n---", host)
    try:
        txt = get(host)
        # 只打印与 * / bingbot 相关的规则段
        cur = None
        for line in txt.splitlines():
            l = line.strip()
            if l.lower().startswith("user-agent:"):
                cur = l.split(":", 1)[1].strip()
            if cur in ("*",) or "bingbot" in cur.lower():
                if l and not l.startswith("#"):
                    print("   ", l)
    except Exception as e:
        print("    FETCH FAIL:", str(e)[:100])

print("\nSERP 入口核对: 我们抓的是 cn.bing.com/search?q=... (拿结果链接+摘要当索引用)")
for host, path in (("https://www.bing.com", "/search"),
                   ("https://cn.bing.com", "/search")):
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(host + "/robots.txt")
        rp.read()
        print("  robotparser %s%s -> %s" % (host, path, rp.can_fetch(UA, host + path)))
        print("  robotparser(通配UA) ->", rp.can_fetch("*", host + path))
    except Exception as e:
        print("  robotparser fail:", str(e)[:80])

print()
print("=" * 30, "PART 2: FIREFOX REAL DOCS", "=" * 30)
from creeper.engine import FetchKernel, fetch_generic
from creeper.quality import quality

kernel = FetchKernel(min_interval=1.0)
# 首页壳 vs 真文档页 对照
targets = [
    ("firefox首页", "https://www.firefox.com/zh-CN/"),
    ("firefox英文首页", "https://www.firefox.com/en-US/"),
    ("Mozilla支持文档", "https://support.mozilla.org/zh-CN/kb/firefox-basics"),
    ("MDN文档页", "https://developer.mozilla.org/zh-CN/docs/Web/JavaScript"),
    ("Firefox release notes", "https://www.mozilla.org/en-US/firefox/140.0/releasenotes/"),
]
for name, url in targets:
    try:
        doc, st = fetch_generic(kernel, url, title=name)
        if not doc:
            print("%-14s %s -> %s" % (name, "NO DOC", st[:80]))
            continue
        ok, sc, rs = quality(doc["text"])
        print("%-14s -> %d 字 | quality=%.2f (%s) | 标题: %s" % (
            name, len(doc["text"]), sc, "PASS" if ok else "REJECT:" + rs[:40],
            (doc["title"] or "")[:50]))
        print("    正文样本:", doc["text"][:160].replace("\n", " "))
    except Exception as e:
        print("%-14s -> EXC %s" % (name, str(e)[:90]))
