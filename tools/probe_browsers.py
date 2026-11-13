# -*- coding: utf-8 -*-
"""四家浏览器厂商官网/文档站 正文可抓性实测
宙斯浏览器 / 火狐(Mozilla) / 迅雷浏览器 / 360浏览器"""
import sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from creeper.serp import multi_search
from creeper.engine import FetchKernel, fetch_generic
from creeper.render import render_fetch
from creeper.quality import quality
import urllib.request, urllib.robotparser

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0"

def robots_summary(host):
    """返回 (对通配 * 的判定, 原始关键行)"""
    try:
        t = urllib.request.urlopen(urllib.request.Request(
            "https://" + host + "/robots.txt", headers={"User-Agent": UA}),
            timeout=12).read().decode("utf-8", "ignore")
        cur, star = None, []
        for line in t.splitlines():
            s = line.strip()
            if s.lower().startswith("user-agent:"):
                cur = s.split(":", 1)[1].strip()
            elif cur == "*" and s:
                star.append(s)
        return ("全站禁入" if any("Disallow: /" == x for x in star)
                else ("有限制" if star else "默认放行"), star[:6])
    except Exception as e:
        return ("无 robots/不可达", str(e)[:50])

kernel = FetchKernel(min_interval=1.0)
TARGETS = [("宙斯浏览器", "宙斯浏览器 官网"), ("火狐Mozilla", "火狐浏览器 官网 Mozilla"),
           ("迅雷浏览器", "迅雷浏览器 官网"), ("360浏览器", "360浏览器 官网")]

for label, q in TARGETS:
    print("=" * 66, flush=True)
    print("◆", label, flush=True)
    hits = multi_search(q, engines=("bing", "baidu", "so360"), per_engine=4)
    picked, seen = [], set()
    for h in hits:
        try:
            dom = h["url"].split("/")[2]
        except Exception:
            continue
        if dom not in seen:
            seen.add(dom)
            picked.append(h)
        if len(picked) >= 3:
            break
    for h in picked:
        url = h["url"]
        try:
            dom = url.split("/")[2]
        except Exception:
            continue
        verdict, key = robots_summary(dom)
        print("  官网候选:", url[:58], flush=True)
        print("    robots:", verdict, key, flush=True)
        doc, st = fetch_generic(kernel, url, title=h.get("title"))
        if not doc and st == "too_little_text" and kernel._robot_ok(url):
            doc, st = render_fetch(url, timeout_ms=30000)
        if not doc and st and "fetch_error" in str(st) and kernel._robot_ok(url):
            doc, st = render_fetch(url, timeout_ms=30000)
        if doc:
            ok, sc, rs = quality(doc["text"])
            print("    正文: %d 字 质量分 %.2f (%s) 首行: %s"
                  % (len(doc["text"]), sc, "过闸" if ok else rs, doc["text"][:40].replace("\n", " ")), flush=True)
        else:
            print("    正文: FAIL", st, flush=True)
        print(flush=True)
