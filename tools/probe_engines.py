# -*- coding: utf-8 -*-
"""探测: 百度/360 SERP 裸抓可达性 + 渲染救超时站"""
import sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from creeper.serp import _get

Q = "%E4%B8%AD%E5%9B%BD%E7%AE%97%E5%8A%9B%E7%8E%B0%E7%8A%B6"  # 中国算力现状
TESTS = [
    ("BAIDU", "https://www.baidu.com/s?wd=" + Q + "&rn=8", "h3"),
    ("SO360", "https://www.so.com/s?q=" + Q, "res-list"),
    ("SO360b", "https://www.so.com/s?q=" + Q, "result"),
]
for name, u, mk in TESTS:
    try:
        t0 = time.time()
        h = _get(u)
        print("%-7s %5.1fs LEN=%7d marker[%s]=%s" % (name, time.time() - t0, len(h), mk, mk in h))
        if name == "BAIDU" and "h3" in h:
            print("   title sample:", h[h.find("<title"):h.find("</title>") + 8][:80])
    except Exception as e:
        print("%-7s ERR %s" % (name, str(e)[:90]))

# 渲染救超时站: sz.gov.cn 之前裸抓 fetch_error 超时
print("--- render rescue test ---")
from creeper.render import render_fetch
t0 = time.time()
doc, st = render_fetch("https://www.sz.gov.cn/", timeout_ms=30000)
print("sz.gov.cn: %.1fs" % (time.time() - t0), "OK" if doc else st,
      len(doc["text"]) if doc else "")
