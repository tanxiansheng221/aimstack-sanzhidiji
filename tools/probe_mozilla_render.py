# -*- coding: utf-8 -*-
"""火狐官方文档终测: support.mozilla.org 壳页走 Edge 渲染级联"""
import sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from creeper.render import render_fetch
from creeper.quality import quality

targets = [
    ("SUM支持文档", "https://support.mozilla.org/zh-CN/kb/firefox-basics"),
    ("SUM英文文档", "https://support.mozilla.org/en-US/kb/get-started-firefox"),
]
for name, url in targets:
    t0 = time.time()
    try:
        doc, st = render_fetch(url, timeout_ms=30000)
        dt = time.time() - t0
        if not doc:
            print("%-10s -> %s (%.1fs)" % (name, st, dt))
            continue
        ok, sc, rs = quality(doc["text"])
        print("%-10s -> %d 字 | q=%.2f | %s | %.1fs" % (
            name, len(doc["text"]), sc, "PASS" if ok else "REJECT",
            dt))
        print("    样本:", doc["text"][:120].replace("\n", " "))
    except Exception as e:
        print("%-10s -> EXC %s" % (name, str(e)[:90]))
