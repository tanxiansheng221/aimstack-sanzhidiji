# -*- coding: utf-8 -*-
import sys, traceback
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from creeper.engine import FetchKernel, fetch_generic

kernel = FetchKernel(min_interval=1.0)
for url in ("https://www.firefox.com/zh-CN/", "https://support.mozilla.org/zh-CN/kb/firefox-basics"):
    print("=" * 20, url)
    try:
        doc, st = fetch_generic(kernel, url, title="t")
        print("st:", st, "| chars:", len(doc["text"]) if doc else 0)
    except Exception:
        traceback.print_exc()
