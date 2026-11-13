# -*- coding: utf-8 -*-
"""摸百度/360 SERP 的结果块结构"""
import re, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from creeper.serp import _get

h = _get("https://www.baidu.com/s?wd=%E6%B7%B1%E5%9C%B3%E5%A5%BD%E7%8E%A9%E7%9A%84%E6%99%AF%E5%8C%BA&rn=8")
i = h.find("<h3")
print("BAIDU h3 block:")
print(re.sub(r"\s+", " ", h[i:i + 900])[:900])

h2 = _get("https://www.so.com/s?q=%E6%B7%B1%E5%9C%B3%E5%A5%BD%E7%8E%A9%E7%9A%84%E6%99%AF%E5%8C%BA")
j = h2.find("res-list")
print("\nSO360 res block:")
print(re.sub(r"\s+", " ", h2[j:j + 900])[:900])
