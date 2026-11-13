# -*- coding: utf-8 -*-
"""验证聊天底栏插件直调: 页面含 capgo/capsel + capcall 端点仍通"""
import sys, json, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

B = "http://127.0.0.1:8600"
html = urllib.request.urlopen(B + "/", timeout=10).read().decode("utf-8", "ignore")
print("页面含 capgo 按钮:", 'id=capgo' in html)
print("页面含 capsel 下拉:", 'id=capsel' in html)
print("页面含 capCall 函数:", 'async function capCall' in html)

req = urllib.request.Request(B + "/api/capcall",
                             data=json.dumps({"cap": "demo.time", "q": "ui", "topk": 1}).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    print("capcall ->", json.loads(r.read().decode("utf-8", "ignore")).get("cap"))
