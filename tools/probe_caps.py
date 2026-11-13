# -*- coding: utf-8 -*-
"""插件面板端点验证: GET /api/caps + POST /api/capcall (demo.time 与 kb.search)"""
import sys, json, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

B = "http://127.0.0.1:8600"

def get(p):
    with urllib.request.urlopen(B + p, timeout=15) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))

def post(p, payload, timeout=120):
    req = urllib.request.Request(B + p, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))

caps = get("/api/caps")
print("GET /api/caps ->", [c["name"] for c in caps.get("caps", [])])

r1 = post("/api/capcall", {"cap": "demo.time", "q": "hello", "topk": 1})
print("capcall demo.time ->", json.dumps(r1, ensure_ascii=False)[:200])

r2 = post("/api/capcall", {"cap": "kb.search", "q": "算力", "topk": 2})
n = len((r2.get("results") or []))
print("capcall kb.search -> results=%d _bus_s=%s" % (n, r2.get("_bus_s")))

r3 = post("/api/capcall", {"cap": "not.exist", "q": "x", "topk": 1})
print("capcall 未注册插件 ->", r3)
