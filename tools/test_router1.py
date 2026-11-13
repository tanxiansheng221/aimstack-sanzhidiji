# -*- coding: utf-8 -*-
"""路由第 1 项终验 (修正判据)"""
import json, sys, time, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

req = urllib.request.Request("http://127.0.0.1:8600/api/chat",
    data=json.dumps({"q": "帮我搜索一下如今中国算力情况", "mode": "auto"}).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=600).read().decode("utf-8"))
steps = [t["title"] for t in r.get("trace", [])]
print("path:", r.get("path"), "| %.0fs" % (time.time() - t0))
print("trace:", steps)
for t in r.get("trace", []):
    print("  ", t["title"], "--", t["summary"])
print("answer:", (r.get("answer") or "")[:120])
ok = r.get("path") in ("web", "honest") and len(steps) == 3
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
