# -*- coding: utf-8 -*-
"""路由 v2 终验: 你截图的原话必须走联网, 闲聊仍直答"""
import json, sys, time, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

URL = "http://127.0.0.1:8600"

def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), time.time() - t0

total = npass = 0
def rec(name, ok, notes):
    global total, npass
    total += 1
    npass += bool(ok)
    print("[%s] %-14s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

# 1. 你截图的原话 (无 搜索/查一下 字眼) -> 必须走联网链, 不得纯对话
r, dt = post("/api/chat", {"q": "现在中国的算力如何什么情况", "mode": "auto"})
steps = [t["title"] for t in r.get("trace", [])]
ok = (r.get("path") in ("web", "digest", "honest")) and "🌐 搜索引擎" in steps
rec("AUTO_WEB_FALLBACK", ok, "%.0fs path=%s trace=%s" % (dt, r.get("path"), steps))
print("   answer:", (r.get("answer") or "")[:150], flush=True)

# 2. 闲聊白名单 -> 直答 (不被误送联网)
r, dt = post("/api/chat", {"q": "在吗", "mode": "auto"})
rec("CHITCHAT_DIRECT", r.get("path") == "direct", "%.0fs path=%s" % (dt, r.get("path")))

print("\n=== ROUTER-V2: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
