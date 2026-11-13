# -*- coding: utf-8 -*-
"""服务器就绪探针 + 端到端延迟测试 (联网全链)"""
import sys, json, time, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def post(path, payload, timeout=660):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8600" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8", "ignore"))
    return body, time.time() - t0

# 1) 就绪探针 (stats 在 POST 分支, GET 用 /api/logs)
try:
    with urllib.request.urlopen("http://127.0.0.1:8600/api/logs", timeout=10) as r:
        print("SERVER READY")
except Exception as e:
    print("NOT READY:", str(e)[:100])
    sys.exit(1)

# 2) 端到端: 联网问题 (会走 auto -> KB miss -> web 链)
q = sys.argv[1] if len(sys.argv) > 1 else "中国的算力现在是什么情况"
res, dt = post("/api/chat", {"q": q, "mode": "auto"})
print("\n端到端耗时: %.1fs | path=%s | ok=%s" % (dt, res.get("path"), res.get("ok")))
tr = res.get("trace") or []
for t in tr:
    print("  [%s] %s | %s" % (t.get("title"), t.get("summary"), (t.get("detail") or "")[:80]))
print("\n回答:", (res.get("answer") or "")[:400])
srcs = res.get("sources") or []
print("来源:", [(s.get("title", "")[:30], s.get("store", s.get("origin", ""))) for s in srcs[:6]])
