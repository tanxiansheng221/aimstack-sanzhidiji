# -*- coding: utf-8 -*-
"""意图路由端到端验收: 你截图里那句原话 + 各类路由"""
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

# 1. 你截图里的原话 -> 必须走联网全链三步轨迹 (答不出也是诚实拒答, 不许纯对话瞎编)
r, dt = post("/api/chat", {"q": "帮我搜索一下如今中国算力情况", "mode": "auto"})
steps = [t["title"] for t in r.get("trace", [])]
rec("ROUTER_WEB", r.get("ok") and r.get("path") in ("web", "honest")
    and "🌐 搜索引擎" in steps and "🕸 抓取过闸" in steps and "📚 知识库" in steps,
    "%.0fs path=%s trace=%s" % (dt, r.get("path"), steps))
print("   answer: %s" % (r.get("answer") or "")[:120], flush=True)

# 2. 闲聊 -> 仍然直答 (路由不误伤)
r, dt = post("/api/chat", {"q": "在吗", "mode": "auto"})
rec("ROUTER_CHAT", r.get("ok") and r.get("path") == "direct",
    "%.0fs path=%s" % (dt, r.get("path")))

# 3. 知识题 -> 查库引答
r, dt = post("/api/chat", {"q": "Textadept 是什么", "mode": "auto"})
rec("ROUTER_KB", r.get("ok") and r.get("path") == "kb",
    "%.0fs path=%s hits=%s" % (dt, r.get("path"), r.get("kb_hits")))

# 4. 入库意图 -> 网页入库流
r, dt = post("/api/chat", {"q": "把 BitNet 相关网页爬下来入库", "mode": "auto"}, timeout=600)
rec("ROUTER_ING", r.get("ok") and r.get("path") == "ingest",
    "%.0fs path=%s trace=%s" % (dt, r.get("path"),
    [t["title"] for t in r.get("trace", [])]))

# 5. 搜索不可达目标 -> 诚实拒答带轨迹
r, dt = post("/api/chat", {"q": "帮我搜索一下 UNreachable topic zzx", "mode": "auto"}, timeout=600)
ok = r.get("ok") and (r.get("path") in ("web", "honest"))
rec("ROUTER_HONEST", ok, "%.0fs path=%s" % (dt, r.get("path")))

print("\n=== ROUTER: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
