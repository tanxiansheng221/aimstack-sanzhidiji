# -*- coding: utf-8 -*-
"""UI 服务 API 端点验收"""
import json, os, urllib.request, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

URL = "http://127.0.0.1:8600"

def post(path, body=None, raw=None, headers=None, timeout=420):
    data = raw if raw is not None else json.dumps(body or {}).encode("utf-8")
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(URL + path, data=data, headers=h)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), time.time() - t0

total = npass = 0
def rec(name, ok, notes):
    global total, npass
    total += 1
    npass += bool(ok)
    print("[%s] %-14s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

# 1. 页面
with urllib.request.urlopen(URL + "/", timeout=10) as r:
    html = r.read().decode("utf-8")
rec("PAGE", "aimp 加载器" in html and len(html) > 4000, "html %d bytes" % len(html))

# 2. stats
r, dt = post("/api/stats")
rec("STATS", r.get("base", 0) > 700000 and r.get("dyn", 0) > 300,
    "base=%s dyn=%s" % (r.get("base"), r.get("dyn")))

# 3. kbsearch
r, dt = post("/api/kbsearch", {"q": "文艺复兴", "topk": 3})
rec("KBSEARCH", r.get("results") and r["results"][0].get("store") == "base",
    "%.1fs %d hits store=%s" % (dt, len(r.get("results", [])),
                                r["results"][0]["store"] if r.get("results") else "-"))

# 4. websearch
r, dt = post("/api/websearch", {"q": "BitNet b1.58 inference", "topk": 3})
rec("WEBSEARCH", bool(r.get("results")), "%.1fs %d hits" % (dt, len(r.get("results", []))))

# 5. chat: 纯对话 (闲聊走 direct)
r, dt = post("/api/chat", {"q": "在吗", "mode": "direct"})
rec("CHAT_DIRECT", r.get("ok") and r.get("path") == "direct" and r.get("answer"),
    "%.1fs [%s] %s" % (dt, r.get("path"), (r.get("answer") or "")[:20]))

# 6. chat: auto + 知识题 (RAG 引答, 带 sources)
r, dt = post("/api/chat", {"q": "Textadept 是什么", "mode": "auto"})
rec("CHAT_KB", r.get("ok") and r.get("path") == "kb" and r.get("sources"),
    "%.1fs [%s] hits=%s %s" % (dt, r.get("path"), r.get("kb_hits"),
                               (r.get("answer") or "")[:20]))

# 7. chat: auto + 库里没有 -> honest
r, dt = post("/api/chat", {"q": "M11 步长调度 复现判决", "mode": "kb"})
rec("CHAT_HONEST", r.get("ok") and r.get("honest_miss"), "%.1fs [%s]" % (dt, r.get("path")))

# 8. kbadd: 上传一个新文档
mk = "AIMPUI%s" % str(int(time.time()))[-6:]
body = ("# %s\n\nUI 上传入库验收文档: aimp local ui kbadd endpoint test %s.\n"
        % (mk, mk)).encode("utf-8")
r, dt = post("/api/kbadd", raw=body, headers={"X-Filename": "ui_test_%s.md" % mk})
rec("KBADD", r.get("added") == 1 and r.get("store") == "dyn",
    "%.1fs added=%s dyn_total=%s" % (dt, r.get("added"), r.get("dyn_total")))

# 9. 查回刚传的
r, dt = post("/api/kbsearch", {"q": mk, "topk": 3})
rec("SEARCHBACK", bool(r.get("results")), "%.1fs %d hits" % (dt, len(r.get("results", []))))

# 10. webingest (可达目标)
r, dt = post("/api/webingest", {"q": "Textadept editor", "topn": 1}, timeout=600)
rec("WEBINGEST", r.get("crawled", 0) >= 1 or r.get("rejected", 0) >= 1,
    "%.1fs crawled=%s rejected=%s" % (dt, r.get("crawled"), r.get("rejected")))

print("\n=== UI API: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
