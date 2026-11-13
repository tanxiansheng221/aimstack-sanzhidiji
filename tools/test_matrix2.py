# -*- coding: utf-8 -*-
"""全因子矩阵: 对话 3 问题类型 x 3 问法 + KB 3 查询 + WEB 3 查询 + 3 插件各 3 遍"""
import re, subprocess, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
W = r"H:\AimCode\aimp\data\bitnet2b\m2b_tern.aimw"
TOK = r"H:\AimCode\aimp\data\m2b_llama3_head.gguf"
MINI = r"H:\AimCode\aimp\build\mini_llama.gguf"
CWD = r"H:\AimCode\aimp"

def run(args, timeout=600):
    t0 = time.time()
    p = subprocess.run(args, cwd=CWD, capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "ignore")
    return out, p.returncode, time.time() - t0

def rec(name, ok, notes):
    print("[%s] %-22s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)
    return bool(ok)

total = npass = 0

# ============ 对话: 3 问题类型 x 3 问法 ============
print("== A 闲聊题 (--gt 直答, 无知识库) ==", flush=True)
for i, q in enumerate(["在吗", "你好呀", "嗨，有人吗"]):
    out, rc, dt = run([EXE, W, TOK, "--gt", q, "32"])
    g = re.search(r"GEN_TEXT:(.*)", out)
    ok = rc == 0 and g and g.group(1).strip()
    npass += rec("A%d_%s" % (i + 1, q[:6]),
                 ok, "%ds reply=%s" % (dt, g.group(1).strip()[:18] if g else "-"))
    total += 1

print("== B 知识题 (--ask RAG, 库里有) ==", flush=True)
for i, q in enumerate(["Textadept 是什么",
                       "介绍一下 Textadept 编辑器",
                       "Textadept 编辑器用 Lua 扩展"]):
    out, rc, dt = run([EXE, W, TOK, "--ask", q, "32"])
    h = re.search(r"KB_HITS:(\d+)", out)
    g = re.search(r"GEN_TEXT:(.*)", out)
    ok = rc == 0 and h and int(h.group(1)) > 0 and g
    npass += rec("B%d_hits%d" % (i + 1, int(h.group(1)) if h else -1),
                 ok, "%ds reply=%s" % (dt, g.group(1).strip()[:18] if g else "-"))
    total += 1

print("== C 边界题 (--ask, 库里没有 -> 必须拒答) ==", flush=True)
for i, q in enumerate(["M11 步长调度 复现判决",
                       "微核 M11 步长调度复现怎么判的",
                       "microcore M11 步长调度 复现失败"]):
    out, rc, dt = run([EXE, W, TOK, "--ask", q, "32"])
    hm = "HONEST_MISS:1" in out
    ok = rc == 0 and hm and "GEN_TEXT" not in out
    npass += rec("C%d_reject" % (i + 1), ok, "%ds honest=%s gen_absent=%s"
                 % (dt, hm, "GEN_TEXT" not in out))
    total += 1

# ============ 知识库 3 种查询 ============
print("== KB 3 查询 (user文档/wiki中文/无此内容) ==", flush=True)
for i, (q, expect) in enumerate([("BitNet 三值权重", "hit"),
                                 ("文艺复兴", "hit"),
                                 ("M11 步长调度 复现判决", "empty")]):
    out, rc, dt = run([EXE, MINI, "--cap", "kb.search", q, "3"])
    nonempty = '"results": [{' in out
    ok = rc == 0 and (nonempty if expect == "hit" else not nonempty)
    npass += rec("KB%d_%s" % (i + 1, expect), ok, "%ds %s" % (dt, q[:14]))
    total += 1

# ============ 搜索 3 种问题 ============
print("== WEB 3 查询 (英文技术/中文新闻/中英混合) ==", flush=True)
for i, q in enumerate(["BitNet b1.58 inference",
                       "SpaceX 星舰 最新试飞",
                       "Python 3.13 新特性"]):
    out, rc, dt = run([EXE, MINI, "--cap", "web.search", q, "3"])
    ok = rc == 0 and '"results": [{' in out
    npass += rec("WEB%d" % (i + 1), ok, "%ds %s" % (dt, q[:14]))
    total += 1

# ============ 3 插件各 3 遍 ============
print("== 插件 3 x 3 遍 ==", flush=True)
for cap, probe in [("demo.time", "ping"), ("kb.search", "BitNet 三值权重"),
                   ("web.search", "BitNet b1.58 inference")]:
    okall = True
    for i in range(3):
        out, rc, dt = run([EXE, MINI, "--cap", cap, probe, "3"])
        ok = rc == 0 and '"results"' in out or cap == "demo.time" and '"cap"' in out and rc == 0
        okall = okall and ok
        total += 1
        npass += ok
    rec("PLUGIN_%s_x3" % cap.split(".")[0], okall, "3 runs done")

print("\n=== SUMMARY: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
