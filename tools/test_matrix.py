# -*- coding: utf-8 -*-
"""aimp 全量回归矩阵: 对话三方式 + 总线三插件 + 分词器往返, 各 3 遍"""
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

def run(args, timeout=900):
    t0 = time.time()
    p = subprocess.run(args, cwd=CWD, capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "ignore")
    err = p.stderr.decode("utf-8", "ignore")
    return out, err, p.returncode, time.time() - t0

def fmt(v):
    return "%.1fs" % v

results = []  # (name, pass, notes, wall)

def record(name, ok, notes, wall):
    results.append((name, ok, notes, wall))
    print("[%s] %-16s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

# ---------- 1. 对话方式 A: --gt 直接对话 ----------
for i in range(3):
    out, err, rc, wall = run([EXE, W, TOK, "--gt", "在吗", "32"])
    m = re.search(r"TIME ([\d.]+)s\s+([\d.]+) tok/s", out)
    g = re.search(r"GEN_TEXT:(.*)", out)
    ok = rc == 0 and m and g and g.group(1).strip() != ""
    note = "rc=%d %s reply=%s" % (rc, fmt(wall), (g.group(1).strip()[:20] if g else "-"))
    record("CHAT_GT_%d" % (i + 1), ok, note, wall)

# ---------- 2. 对话方式 B: --ask 知识库命中 ----------
for i in range(3):
    out, err, rc, wall = run([EXE, W, TOK, "--ask", "Textadept 是什么", "32"])
    h = re.search(r"KB_HITS:(\d+)", out)
    g = re.search(r"GEN_TEXT:(.*)", out)
    ok = rc == 0 and h and int(h.group(1)) > 0 and g
    note = "rc=%d %s hits=%s reply=%s" % (rc, fmt(wall),
          h.group(1) if h else "-", (g.group(1).strip()[:20] if g else "-"))
    record("CHAT_ASK_HIT_%d" % (i + 1), ok, note, wall)

# ---------- 3. 对话方式 C: --ask 诚实拒绝 ----------
for i in range(3):
    out, err, rc, wall = run([EXE, W, TOK, "--ask", "M11 步长调度 复现判决", "32"])
    hm = "HONEST_MISS:1" in out
    g = "GEN_TEXT" in out
    ok = rc == 0 and hm and not g   # 拒答轮必须零生成
    note = "rc=%d %s honest_miss=%s gen_absent=%s" % (rc, fmt(wall), hm, not g)
    record("CHAT_ASK_MISS_%d" % (i + 1), ok, note, wall)

# ---------- 4. 纯 token 生成 (对话底座) ----------
IDS = "128000,1502,25,74662,103054,128009,72803,25,220"
gen_ids_set = set()
for i in range(3):
    out, err, rc, wall = run([EXE, W, "--gen", IDS, "32"])
    m = re.search(r"TIME ([\d.]+)s\s+([\d.]+) tok/s", out)
    ids = re.search(r"GEN_IDS:(.*)", out)
    if ids:
        gen_ids_set.add(ids.group(1).strip())
    ok = rc == 0 and m and ids
    note = "rc=%d %s %s" % (rc, fmt(wall),
          ("%s tok/s" % m.group(2)) if m else "-")
    record("GEN_IDS_%d" % (i + 1), ok, note, wall)
deterministic = len(gen_ids_set) == 1
print("[INFO] GEN_IDS determinism (3 runs identical): %s" % deterministic, flush=True)

# ---------- 5. 总线插件 × 3 ----------
for i in range(3):
    out, err, rc, wall = run([EXE, MINI, "--cap", "demo.time", "ping"])
    ok = rc == 0 and '"cap": "demo.time"' in out
    record("BUS_DEMO_%d" % (i + 1), ok, "rc=%d %s" % (rc, fmt(wall)), wall)

for i in range(3):
    out, err, rc, wall = run([EXE, MINI, "--cap", "kb.search", "BitNet 三值权重", "3"])
    ok = rc == 0 and '"results": [{' in out   # 非空命中
    record("BUS_KB_%d" % (i + 1), ok, "rc=%d %s" % (rc, fmt(wall)), wall)

for i in range(3):
    out, err, rc, wall = run([EXE, MINI, "--cap", "web.search", "BitNet b1.58 inference", "3"])
    ok = rc == 0 and '"results": [{' in out
    record("BUS_WEB_%d" % (i + 1), ok, "rc=%d %s" % (rc, fmt(wall)), wall)

# ---------- 6. 分词器往返 × 3 ----------
for i in range(3):
    out, err, rc, wall = run([EXE, TOK, "--enc", "hello world 你好"])
    m = re.search(r"ENC_IDS:(.*)", out)
    ok = rc == 0 and m
    if ok:
        out2, err2, rc2, wall2 = run([EXE, TOK, "--dec", m.group(1).strip()])
        ok = rc2 == 0 and "DEC_TEXT:hello world 你好" in out2
        note = "enc->dec roundtrip %s" % ("OK" if ok else "MISMATCH")
    else:
        note = "enc fail"
    record("TOK_RT_%d" % (i + 1), ok, note, wall)

# ---------- 汇总 ----------
npass = sum(1 for _, ok, _, _ in results if ok)
print("\n=== SUMMARY: %d/%d PASS ===" % (npass, len(results)), flush=True)
for name, ok, notes, wall in results:
    if not ok:
        print("  FAIL %s: %s" % (name, notes), flush=True)
sys.exit(0 if npass == len(results) else 1)
