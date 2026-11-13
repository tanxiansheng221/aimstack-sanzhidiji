# -*- coding: utf-8 -*-
"""补跑: md 查回 (题干修正) + web.ingest 活网入库"""
import re, subprocess, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
MINI = r"H:\AimCode\aimp\build\mini_llama.gguf"
CWD = r"H:\AimCode\aimp"

def run(args, timeout=600):
    t0 = time.time()
    p = subprocess.run(args, cwd=CWD, capture_output=True, timeout=timeout)
    return p.stdout.decode("utf-8", "ignore"), p.returncode, time.time() - t0

total = npass = 0
def rec(name, ok, notes):
    global total, npass
    total += 1
    npass += bool(ok)
    print("[%s] %-16s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

# 1. md 查回: 单标记词 (题干修正 — 之前 "AIMPKBTEST2026 md" 里 md 不在正文)
out, rc, dt = run([EXE, MINI, "--cap", "kb.search", "AIMPKBTEST2026", "3"])
hits = out.count('"id": "kb-')
ok = rc == 0 and '"results": [{' in out
rec("SEARCHBACK_MD", ok, "%ds marker_hits=%d" % (dt, hits))

# 2. web.ingest 活网
out, rc, dt = run([EXE, MINI, "--cap", "web.ingest", "BitNet b1.58 推理", "2"], 600)
cw = re.search(r'"crawled":\s*(\d+)', out)
ad = re.search(r'"fts_indexed":\s*(\d+)', out)
det = re.search(r'"details":\s*(\[.*?\])', out)
ok = rc == 0 and cw and int(cw.group(1)) > 0 and ad and int(ad.group(1)) > 0
rec("WEB_INGEST", ok, "%ds crawled=%s fts=%s %s" % (
    dt, cw.group(1) if cw else "-", ad.group(1) if ad else "-",
    det.group(1)[:200] if det else "-"))

print("=== SUMMARY: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
