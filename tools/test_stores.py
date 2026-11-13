# -*- coding: utf-8 -*-
"""双库架构验收: 写入隔离 / 双库合并检索 / 整包换库"""
import os, re, subprocess, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
MINI = r"H:\AimCode\aimp\build\mini_llama.gguf"
CWD = r"H:\AimCode\aimp"
ALT_DYN = r"H:\AimCode\aimp\testdocs\kb_alt_dyn"

def run(args, env_extra=None, timeout=300):
    t0 = time.time()
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(args, cwd=CWD, capture_output=True, timeout=timeout, env=env)
    return p.stdout.decode("utf-8", "ignore"), p.returncode, time.time() - t0

total = npass = 0
def rec(name, ok, notes):
    global total, npass
    total += 1
    npass += bool(ok)
    print("[%s] %-18s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

def stores(out):
    return sorted(set(re.findall(r'"store": "(\w+)"', out)))

# 1. 动态库写入 (store=dyn)
doc = os.path.join(CWD, "testdocs", "test_md.md")
out, rc, dt = run([EXE, MINI, "--cap", "kb.add", doc])
m = re.search(r'"added":\s*(\d+)', out)
s = '"store": "dyn"' in out
rec("ADD_TO_DYN", rc == 0 and m and int(m.group(1)) == 0 and s,
    "re-add dedup: added=%s store_tag=%s %ds" % (m.group(1) if m else "-", s, dt))

# 2. 双库合并检索: wiki 题应标 base, user/crawl 题应标 dyn
out, rc, dt = run([EXE, MINI, "--cap", "kb.search", "文艺复兴", "3"])
rec("MERGE_BASE", rc == 0 and '"results": [{' in out and stores(out) == ["base"],
    "%ds stores=%s" % (dt, stores(out)))

out, rc, dt = run([EXE, MINI, "--cap", "kb.search", "BitNet 三值权重", "3"])
rec("MERGE_DYN", rc == 0 and '"results": [{' in out and stores(out) == ["dyn"],
    "%ds stores=%s" % (dt, stores(out)))

out, rc, dt = run([EXE, MINI, "--cap", "kb.search", "BitNet CPU inference", "4"])
rec("MERGE_DYN_CRAWL", rc == 0 and '"results": [{' in out and stores(out) == ["dyn"],
    "%ds stores=%s (crawl 卡归 dyn)" % (dt, stores(out)))

# 3. 整包换库: 备用动态库 (空) -> 里面加独家文档 -> 切 env 检索
os.makedirs(ALT_DYN, exist_ok=True)
alt_doc = os.path.join(CWD, "testdocs", "test_swap.md")
with open(alt_doc, "w", encoding="utf-8") as f:
    f.write("# AIMPDYNSWAP3 独家\n\n备用动态库专属内容: swap verification doc.\n")
out, rc, dt = run([EXE, MINI, "--cap", "kb.add", alt_doc],
                  {"AIM_KB_DYN_ROOT": ALT_DYN})
m = re.search(r'"added":\s*(\d+)', out)
rec("ADD_TO_ALT", rc == 0 and m and int(m.group(1)) > 0,
    "%ds added=%s (alt dyn)" % (dt, m.group(1) if m else "-"))

out, rc, dt = run([EXE, MINI, "--cap", "kb.search", "AIMPDYNSWAP3", "3"],
                  {"AIM_KB_DYN_ROOT": ALT_DYN})
rec("SWAP_ON", rc == 0 and '"results": [{' in out, "%ds 换库后可检索" % dt)

out, rc, dt = run([EXE, MINI, "--cap", "kb.search", "AIMPDYNSWAP3", "3"])
ok = rc == 0 and '"results": []' in out
rec("SWAP_OFF", ok, "%ds 换回默认库 -> 独家内容消失 (库隔离实证)" % dt)

print("\n=== SUMMARY: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
