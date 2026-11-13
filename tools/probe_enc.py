# -*- coding: utf-8 -*-
"""解剖 --enc 卡死/MISMATCH: 精确复刻矩阵调用 + 解码可疑 id"""
import subprocess, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
TOK = r"H:\AimCode\aimp\data\m2b_llama3_head.gguf"
CWD = r"H:\AimCode\aimp"

def enc(text, timeout=30):
    t0 = time.time()
    p = subprocess.run([EXE, TOK, "--enc", text], cwd=CWD,
                       capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "ignore")
    line = [l for l in out.splitlines() if l.startswith("ENC_IDS")]
    return (line[0] if line else "NO_ENC_IDS"), p.returncode, time.time() - t0

def dec(ids, timeout=30):
    t0 = time.time()
    p = subprocess.run([EXE, TOK, "--dec", ids], cwd=CWD,
                       capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "ignore")
    line = [l for l in out.splitlines() if l.startswith("DEC_TEXT")]
    return (line[0] if line else "NO_DEC_TEXT"), p.returncode, time.time() - t0

print("== 1. 矩阵原输入 (python unicode args, 精确复刻) ==")
for trial in range(3):
    ids, rc, dt = enc("hello world 你好")
    print("enc %d: rc=%d %.1fs %s" % (trial, rc, dt, ids))

print("== 2. 解码可疑单 id 15339 ==")
txt, rc, dt = dec("15339")
print("dec 15339: %s (%.1fs)" % (txt, dt))

print("== 3. 分步: 各自编码再解码 ==")
for t in ["hello world", "你好", "hello", "world"]:
    ids, rc, dt = enc(t)
    print("enc %-12r %.1fs %s" % (t, dt, ids))
    raw = ids.replace("ENC_IDS: ", "")
    if raw:
        txt, rc2, dt2 = dec(raw)
        print("  dec back: %s (%.1fs)" % (txt, dt2))
