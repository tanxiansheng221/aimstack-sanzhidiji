# -*- coding: utf-8 -*-
"""TOK_RT 补测 3 遍 (空格分隔 ids = 原矩阵死循环格式)"""
import subprocess, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
TOK = r"H:\AimCode\aimp\data\m2b_llama3_head.gguf"
CWD = r"H:\AimCode\aimp"

npass = 0
for i in range(3):
    t0 = time.time()
    p = subprocess.run([EXE, TOK, "--enc", "hello world 你好"], cwd=CWD,
                       capture_output=True, timeout=60)
    ids = [l for l in p.stdout.decode("utf-8", "ignore").splitlines()
           if l.startswith("ENC_IDS")][0].replace("ENC_IDS: ", "")
    p2 = subprocess.run([EXE, TOK, "--dec", ids], cwd=CWD,
                        capture_output=True, timeout=60)   # 空格分隔原样传回
    out2 = p2.stdout.decode("utf-8", "ignore")
    ok = ("DEC_TEXT:hello world 你好" in out2)
    npass += ok
    print("[%s] TOK_RT_%d %s ids=[%s]" % ("PASS" if ok else "FAIL", i + 1,
          "%.1fs" % (time.time() - t0), ids), flush=True)
print("TOK_RT: %d/3 PASS" % npass, flush=True)
sys.exit(0 if npass == 3 else 1)
