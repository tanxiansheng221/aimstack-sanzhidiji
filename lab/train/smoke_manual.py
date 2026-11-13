# -*- coding: utf-8 -*-
"""smoke_manual.py — MANUAL(AIM_AUTOGRAD=0) 模式端到端冒烟：小缓存、20 步、双模式对照 tok/s"""
import os
import subprocess
import sys
import time

import numpy as np

z = np.load(r"H:\AimCode\lab\cpu-native\frontier_pack_cache.npz")
print("cache files:", z.files)
print("win_tok:", z["win_tok"].shape)
T = int(z["win_tok"].shape[1])
z.close()

env = dict(os.environ)
env.update({
    "AIM_CACHE": r"H:\AimCode\lab\cpu-native\frontier_pack_cache.npz",
    "AIM_D": "64", "AIM_L": "2", "AIM_H1": "128",
    "AIM_T": str(T), "AIM_B": "8", "AIM_STEPS": "100",
    "AIM_LR": "1e-3", "AIM_CHUNK": "32", "AIM_EVAL": "100", "AIM_SAVE": "100",
    "AIM_OUT": r"H:\AimCode\lab\cpu-native\train_out_smoke",
    "AIM_WARM": "5",
})
import re
for mode in ["0", "1"]:
    env["AIM_AUTOGRAD"] = mode
    env["AIM_TAG"] = "smoke_manual" if mode == "0" else "smoke_auto"
    t0 = time.time()
    r = subprocess.run([sys.executable, r"H:\AimCode\lab\cpu-native\aim_train_real.py"],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    losses = re.findall(r"step (\d+) loss ([\d.]+) lr [\d.e-]+ tok/s (\d+)", r.stdout)
    print("MODE AIM_AUTOGRAD=%s exit %d wall %.1fs" % (mode, r.returncode, time.time() - t0))
    for s, l, tk in losses:
        print("  step %s loss %s tok/s %s" % (s, l, tk))
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
