# -*- coding: utf-8 -*-
# graph_uniq_diag.py — 按题型拆唯一率，定位碰撞源
import io, sys, json, importlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import os
os.chdir(r"H:\AimCode\lab\cpu-native")
import random
import curriculum_graph_lab as G

seen = {}
dup = {}
rng = random.Random(12345)
N = 20000
for _ in range(N):
    sm = G.gen_sample(rng)
    q = sm["q"]
    fam = sm["fam"]
    seen.setdefault(fam, set())
    if q in seen[fam]:
        dup[fam] = dup.get(fam, 0) + 1
    seen[fam].add(q)

tot_s = 0; tot_n = 0
for fam in sorted(seen):
    s = len(seen[fam])
    # 该 fam 抽样数 ≈ (N - dup 与否不好反推) —— 用 s+dup 估算
    n_est = s + dup.get(fam, 0)
    print("%-10s uniq %6d / ~%d = %.1f%%" % (fam, s, n_est, s / n_est * 100.0))
    tot_s += s; tot_n += n_est
print("TOTAL uniq %.1f%%" % (tot_s / tot_n * 100.0))
