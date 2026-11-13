# -*- coding: utf-8 -*-
# fix_pool_exposure.py — 超标子池回抽：所有科特训池压到 ≤1500 窗，保证段内曝光 ≥50 次
import sys, os, io, json, re
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import numpy as np
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
MIX = ROOT + r"\v2_1p5b_full_mix.jsonl"
T = 512
TARGET_WIN = 1500

meta = json.load(open(ROOT + r"\exam_meta.json", encoding="utf-8"))
names = meta["subjects"]

# 重放一遍流，按科收集行（与首遍同过滤、同顺序）
need = {}
for s in names:
    short = s.replace("v2_", "").replace("replay_", "")
    z = np.load(ROOT + r"\pool_%s.npz" % short)
    w = len(z["win_tok"]); z.close()
    if w > TARGET_WIN:
        need[s] = (short, w, int(2800 * TARGET_WIN / w))  # 按窗口率回抽
print("rebuild:", {s: v[2] for s, v in need.items()}, flush=True)

rows = {s: [] for s in need}
with io.open(MIX, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["q"].encode("utf-8")[:1600]
        a = r["a"].encode("utf-8")[:1600]
        if q and a and len(q) + len(a) + 3 <= T:
            s = r["src"]
            if s in need and len(rows[s]) < 400000:
                rows[s].append(line)

er = np.random.RandomState(2026)
for s, (short, old_w, new_n) in need.items():
    idx = er.choice(len(rows[s]), size=new_n, replace=False)
    sub = ROOT + r"\_fix_%s.jsonl" % short
    with io.open(sub, "w", encoding="utf-8") as w:
        for i in idx:
            w.write(rows[s][i])
    A.build_packed_cache(sub, ROOT + r"\pool_%s.npz" % short, T=T, seed=42)
    z = np.load(ROOT + r"\pool_%s.npz" % short)
    w = len(z["win_tok"]); z.close()
    os.remove(sub)
    meta["meta"][s]["focus_windows"] = int(w)
    print("%-18s rebuilt %d -> %d windows" % (s, old_w, w), flush=True)

json.dump(meta, open(ROOT + r"\exam_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("FIX DONE", flush=True)
