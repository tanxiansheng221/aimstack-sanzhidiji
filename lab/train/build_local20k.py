# -*- coding: utf-8 -*-
# build_local20k.py — 本地复现实验：十科分层 20K 题池 + 打包缓存（748K 级 CPU 过夜跑）
import io, json, random, os, sys
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A
import numpy as np

ROOT = r"H:\AimCode\lab\cpu-native"
rng = random.Random(20261031)
by_sub = {}
with io.open(ROOT + r"\curriculum_mix_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") != "train":
            continue
        a = (r.get("cot") or "").strip()
        ans = (r.get("ans") or "").strip()
        a = (a + " || " + ans) if a else ans
        if r.get("q") and a:
            by_sub.setdefault(r["subject"], []).append({"q": r["q"], "a": a})
rows = []
per = 20000 // len(by_sub) + 1
for s in sorted(by_sub):
    pool = by_sub[s]
    rng.shuffle(pool)
    rows += pool[:per]
rng.shuffle(rows)
mix = os.path.join(ROOT, "local20k_mix.jsonl")
with io.open(mix, "w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("local20k mix: %d rows" % len(rows))
A.build_packed_cache(mix, os.path.join(ROOT, "local20k_pack_cache.npz"), T=512, seed=42)
