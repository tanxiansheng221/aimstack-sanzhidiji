# -*- coding: utf-8 -*-
# build_frontier.py — 条件反射边界扫描用小池混料：十科分层 2600 题 + 打包缓存
# 目的：存在性证明——152M 在 500K token 池 × ~37 次曝光下能否把 tokacc 打到 0.8+
import io, json, random, sys, os
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
per = 2600 // len(by_sub) + 1
for s in sorted(by_sub):
    pool = by_sub[s]
    rng.shuffle(pool)
    rows += pool[:per]
rng.shuffle(rows)
mix = os.path.join(ROOT, "frontier_mix.jsonl")
with io.open(mix, "w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("frontier mix: %d rows" % len(rows))

raw = os.path.join(ROOT, "frontier_cache_packed.npz")
A.build_packed_cache(mix, raw, T=512, seed=42)
comp = os.path.join(ROOT, "frontier_pack_cache.npz")
z = np.load(raw)
np.savez_compressed(comp, **{k: z[k] for k in z.files})
z.close()
print("compressed %.1fMB" % (os.path.getsize(comp) / 1e6))
