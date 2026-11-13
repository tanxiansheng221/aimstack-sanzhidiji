# -*- coding: utf-8 -*-
# build_spec_pool.py — 阶段 2 特训池（80/20，高曝光，排除主 val）
# 曝光倒推：24000 步 × 32 窗/步 ÷ ~1500 窗池 ≈ 512 次/窗（>> 用户底线 50，历史 frontier=870）
# 结构：80% 特训子池（全科均匀）+ 20% 回放子池；主 val 15485 题严格排除（同种子复现原 perm）
import sys, os, io, json, random
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
MIX = ROOT + r"\v2_1p5b_full_mix.jsonl"
SUB = ROOT + r"\v2spec_pool_mix.jsonl"
T = 512

lines = []   # 原始 jsonl 行（与打包时同序）
with io.open(MIX, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["q"].encode("utf-8")[:1600]
        a = r["a"].encode("utf-8")[:1600]
        if q and a and len(q) + len(a) + 3 <= T:
            lines.append(line)
n = len(lines)
print("kept samples:", n, flush=True)

# 复现打包器的 val 划分（RandomState(42).permutation(n)，前 max(200,1%) 为 val）
import numpy as np
rng = np.random.RandomState(42)
perm = rng.permutation(n)
n_val = max(200, int(n * 0.01))
val_set = set(perm[:n_val].tolist())
print("val excluded:", len(val_set), flush=True)

# 80/20 抽样：80% 特训 + 20% 回放，均从非 val 训练区抽（等价于十三科均匀——语料本就混编）
pool_train = [i for i in range(n) if i not in val_set]
pick = random.Random(7).sample(pool_train, 2700)
spec = pick[:2160]        # 80%
replay = pick[2160:]      # 20%
with io.open(SUB, "w", encoding="utf-8") as w:
    for i in spec + replay:
        w.write(lines[i])
print("pool mix written: %d spec + %d replay" % (len(spec), len(replay)), flush=True)

A.build_packed_cache(SUB, ROOT + r"\v2spec_pool_cache.npz", T=T, seed=42)
