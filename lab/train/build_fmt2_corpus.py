# -*- coding: utf-8 -*-
# build_fmt2_corpus.py — fmt2 三期课程语料 + 特训包（各期 = 五科 fmt2 行 + 50% 回放）
# 产物: fmt2_T1/T2/T3_mix.jsonl → fmt2_T1/T2/T3_pack.npz
import io, json, random, zlib, time, os, sys
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import fmt2_generators as G
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
FMT5 = {"math": "v2_math", "mod_arith": "v2_mod_arith", "plan": "v2_plan",
        "progexec": "v2_progexec", "seqind": "v2_seqind"}
TIER_N = {"T1": 8000, "T2": 10000, "T3": 12000}   # 每科每期条数

t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

# 回放池：v2fmt 全量混料按 src 分组索引（一次性建索引）
log("index replay pool (v2fmt mix)...")
replay_idx = {}
with io.open(ROOT + r"\v2fmt_1p5b_full_mix.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("src", "")
        if s not in FMT5.values():          # 回放只用主盘八科+replay，防旧格式五科混入
            replay_idx.setdefault(s, []).append(i)
log("replay srcs: %d" % len(replay_idx))

def emit(subject, tier, n, seed):
    rng = random.Random((zlib.crc32(subject.encode()) * 31 + zlib.crc32(tier.encode()) + seed) % (2**31))
    rows = []
    while len(rows) < n:
        sm = G.gen_sample(rng, subject, tier)
        if sm is None:
            continue
        rows.append({"q": sm["q"], "a": sm["cot"] + " || " + sm["ans"],
                     "src": "fmt2_" + subject, "split": "train"})
    return rows

for tier in ("T1", "T2", "T3"):
    mix_path = os.path.join(ROOT, "fmt2_%s_mix.jsonl" % tier)
    rows = []
    for subj, src in FMT5.items():
        r = emit(subj, tier, TIER_N[tier], 20261104)
        rows.extend(r)
        log("%s %s: %d rows (%.0fs)" % (tier, subj, len(r), time.time() - t0))
    # 50% 回放：从 replay 池等比抽样
    rng = random.Random(20261104 + zlib.crc32(tier.encode()))
    n_replay = len(rows)
    all_lines = []
    per = n_replay // max(1, len(replay_idx))
    for s, idxs in replay_idx.items():
        take = rng.sample(idxs, min(per + 1, len(idxs)))
        all_lines.extend(idxs)
    log("%s replay rows target: %d (per src ~%d)" % (tier, n_replay, per))
    # 流式抽回放行
    need = set()
    for s, idxs in replay_idx.items():
        per_s = n_replay // len(replay_idx)
        take = rng.sample(idxs, min(per_s, len(idxs)))
        need.update(take)
    got = 0
    with io.open(ROOT + r"\v2fmt_1p5b_full_mix.jsonl", encoding="utf-8") as f, \
         io.open(mix_path, "w", encoding="utf-8", newline="") as w:
        for i, line in enumerate(f):
            if i in need:
                w.write(line)
                got += 1
    with io.open(mix_path, "a", encoding="utf-8", newline="") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("%s mix: %d replay + %d fmt2 rows, %.1fMB" % (
        tier, got, len(rows), os.path.getsize(mix_path) / 1e6))
    A.build_packed_cache(mix_path, os.path.join(ROOT, "fmt2_%s_pack.npz" % tier), T=512, seed=42)
    z = np = None
    import numpy as np_
    z = np_.load(os.path.join(ROOT, "fmt2_%s_pack.npz" % tier))
    log("%s pack: %d windows" % (tier, len(z["win_tok"])))

log("ALL TIERS DONE %.0fs" % (time.time() - t0))
