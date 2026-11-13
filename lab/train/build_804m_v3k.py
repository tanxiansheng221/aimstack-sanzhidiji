# -*- coding: utf-8 -*-
# build_804m_v3k.py — Kaggle 804M 多 epoch 对照实验混料（双臂）
# 假设：多 epoch 重复曝光能把 val_exact 从 0 拉起来（v7 基线=单遍欠采样 0.24 次/题 → 全程 0）
# 设计：3000 步预算 ≈ 23M tokens 消耗。
#   v3k2: 十科 5.3% 分层子集 ×2 副本 + 10% 回放 → 每题曝光 2 次
#   v3k3: 十科 3.55% ×3 副本 + 10% 回放 → 每题曝光 3 次
#   val:  每科 273 题分层 holdout（不进训练），存独立数组进 npz
import io, json, random, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
SEED = 20261031
T_LIMIT = 512

ARMS = {
    "v3k2": dict(frac=0.053, copies=2),
    "v3k3": dict(frac=0.0355, copies=3),
}
VAL_PER_SUBJ = 273

# ---- 载入十科（train split），fit 过滤 ----
by_sub = {}
with io.open(ROOT + r"\curriculum_mix_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") != "train":
            continue
        a = (r.get("cot") or "").strip()
        ans = (r.get("ans") or "").strip()
        a = (a + " || " + ans) if a else ans
        if not r.get("q") or not a:
            continue
        q = r["q"].encode("utf-8")[:1600].decode("utf-8", "ignore")
        a = a.encode("utf-8")[:1600].decode("utf-8", "ignore")
        if len(q.encode()) + len(a.encode()) + 3 > T_LIMIT:
            continue
        by_sub.setdefault(r["subject"], []).append({"q": q, "a": a})

# ---- holdout val（每科 273，不进任何训练臂）----
rng = random.Random(SEED)
val_rows = []
for s in sorted(by_sub):
    pool = by_sub[s]
    rng.shuffle(pool)
    val_rows += pool[:VAL_PER_SUBJ]
    by_sub[s] = pool[VAL_PER_SUBJ:]
print("val holdout: %d rows" % len(val_rows), flush=True)


def tok(q, a):
    return len(q.encode("utf-8")) + len(a.encode("utf-8")) + 3


# ---- 回放池（fit 过滤）----
def load_lib(path, src, qf, af):
    pool = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            q, a = qf(r), af(r)
            if q and a and tok(q, a) <= T_LIMIT:
                pool.append({"q": q, "a": a, "src": "replay_" + src})
    rng.shuffle(pool)
    return pool


libs = [
    load_lib(ROOT + r"\emp2_corpus\emp2_corpus.jsonl", "emp2",
             lambda r: "[情绪:%s] %s" % (r.get("emotion", "?"), r["user_situation"]), lambda r: r.get("resp", "")),
    load_lib(ROOT + r"\creative_corpus\creative_v0.jsonl", "creative",
             lambda r: "[创意:%s] %s" % (r.get("task", "story"), r["q"]), lambda r: r.get("ans", "")),
    load_lib(ROOT + r"\refusal_corpus\refusal_v0.jsonl", "refusal",
             lambda r: "[安全] %s" % r["user_situation"], lambda r: r.get("resp", "")),
]
print("replay pools: %s" % [len(p) for p in libs], flush=True)


def build_cache_npz(rows, val_rows, cache_path):
    def enc(rs):
        starts, qlens, alens, buf = [], [], [], bytearray()
        for r in rs:
            q = r["q"].encode("utf-8")
            a = r["a"].encode("utf-8")
            if not q or not a:
                continue
            starts.append(len(buf))
            qlens.append(len(q))
            alens.append(len(a))
            buf += q + b"\n" + a
        return (np.frombuffer(bytes(buf), dtype=np.uint8),
                np.array(starts, dtype=np.int64),
                np.array(qlens, dtype=np.int32),
                np.array(alens, dtype=np.int32))
    data, starts, qlens, alens = enc(rows)
    vdata, vstarts, vqlens, valens = enc(val_rows)
    np.savez(cache_path, data=data, starts=starts, qlens=qlens, alens=alens,
             vdata=vdata, vstarts=vstarts, vqlens=vqlens, valens=valens)
    print("cache: train %d samples %.1fMB | val %d samples %.1fMB -> %s"
          % (len(starts), len(data) / 1e6, len(vstarts), len(vdata) / 1e6, cache_path), flush=True)


for arm, cfg in ARMS.items():
    rows = []
    cur_tok = 0
    for s in sorted(by_sub):
        pool = by_sub[s]
        rng2 = random.Random(SEED + 1)  # 独立于 holdout 洗牌，保证两臂子集嵌套
        pool2 = pool[:]
        rng2.shuffle(pool2)
        keep = pool2[: int(len(pool2) * cfg["frac"])]
        for r in keep:
            rr = {"q": r["q"], "a": r["a"], "src": "cur_" + s}
            for _ in range(cfg["copies"]):
                rows.append(rr)
            cur_tok += tok(r["q"], r["a"])
    # 回放 = 课程 token 量的 10%（三库均分）
    rep_budget = cur_tok * cfg["copies"] / 9.0
    rep_tok = 0
    for pool in libs:
        used = 0
        for r in pool:
            t = tok(r["q"], r["a"])
            if used + t > rep_budget / 3:
                break
            rows.append(dict(r))
            used += t
            rep_tok += t
    random.Random(SEED + arm.__hash__()).shuffle(rows)
    mix_path = "%s\\%s_804m_mix.jsonl" % (ROOT, arm)
    with io.open(mix_path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cache_path = "%s\\%s_804m_cache.npz" % (ROOT, arm)
    build_cache_npz(rows, val_rows, cache_path)
    print("%s: %d rows, curriculum %.1fM x%d + replay %.1fM = file %.1fM tok | 3000步曝光=%.2f 次/题"
          % (arm, len(rows), cur_tok / 1e6, cfg["copies"], rep_tok / 1e6,
             (cur_tok * cfg["copies"] + rep_tok) / 1e6,
             23e6 / (cur_tok * cfg["copies"] + rep_tok) * cfg["copies"]), flush=True)
