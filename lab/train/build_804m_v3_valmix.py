# -*- coding: utf-8 -*-
# build_804m_v3_valmix.py — 804M v3 配方验证混料：十科 20% 分层子集 × 2 epoch（文件内复制）+ 回放 10%
# 假设检验：多 epoch 干净科能否把 val_exact 从 0 拉起来（v4/v5 证据）
import io, json, random, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
OUT = ROOT + r"\v3_804m_valmix.jsonl"
SUBSET = 0.20      # 十科每科抽 20%
EPOCHS = 2         # 池内复制 2 份 = 期望曝光 ×2
REPLAY_RATIO = 0.10

rng = random.Random(20261031)
rows = []

by_sub = {}
with io.open(ROOT + r"\curriculum_mix_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") != "train":
            continue
        by_sub.setdefault(r["subject"], []).append(r)

tok_cur = 0
for s in sorted(by_sub):
    pool = by_sub[s]
    rng.shuffle(pool)
    keep = pool[: int(len(pool) * SUBSET)]
    for r in keep:
        a = (r.get("cot") or "").strip()
        ans = (r.get("ans") or "").strip()
        a = (a + " || " + ans) if a else ans
        if not r.get("q") or not a:
            continue
        t = len(r["q"].encode("utf-8")) + len(a.encode("utf-8")) + 3
        for _ in range(EPOCHS):
            rows.append({"q": r["q"], "a": a, "src": "v3_" + s, "split": "train"})
        tok_cur += t
    print("%-10s subset %d rows x%d" % (s, len(keep), EPOCHS), flush=True)
print("curriculum: %.1fM tokens/epoch x%d (%d rows)" % (tok_cur / 1e6, EPOCHS, len(rows)), flush=True)

# 回放 10%
def tok(r):
    return len(r["q"].encode("utf-8")) + len(r["a"].encode("utf-8")) + 3

replay_budget = tok_cur * EPOCHS / (1 - REPLAY_RATIO) * REPLAY_RATIO
per_lib = replay_budget / 3
libs = [
    (ROOT + r"\emp2_corpus\emp2_corpus.jsonl", "emp2", lambda r: "[情绪:%s] %s" % (r.get("emotion", "?"), r["user_situation"]), lambda r: r.get("resp", "")),
    (ROOT + r"\creative_corpus\creative_v0.jsonl", "creative", lambda r: "[创意:%s] %s" % (r.get("task", "story"), r["q"]), lambda r: r.get("ans", "")),
    (ROOT + r"\refusal_corpus\refusal_v0.jsonl", "refusal", lambda r: "[安全] %s" % r["user_situation"], lambda r: r.get("resp", "")),
]
tok_rep = 0
for path, src, qf, af in libs:
    pool = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = qf(r)
            a = af(r)
            if q and a:
                pool.append({"q": q, "a": a, "src": "replay_" + src, "split": "train"})
    rng.shuffle(pool)
    used = 0
    for r in pool:
        t = tok(r)
        if used + t > per_lib:
            break
        rows.append(r)
        used += t
        tok_rep += t
    print("replay_%s %.1fM (%d)" % (src, used / 1e6, len([x for x in rows if x["src"] == "replay_" + src])), flush=True)

rng.shuffle(rows)
data = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
io.open(OUT, "w", encoding="utf-8", newline="").write(data)
total = tok_cur * EPOCHS + tok_rep
print("TOTAL: %.1fM tokens, %d rows -> %s" % (total / 1e6, len(rows), OUT))
print("steps @9.8K/step(804M cfg): %d" % round(total / 9800))
