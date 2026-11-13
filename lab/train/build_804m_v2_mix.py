# -*- coding: utf-8 -*-
# build_804m_v2_mix.py — 804M v2 续训混料：十科 v2 新题 91% + 三库回放 9%（token 预算抽样）
# 输出: v2_804m_mix.jsonl {q,a,src,split}
import io, json, random, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
OUT = ROOT + r"\v2_804m_mix.jsonl"
TARGET_TOTAL = 120_000_000   # 总 token（字节级分词 1 token=1 字节）预算
CUR_RATIO = 0.91

rng = random.Random(20261030)
rows = []

# ---------- 1) 十科 v2：分层抽样到 CUR 预算 ----------
by_sub = {}
with io.open(ROOT + r"\curriculum_mix_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") != "train":
            continue  # val 留在包外做独立评测
        by_sub.setdefault(r["subject"], []).append(r)

subs = sorted(by_sub)
per_budget = {}
# 先按科均匀分配 token 预算（graph 存量小自动吃满）
cur_total_budget = TARGET_TOTAL * CUR_RATIO
budget_each = cur_total_budget / len(subs)
for s in subs:
    per_budget[s] = budget_each

tok_cur = 0
for s in subs:
    pool = by_sub[s]
    rng.shuffle(pool)
    used = 0
    for r in pool:
        a = (r.get("cot") or "").strip()
        ans = (r.get("ans") or "").strip()
        a = (a + " || " + ans) if a else ans
        if not r.get("q") or not a:
            continue
        t = len(r["q"].encode("utf-8")) + len(a.encode("utf-8")) + 3
        if used + t > per_budget[s] or tok_cur + t > cur_total_budget:
            break
        rows.append({"q": r["q"], "a": a, "src": "v2_" + s, "split": "train"})
        used += t
        tok_cur += t
    print("%-10s used %.1fM tokens, kept %d" % (s, used / 1e6, len([x for x in rows if x["src"] == "v2_" + s])), flush=True)

print("curriculum total: %.1fM tokens (%d rows)" % (tok_cur / 1e6, len(rows)), flush=True)

# ---------- 2) 三库回放 9%（带路由前缀，分层） ----------
def tok(r):
    return len(r["q"].encode("utf-8")) + len(r["a"].encode("utf-8")) + 3

replay_budget = TARGET_TOTAL - tok_cur
per_lib = replay_budget / 3
libs = [
    (ROOT + r"\emp2_corpus\emp2_corpus.jsonl", "emp2", lambda r: "[情绪:%s] %s" % (r.get("emotion", "?"), r["user_situation"]), lambda r: r.get("resp", "")),
    (ROOT + r"\creative_corpus\creative_v0.jsonl", "creative", lambda r: "[创意:%s] %s" % (r.get("task", "story"), r["q"]), lambda r: r.get("ans", "")),
    (ROOT + r"\refusal_corpus\refusal_v0.jsonl", "refusal", lambda r: "[安全] %s" % r["user_situation"], lambda r: r.get("resp", "")),
]
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
    print("replay_%s used %.1fM (%d)" % (src, used / 1e6, len([x for x in rows if x["src"] == "replay_" + src])), flush=True)

rng.shuffle(rows)
data = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
io.open(OUT, "w", encoding="utf-8", newline="").write(data)
total = sum(tok(r) for r in rows)
print("TOTAL: %.1fM tokens, %d rows -> %s" % (total / 1e6, len(rows), OUT))
