# -*- coding: utf-8 -*-
# build_1p5b_v2_full_mix.py — 1.5B v2 全量混料：十科池全量(每唯一题一遍) + 三库全量，零采样
# 输出: v2_1p5b_full_mix.jsonl {q,a,src,split}
import io, json, random, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
OUT = ROOT + r"\v2_1p5b_full_mix.jsonl"

rng = random.Random(20261031)
rows = []
tok_cur = 0

# ---------- 1) 十科全量（train 全进，val 留包外做独立评测） ----------
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
        t = len(r["q"].encode("utf-8")) + len(a.encode("utf-8")) + 3
        rows.append({"q": r["q"], "a": a, "src": "v2_" + r["subject"], "split": "train"})
        tok_cur += t
        by_sub[r["subject"]] = by_sub.get(r["subject"], 0) + 1
for s in sorted(by_sub):
    print("%-10s %7d rows" % (s, by_sub[s]), flush=True)
print("curriculum FULL: %.1fM tokens (%d rows)" % (tok_cur / 1e6, len(rows)), flush=True)

# ---------- 2) 三库全量（每行都进，带路由前缀） ----------
def tok(r):
    return len(r["q"].encode("utf-8")) + len(r["a"].encode("utf-8")) + 3

libs = [
    (ROOT + r"\emp2_corpus\emp2_corpus.jsonl", "emp2", lambda r: "[情绪:%s] %s" % (r.get("emotion", "?"), r["user_situation"]), lambda r: r.get("resp", "")),
    (ROOT + r"\creative_corpus\creative_v0.jsonl", "creative", lambda r: "[创意:%s] %s" % (r.get("task", "story"), r["q"]), lambda r: r.get("ans", "")),
    (ROOT + r"\refusal_corpus\refusal_v0.jsonl", "refusal", lambda r: "[安全] %s" % r["user_situation"], lambda r: r.get("resp", "")),
]
tok_rep = 0
for path, src, qf, af in libs:
    n = 0
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = qf(r)
            a = af(r)
            if q and a:
                rows.append({"q": q, "a": a, "src": "replay_" + src, "split": "train"})
                tok_rep += tok(rows[-1])
                n += 1
    print("replay_%s FULL: %d rows" % (src, n), flush=True)
print("replay total: %.1fM tokens" % (tok_rep / 1e6), flush=True)

rng.shuffle(rows)
data = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
io.open(OUT, "w", encoding="utf-8", newline="").write(data)
total = tok_cur + tok_rep
print("TOTAL FULL: %.1fM tokens, %d rows -> %s" % (total / 1e6, len(rows), OUT))
print("steps @14K/step: %d" % round(total / 14000))
