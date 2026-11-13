# -*- coding: utf-8 -*-
"""build_v1_mix.py — v1 首训混料器：十科(×2) + 情绪 + 创意 + 拒绝 → v1_mix.jsonl
统一 schema: {q, a, src, split}；任务路由前缀让模型显式分科。
每源 1% 抽 val。输出统计 token（字节）量。
"""
import io, json, os, random, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CN = r"H:\AimCode\lab\cpu-native"
OUT = os.path.join(CN, "v1_mix.jsonl")
random.seed(42)

rows = []

# 1) 十科课程（q/cot/ans，已有 split）×2（课程是主粮，两遍）
for fn in os.listdir(CN):
    if fn.startswith("curriculum_") and fn.endswith(".jsonl"):
        with io.open(os.path.join(CN, fn), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                a = (r.get("cot") or "").strip()
                ans = (r.get("ans") or "").strip()
                a = (a + " || " + ans) if a else ans
                if not r.get("q") or not a:
                    continue
                rows.append({"q": r["q"], "a": a, "src": fn.replace(".jsonl", ""),
                             "split": r.get("split", "train")})
print("curriculum rows:", len(rows))

n0 = len(rows)
n_cur = n0

# 2) 情绪 emp2（user_situation→resp，带情绪标签路由）
emp = r"H:\AimCode\lab\cpu-native\emp2_corpus\emp2_corpus.jsonl"
with io.open(emp, encoding="utf-8", errors="replace") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("user_situation") or not r.get("resp"):
            continue
        rows.append({"q": "[情绪:%s] %s" % (r.get("emotion", "?"), r["user_situation"]),
                     "a": r["resp"], "src": "emp2_" + r.get("source", "distill"),
                     "split": r.get("split", "train")})
print("after emp2:", len(rows) - n0, "added")
n0 = len(rows)

# 3) 创意（q→ans）
cre = r"H:\AimCode\lab\cpu-native\creative_corpus\creative_v0.jsonl"
with io.open(cre, encoding="utf-8", errors="replace") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("q") or not r.get("ans"):
            continue
        rows.append({"q": "[创意:%s] %s" % (r.get("task", "story"), r["q"]),
                     "a": r["ans"], "src": "creative_" + r.get("teacher", "?"),
                     "split": r.get("split", "train")})
print("after creative:", len(rows) - n0, "added")
n0 = len(rows)

# 4) 拒绝（user_situation→resp）
ref = r"H:\AimCode\lab\cpu-native\refusal_corpus\refusal_v0.jsonl"
with io.open(ref, encoding="utf-8", errors="replace") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("user_situation") or not r.get("resp"):
            continue
        rows.append({"q": "[安全] %s" % r["user_situation"],
                     "a": r["resp"], "src": "refusal_" + r.get("source", "distill"),
                     "split": r.get("split", "train")})
print("after refusal:", len(rows) - n0, "added")

# 十科 ×2（train 部分追加一遍）
cur_train = [r for r in rows[:n_cur] if r["split"] == "train"]
rows.extend(cur_train)
print("curriculum x2: +%d" % len(cur_train))

# 每源 1% 抽 val（仅对没有 split 标注的行）
by_src = {}
for r in rows:
    by_src.setdefault(r["src"], []).append(r)
for src, rs in by_src.items():
    if all(r["split"] == "train" for r in rs):
        k = max(1, int(len(rs) * 0.01))
        for r in random.sample(rs, k):
            r["split"] = "val"

random.shuffle(rows)
qbytes = sum(len(r["q"].encode("utf-8")) for r in rows)
abytes = sum(len(r["a"].encode("utf-8")) for r in rows)
val_n = sum(1 for r in rows if r["split"] == "val")
with io.open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("TOTAL rows=%d val=%d qbytes=%.1fMB abytes=%.1fMB tokens~=%.1fMB" %
      (len(rows), val_n, qbytes / 1e6, abytes / 1e6, (qbytes + abytes) / 1e6))
