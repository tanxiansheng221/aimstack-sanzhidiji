# -*- coding: utf-8 -*-
# subj_audit.py — 混料文件 src/split 字段普查（逐科开课的前提验证）
import sys, io, json, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"H:\AimCode\lab\cpu-native\v2_1p5b_full_mix.jsonl"
src_cnt = collections.Counter()
split_cnt = collections.Counter()
sample_by_src = {}
with io.open(SRC, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("src", "?")
        src_cnt[s] += 1
        split_cnt[r.get("split", "?")] += 1
        if s not in sample_by_src:
            sample_by_src[s] = r["q"][:50]
print("=== src 分布（科目标签） ===")
for s, c in src_cnt.most_common():
    print("%8d  %s   例: %s" % (c, s, sample_by_src[s]))
print("=== split 分布 ===")
for s, c in split_cnt.most_common():
    print("%8d  %s" % (c, s))
