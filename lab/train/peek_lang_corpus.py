# -*- coding: utf-8 -*-
# peek_lang_corpus.py - 语言包原料盘点：各语料行格式 + 行数
import io, sys, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))

def peek(path, n=2, maxlen=220):
    print("==== %s ====" % path)
    if not os.path.exists(path):
        print("  MISSING")
        return
    cnt = 0
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < n:
                print("  ", line.strip()[:maxlen])
            cnt += 1
    print("  lines:", cnt)

peek(os.path.join(BASE, "emp2_corpus", "emp2_corpus.jsonl"))
peek(os.path.join(BASE, "creative_corpus", "creative_v0.jsonl"))
# 拒绝语料在混料里，抽查 v2fmt 全量混料的 src 分布（抽 2 万行统计）
import collections
srcs = collections.Counter()
samples = {}
with io.open(os.path.join(BASE, "v2fmt_1p5b_full_mix.jsonl"), encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = (r.get("src", "?"), r.get("split", "?"))
        srcs[key] += 1
        if key not in samples:
            samples[key] = line.strip()[:220]
for k, v in srcs.most_common(20):
    print(k, v)
print("---- refusal/creative/emp2 samples ----")
for k, s in samples.items():
    if any(w in k[0] for w in ("refus", "creativ", "emp2")):
        print(k, "=>", s)
