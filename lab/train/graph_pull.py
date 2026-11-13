# -*- coding: utf-8 -*-
# graph_pull.py — 图论科大 n 导出 + 唯一率统计
import io, sys, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")

import os
os.chdir(r"H:\AimCode\lab\cpu-native")

import curriculum_graph_lab as G

t0 = time.time()
res = G.self_test()
print("SELF_TEST:", json.dumps(res, ensure_ascii=False), flush=True)
if not res["ALL_PASS"]:
    sys.exit(2)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
exp = G.export(n_train=N, n_val=5000)
print("EXPORT:", exp, "(%.0fs)" % (time.time() - t0), flush=True)

# 唯一率：按 q 文本去重
qs = set()
tot = 0
famc = {}
with io.open("curriculum_graph_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qs.add(r["q"])
        tot += 1
        famc[r["qtype"]] = famc.get(r["qtype"], 0) + 1
print("UNIQUENESS: %d/%d = %.1f%%" % (len(qs), tot, len(qs) / tot * 100.0))
print("FAM_MIX:", json.dumps(famc, sort_keys=True))
print("=== GRAPH PULL DONE ===")
