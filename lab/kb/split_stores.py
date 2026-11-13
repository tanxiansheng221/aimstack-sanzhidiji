# -*- coding: utf-8 -*-
"""一次性迁移: cards.jsonl 按 origin 切两库
wiki -> kb_store (固定底座, 不动)
user+crawl -> kb_dyn_store (动态库, 可整包更换)
产出: 两库 cards.jsonl + 动态库 FTS 重建 (小)
固定库 FTS 暂不重建 (残留少量已迁走卡的索引项, 检索合并按 id 去重兜底)"""
import io, json, os, shutil, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = r"H:\AimCode\lab\cpu-native"
BASE_STORE = os.path.join(BASE, "kb_store")
DYN_STORE = os.environ.get("AIM_KB_DYN_ROOT", os.path.join(BASE, "kb_dyn_store"))

src = os.path.join(BASE_STORE, "cards.jsonl")
base_tmp = src + ".base.tmp"
dyn_path = os.path.join(DYN_STORE, "cards.jsonl")

os.makedirs(DYN_STORE, exist_ok=True)
n = {"wiki": 0, "user": 0, "crawl": 0, "other": 0}
with io.open(src, encoding="utf-8") as f, \
     io.open(base_tmp, "w", encoding="utf-8", newline="\n") as fb, \
     io.open(dyn_path, "w", encoding="utf-8", newline="\n") as fd:
    for line in f:
        if '"origin": "wiki"' in line:
            fb.write(line); n["wiki"] += 1
        elif '"origin": "user"' in line or '"origin": "crawl"' in line:
            fd.write(line)
            n["user" if '"origin": "user"' in line else "crawl"] += 1
        else:
            fb.write(line); n["other"] += 1   # origin 缺失的老卡留底座

# 原子换底座 cards.jsonl
shutil.move(base_tmp, src)
print("partitioned:", n, flush=True)

# 动态库 FTS 重建 (卡少, 秒级)
sys.path.insert(0, BASE)
from kb.fts_index import FtsIndex
fts = FtsIndex(DYN_STORE)
added, total = fts.add_jsonl(dyn_path, quiet=True)
print("dyn fts: +%d/%d cards | dyn total %d" % (added, total, fts.count()), flush=True)
print("BASE_STORE:", BASE_STORE, flush=True)
print("DYN_STORE :", DYN_STORE, flush=True)
