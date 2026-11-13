# -*- coding: utf-8 -*-
"""底座 FTS 清残留: 已迁到动态库的 369 卡, 按行 id 从 base fts.db 定点 DELETE
(免重建 77 万行大索引; rowid 收集 -> 分批 DELETE)"""
import io, json, os, sqlite3, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = r"H:\AimCode\lab\cpu-native"
DYN = os.environ.get("AIM_KB_DYN_ROOT", os.path.join(BASE, "kb_dyn_store"))

want = set()
with io.open(os.path.join(DYN, "cards.jsonl"), encoding="utf-8") as f:
    for line in f:
        try:
            want.add(json.loads(line)["id"])
        except Exception:
            pass
print("dyn ids:", len(want), flush=True)

con = sqlite3.connect(os.path.join(BASE, "kb_store", "kb_fts.db"))
rowids = [r[0] for r in con.execute("SELECT rowid, id FROM cards") if r[1] in want]
print("ghost entries in base fts:", len(rowids), flush=True)

CH = 100
for i in range(0, len(rowids), CH):
    chunk = rowids[i:i + CH]
    con.execute("DELETE FROM cards WHERE rowid IN (%s)" % ",".join(map(str, chunk)))
    con.commit()
print("deleted. base fts now:", con.execute("SELECT count(*) FROM cards").fetchone()[0], flush=True)
