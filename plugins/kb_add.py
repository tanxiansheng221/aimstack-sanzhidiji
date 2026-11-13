# -*- coding: utf-8 -*-
"""cap: kb.add — 用户上传文档增量入库 (txt/md/html/docx/pdf, 全 stdlib 解析)
入: argv[1] = 查询文件 (UTF-8 JSON {"path": "..."})
管线: load_any 解析 -> chunk -> DedupGate 查重 (fail-closed) -> cards.jsonl
      -> FTS 增量索引 (只插新卡, 不全库重扫)
出: {"cap","file","added","rejected","fts_indexed","kb_total"}"""
import io, json, os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from kb.kb_core import KB
from kb.fts_index import FtsIndex
from kb import ingest

KB_DYN = os.environ.get("AIM_KB_DYN_ROOT", r"H:\AimCode\lab\cpu-native\kb_dyn_store")
CARDS = os.path.join(KB_DYN, "cards.jsonl")

def fts_add_new(fts, want_ids):
    """只把本次新增的卡插进 FTS (全文件流式扫但逐行廉价比对, 不 json.loads 全库)"""
    want = set(want_ids)
    if not want:
        return 0
    found = []
    with io.open(CARDS, encoding="utf-8") as f:
        for line in f:
            if '"id": "kb-' not in line:
                continue
            if not any(w in line for w in want):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d["id"] in want:
                found.append((d["title"], d["text"], d["id"],
                              d.get("origin", ""), d.get("source", "")))
    if found:
        fts.con.executemany("INSERT INTO cards VALUES (?,?,?,?,?)", found)
        fts.con.commit()
    return len(found)

def main():
    req = json.load(io.open(sys.argv[1], encoding="utf-8"))
    path = req.get("path") or req.get("query", "")   # 兼容总线通用信封
    if not path or not os.path.exists(path):
        json.dump({"cap": "kb.add", "error": "file not found: %s" % path},
                  sys.stdout, ensure_ascii=False)
        return
    title, text = ingest.load_any(path)
    kb = KB(KB_DYN, build_index=False)   # 动态库: gate only
    res = kb.add(title, text, source=path, origin="user", tags=["upload"])
    added = [rid for rid, st in res if st == "added"]
    rejected = len(res) - len(added)
    fts = FtsIndex(KB_DYN)
    n = fts_add_new(fts, added)
    json.dump({"cap": "kb.add", "file": os.path.basename(path),
               "store": "dyn", "dyn_root": KB_DYN,
               "title": title[:60], "added": len(added), "rejected": rejected,
               "fts_indexed": n, "dyn_total": len(kb.cards)},
              sys.stdout, ensure_ascii=False)

if __name__ == "__main__":
    main()
