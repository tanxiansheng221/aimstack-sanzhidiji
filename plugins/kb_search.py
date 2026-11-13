# -*- coding: utf-8 -*-
"""cap: kb.search — 双库检索: 固定底座(wiki) + 动态库(user/crawl, 可整包更换)
入: argv[1] = 查询文件 (UTF-8 JSON {"query": "...", "topk": N})
出: results 每条带 "store": "base"|"dyn"; 按 score 合并排序, id 去重
库根: 固定 AIM_KB_BASE > cpu-native/kb_store ; 动态 AIM_KB_DYN > cpu-native/kb_dyn_store
查询预处理: 空格 + ASCII/CJK 边界切段 -> CJK 去停用词滑 3 字窗 -> AND 精确 / OR 覆盖率重排"""
import io, json, os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from kb.fts_index import FtsIndex

KB_BASE = os.environ.get("AIM_KB_BASE", r"H:\AimCode\lab\cpu-native\kb_store")
KB_DYN = os.environ.get("AIM_KB_DYN_ROOT", r"H:\AimCode\lab\cpu-native\kb_dyn_store")
_STOP = "的了是在呢吗什么啊呀和与对把被它们你我都"


def _cjk(ch):
    return "\u4e00" <= ch <= "\u9fff"


def segment(query):
    """空格 + ASCII/CJK 边界切段"""
    segs = []
    for part in query.split():
        cur, mode = "", None
        for ch in part:
            if ch.isascii() and ch.isalnum():
                m = "a"
            elif _cjk(ch):
                m = "c"
            else:
                m = None
            if m is None:
                if cur:
                    segs.append(cur)
                cur, mode = "", None
            elif m != mode and cur:
                segs.append(cur)
                cur, mode = ch, m
            else:
                cur += ch
                mode = m
        if cur:
            segs.append(cur)
    return segs


def build_terms(query):
    ands, ors, cov = [], [], []
    for seg in segment(query):
        if seg[0].isascii():
            if len(seg) >= 2:
                ands.append('"%s"' % seg)
                ors.append('"%s"' % seg)
                cov.append(seg)
        else:
            core = "".join(c for c in seg if c not in _STOP)
            if len(core) >= 3:
                wins = [core[i:i + 3] for i in range(len(core) - 2)]
                ands.extend('"%s"' % w for w in wins)
                ors.extend('"%s"' % w for w in wins)
                cov.extend(wins)
            elif len(core) >= 2:
                cov.append(core)   # 2 字词 trigram 查不了, 只进覆盖率
    return ands, ors, cov


def _fts_raw(idx, terms, topk):
    sql = ("SELECT id, title, text, origin, bm25(cards) FROM cards "
           "WHERE cards MATCH ? ORDER BY bm25(cards) LIMIT ?")
    rows = idx.con.execute(sql, [terms, topk]).fetchall()
    return [{"id": r[0], "title": r[1], "text": r[2], "origin": r[3],
             "score": round(-r[4], 2)} for r in rows]


def search_store(idx, q, k):
    """单库三级: 整句短语 -> AND 全窗 -> OR 覆盖率重排"""
    rows = idx.search(q, topk=k) if q else []
    ands, ors, cov = build_terms(q)
    if not rows and ands:
        rows = _fts_raw(idx, " AND ".join(ands), k)
    if not rows and ors:
        pool = _fts_raw(idx, " OR ".join(ors), max(40, k * 8))
        for r in pool:
            hay = r["title"] + " " + r["text"]
            r["_cov"] = sum(1 for w in set(cov) if w in hay)
        min_cov = 2 if len(set(cov)) >= 2 else 1
        pool = [r for r in pool if r["_cov"] >= min_cov]
        pool.sort(key=lambda r: (-r["_cov"], -r["score"]))
        rows = pool[:k]
    return rows


def main():
    req = json.load(io.open(sys.argv[1], encoding="utf-8"))
    q = req.get("query", "")
    k = int(req.get("topk", 5))
    merged, seen, stats = [], set(), {}
    for store, root in (("dyn", KB_DYN), ("base", KB_BASE)):
        try:
            idx = FtsIndex(root)
        except Exception as e:
            stats[store] = {"error": str(e)[:80]}
            continue
        stats[store] = idx.count()
        for r in search_store(idx, q, k):
            if r["id"] in seen:   # 底座 FTS 残留项与动态库同卡 -> 去重
                continue
            seen.add(r["id"])
            r["store"] = store
            merged.append(r)
    merged.sort(key=lambda r: -r["score"])
    json.dump({"cap": "kb.search", "base": KB_BASE, "dyn": KB_DYN,
               "cards": stats, "results": merged[:k]},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
