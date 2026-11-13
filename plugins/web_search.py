# -*- coding: utf-8 -*-
"""cap: web.search — 搜索引擎 SERP (creeper 多引擎: bing_cn + sogou)
入: argv[1] = 查询文件 (UTF-8 JSON {"query": "...", "topk": N})
出: stdout = {"cap", "query", "results": [{title,url}]}"""
import contextlib, io, json, os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab"))
from creeper.serp import multi_search

def main():
    req = json.load(io.open(sys.argv[1], encoding="utf-8"))
    q = req.get("query", "")
    n = int(req.get("topk", 5))
    buf = io.StringIO()
    hits = []
    if q:
        # serp.multi_search 会向 stdout 打引擎失败日志, 重定向保 JSON 纯净
        with contextlib.redirect_stdout(buf):
            # sogou 风控壳页期暂退二线; 必应+百度+360 三腿
            hits = multi_search(q, engines=("bing", "baidu", "so360"),
                                per_engine=max(3, n))[:n]
    json.dump({"cap": "web.search", "query": q, "results": hits},
              sys.stdout, ensure_ascii=False)

if __name__ == "__main__":
    main()
