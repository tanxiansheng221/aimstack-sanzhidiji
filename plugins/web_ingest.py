# -*- coding: utf-8 -*-
"""cap: web.ingest — 搜索引擎 -> 抓取 -> 质量闸门 -> KB 增量入库 (creeper 管线)
入: argv[1] = 查询文件 (UTF-8 JSON {"query": "...", "topn": N})
管线: SERP(bing+百度+360 并行) -> 域名多样化取 topN (白名单优先) -> 并行裸抓(4 工人)
      -> 串行渲染级联 (残文壳页/超时站, robots 先查) -> fetch_generic -> quality 五信号闸
      -> kb.add(origin=crawl) -> FTS 增量索引 (只插新卡)
出: {"cap","query","picked","crawled","rejected","fts_indexed","details"}"""
import contextlib, io, json, os, sys
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
sys.path.insert(0, r"H:\AimCode\aimp\plugins")
from kb.kb_core import KB
from kb.fts_index import FtsIndex
from creeper.serp import multi_search
from creeper.engine import FetchKernel, fetch_generic
from creeper.render import render_fetch
from creeper.quality import quality
from kb_add import fts_add_new   # 增量索引助手 (定义在 kb_add.py)

KB_DYN = os.environ.get("AIM_KB_DYN_ROOT", r"H:\AimCode\lab\cpu-native\kb_dyn_store")

# 优质源白名单 (实测正文质量过硬的站排前面; 同层内保持 SERP 排序不变)
PRIORITY = ("developer.mozilla.org", "wikipedia.org", "firefox.com", "mozilla.org",
            "docs.python.org", "learn.microsoft.com", "github.com", ".gov.cn",
            "stackoverflow.com")

def _prio(url):
    for i, d in enumerate(PRIORITY):
        if d in url:
            return i
    return len(PRIORITY)

def main():
    req = json.load(io.open(sys.argv[1], encoding="utf-8"))
    q = req.get("query", "")
    topn = int(req.get("topn", 2))
    buf = io.StringIO()
    hits = []
    if q:
        with contextlib.redirect_stdout(buf):   # 保 stdout JSON 纯净
            # sogou 风控壳页期暂退二线; 必应+百度+360 三腿
            hits = multi_search(q, engines=("bing", "baidu", "so360"), per_engine=6)
    picked, domains = [], set()
    for h in hits:
        try:
            dom = h["url"].split("/")[2]
        except Exception:
            continue
        if dom not in domains:
            domains.add(dom)
            picked.append(h)
    # 候选上限放宽到 topn+4: 前排站被 robots/质量闸拒后继续试后面的
    picked = picked[:topn + 4]
    picked.sort(key=lambda h: _prio(h["url"]))   # 白名单优先, 稳定排序保层内原序

    kb = KB(KB_DYN, build_index=False)   # 动态库: gate only
    kernel = FetchKernel(min_interval=1.5)
    # 并行裸抓 (4 工人, 不排队); robots/限速由 kernel.fetch 内部执行
    plain = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_generic, kernel, h["url"], h.get("title")): h["url"]
                for h in picked}
        for f, url in futs.items():
            try:
                plain[url] = f.result(timeout=120)
            except Exception as e:
                plain[url] = (None, "thread_error:" + str(e)[:60])

    details, added_ids = [], []
    crawled = rejected = 0
    for h in picked:   # 主线程按序结算: 渲染级联(串行, 一台 Edge) -> 质量闸 -> 入库
        url = h["url"]
        doc, st = plain.get(url, (None, "no_result"))
        # 渲染级联: 残文壳页 + 裸抓超时/断连的站 (真浏览器常能过) — robots 先查
        rescue = st == "too_little_text" or (
            st.startswith("fetch_error")
            and any(k in st for k in ("timed out", "10060", "reset", "10054")))
        if not doc and rescue and kernel._robot_ok(url):
            try:
                doc, st = render_fetch(url, timeout_ms=30000)
            except Exception as e:
                st = "render_fail:" + str(e)[:60]
        if not doc:
            details.append({"url": url[:60], "status": st})
            rejected += 1
            continue
        ok, sc, rs = quality(doc["text"])
        if not ok:
            details.append({"url": url[:60], "status": "lowqual:%s" % rs,
                            "score": round(sc, 2)})
            rejected += 1
            continue
        res = kb.add(doc["title"], doc["text"], source=url,
                     origin="crawl", tags=["creeper"])
        added = [rid for rid, st2 in res if st2 == "added"]
        added_ids.extend(added)
        crawled += 1
        details.append({"url": url[:60], "added": len(added),
                        "gated": len(res) - len(added), "quality": round(sc, 2)})
        if crawled >= topn:
            break

    fts = FtsIndex(KB_DYN)
    idx = fts_add_new(fts, added_ids)
    json.dump({"cap": "web.ingest", "query": q, "picked": len(picked),
               "store": "dyn", "crawled": crawled, "rejected": rejected,
               "fts_indexed": idx, "details": details},
              sys.stdout, ensure_ascii=False)

if __name__ == "__main__":
    main()
