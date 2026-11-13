# -*- coding: utf-8 -*-
import json, urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"

def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(2000).decode("utf-8", "ignore")

# variant 1: original
print("v1:", post("https://api.juejin.cn/content_api/v1/article/query_list",
                 {"id_type": 2, "sort_type": 2, "cursor": "0", "limit": 3,
                  "category_id": "6809637773935378443"})[:300])
# variant 2: with aid param (web client sends these)
print("v2:", post("https://api.juejin.cn/content_api/v1/article/query_list?aid=2608&uuid=0",
                  {"id_type": 2, "sort_type": 2, "cursor": "0", "limit": 3,
                   "category_id": "6809637773935378443"})[:300])
