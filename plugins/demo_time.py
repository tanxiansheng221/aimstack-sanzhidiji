# -*- coding: utf-8 -*-
"""cap: demo.time — 最小验证插件 (总线协议冒烟)
入: argv[1] = 查询文件 (UTF-8 JSON, 原样回显 echo 字段)
出: stdout = {"cap","now","pid","echo"}"""
import io, json, os, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def main():
    try:
        req = json.load(io.open(sys.argv[1], encoding="utf-8"))
    except Exception:
        req = {}
    json.dump({"cap": "demo.time", "now": time.strftime("%Y-%m-%d %H:%M:%S"),
               "pid": os.getpid(), "echo": req}, sys.stdout, ensure_ascii=False)

if __name__ == "__main__":
    main()
