# -*- coding: utf-8 -*-
"""百度/360/sogou robots 对搜索入口路径的态度"""
import sys, urllib.robotparser

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

checks = [
    ("https://www.baidu.com", "/s?wd=test"),
    ("https://www.so.com", "/s?q=test"),
    ("https://www.sogou.com", "/web?query=test"),
]
for host, path in checks:
    print("---", host)
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(host + "/robots.txt")
        rp.read()
        u = host + path
        print("  can_fetch(*) %s -> %s" % (path, rp.can_fetch("*", u)))
    except Exception as e:
        print("  FAIL:", str(e)[:100])
    # 打印原文前 40 行有效规则
    try:
        import urllib.request
        req = urllib.request.Request(host + "/robots.txt",
                                     headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=15).read(200000).decode("utf-8", "ignore")
        n = 0
        for line in txt.splitlines():
            l = line.strip()
            if l and not l.startswith("#"):
                print("   ", l)
                n += 1
                if n >= 40:
                    print("    ...(截断)")
                    break
    except Exception as e:
        print("  FETCH FAIL:", str(e)[:100])
