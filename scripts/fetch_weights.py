# -*- coding: utf-8 -*-
"""fetch_weights.py — 下载微软 BitNet 权重的 AIM-W 转换成品
归属: 权重 © Microsoft Corporation (MIT), 本 Release 为格式转换分发, 保留归属声明。
用法: python scripts/fetch_weights.py"""
import hashlib, os, sys, urllib.request, zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REL = "https://github.com/tanxiansheng221/aimstack-sanzhidiji/releases/download/weights-v1"
FILES = [
    ("m2b_tern.aimw.zip", "6a2badb36fbdec86665cf33c1aab1e71"),
    ("m2b_llama3_head.gguf", "52885b7b772703877483a4008126c833"),
]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "bitnet2b")
os.makedirs(DATA, exist_ok=True)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

for fname, want in FILES:
    url = "%s/%s" % (REL, fname)
    dst = os.path.join(DATA, fname)
    if os.path.exists(dst) and md5(dst) == want:
        print("已存在, 跳过:", fname)
    else:
        print("下载:", url, flush=True)
        urllib.request.urlretrieve(url, dst)
        got = md5(dst)
        if got != want:
            raise SystemExit("MD5 不符: %s got=%s want=%s" % (fname, got, want))
        print("MD5 OK:", got)
    if fname.endswith(".zip"):
        with zipfile.ZipFile(dst) as z:
            z.extractall(DATA)
        print("解压完成 ->", os.path.join(DATA, fname[:-4]))

print("\n完成. 权重: data/bitnet2b/m2b_tern.aimw + data/m2b_llama3_head.gguf")
print("起服务: python ui/server.py")
