# -*- coding: utf-8 -*-
"""fetch_kb.py — 下载知识库成品 (77.6 万卡中文维基, CC-BY-SA)
来源: 本仓库 GitHub Releases. 下载 -> MD5 校验 -> 解压到 kb_store/
用法: python scripts/fetch_kb.py            # 全套下载
      python scripts/fetch_kb.py --cards    # 只下卡片
      python scripts/fetch_kb.py --check    # 只校验已存在文件"""
import hashlib, io, os, shutil, sys, urllib.request, zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Releases 附件直链 (版本号随 Release 更新, 见 README)
REL = "https://github.com/tanxiansheng221/aimstack-sanzhijdii/releases/download/kb-v1"
FILES = [
    ("kb_cards.zip", "cards.jsonl.zip", "<MD5-发布时填写>"),
    ("kb_fts.zip", "kb_fts.db.zip", "<MD5-发布时填写>"),
]
HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(os.path.dirname(HERE), "kb_store")

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(fname, want_md5):
    url = "%s/%s" % (REL, fname)
    dst = os.path.join(KB, fname)
    os.makedirs(KB, exist_ok=True)
    if os.path.exists(dst) and want_md5 not in ("<MD5-发布时填写>",) and md5(dst) == want_md5:
        print("已存在, 跳过:", fname)
        return dst
    print("下载:", url)
    urllib.request.urlretrieve(url, dst)
    got = md5(dst)
    if want_md5 not in ("<MD5-发布时填写>",) and got != want_md5:
        raise SystemExit("MD5 不符: %s got=%s" % (fname, got))
    print("MD5 OK:", got)
    return dst

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for fname, inner, want in FILES:
        if only == "--cards" and "fts" in fname:
            continue
        if only == "--check":
            p = os.path.join(KB, fname)
            print(fname, "MD5:", md5(p) if os.path.exists(p) else "缺失")
            continue
        p = fetch(fname, want)
        with zipfile.ZipFile(p) as z:
            z.extractall(KB)
        print("解压完成 ->", os.path.join(KB, inner.replace(".zip", "")))
    print("\n完成. 起服务: python ui/server.py -> http://127.0.0.1:8600")

if __name__ == "__main__":
    main()
