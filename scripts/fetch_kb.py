# -*- coding: utf-8 -*-
"""fetch_kb.py — 下载知识库成品 (77.6 万卡中文维基, CC-BY-SA)
来源: 本仓库 GitHub Releases. 卡片单件; FTS 索引因上行带宽限制分 22 片, 下载后拼装.
流程: 下载 -> MD5 校验 -> 解压到 kb_store/
用法: python scripts/fetch_kb.py            # 全套
      python scripts/fetch_kb.py --cards    # 只下卡片
      python scripts/fetch_kb.py --check    # 只校验已存在文件"""
import hashlib, os, sys, urllib.request, zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REL = "https://github.com/tanxiansheng221/aimstack-sanzhidiji/releases/download/kb-v1"
CARDS_ZIP = ("kb_cards.zip", "60a095afe7e90aababc52764db74823c")
FTS_ZIP_MD5 = "f1daef2f0ac4a27f059bdac63a13bef8"   # 拼装完成的整包校验
FTS_PARTS = 22                                     # s00..s21
HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(os.path.dirname(HERE), "kb_store")

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, dst, want_md5):
    if os.path.exists(dst) and md5(dst) == want_md5:
        print("已存在, 跳过:", os.path.basename(dst))
        return
    print("下载:", url, flush=True)
    urllib.request.urlretrieve(url, dst)
    got = md5(dst)
    if got != want_md5:
        raise SystemExit("MD5 不符: %s got=%s want=%s (重跑本脚本会重试)" % (dst, got, want_md5))
    print("MD5 OK:", got)

def main():
    os.makedirs(KB, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""

    # 1) 卡片
    cards_zip = os.path.join(KB, CARDS_ZIP[0])
    if only != "--fts-only":
        fetch("%s/%s" % (REL, CARDS_ZIP[0]), cards_zip, CARDS_ZIP[1])
        with zipfile.ZipFile(cards_zip) as z:
            z.extractall(KB)
        print("解压完成 -> cards.jsonl")

    # 2) FTS 索引 (22 片拼装)
    if only != "--cards":
        parts_zip = os.path.join(KB, "kb_fts.zip")
        if not (os.path.exists(parts_zip) and md5(parts_zip) == FTS_ZIP_MD5):
            for i in range(FTS_PARTS):
                name = "kb_fts.zip.s%02d" % i
                p = os.path.join(KB, name)
                src_md5 = None   # 分片无独立校验, 拼装后整包校验兜底
                fetch("%s/%s" % (REL, name), p, src_md5) if src_md5 else (
                    (not os.path.exists(p)) and (
                        print("下载:", REL + "/" + name, flush=True) or
                        urllib.request.urlretrieve(REL + "/" + name, p)))
            print("拼装 22 片...", flush=True)
            with open(parts_zip, "wb") as out:
                for i in range(FTS_PARTS):
                    p = os.path.join(KB, "kb_fts.zip.s%02d" % i)
                    out.write(open(p, "rb").read())
                    os.remove(p)
            got = md5(parts_zip)
            if got != FTS_ZIP_MD5:
                raise SystemExit("拼装后 MD5 不符: got=%s (重跑重下)" % got)
            print("整包 MD5 OK:", got)
        with zipfile.ZipFile(parts_zip) as z:
            z.extractall(KB)
        print("解压完成 -> kb_fts.db")

    print("\n完成. 起服务: python ui/server.py -> http://127.0.0.1:8600")

if __name__ == "__main__":
    main()
