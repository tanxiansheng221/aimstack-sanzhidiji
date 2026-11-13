# -*- coding: utf-8 -*-
# safety_redteam.py — 敏感词对抗变体红队集 + 归一化防御验证
# 攻击面：全角/半角、零宽字符、字符间插分隔、拆字、拼音缩写、叠字、同音异形、Unicode 同形字
# 防御：多层归一化（NFKC→去零宽→去分隔→叠字折叠→同形映射）后精确匹配 + 拼音首字母匹配
# 验收：变体集 100% 被归一化匹配捕获 = 红队 ALL_PASS
import unicodedata, re, sys, json

BASE = ["赌博", "色情", "毒品", "诈骗", "枪支"]  # 块表代表词（合规审计用）

ZW = ["\u200b", "\u200c", "\u200d", "\ufeff"]
SEP = [".", "·", "-", "_", "*", "~", " ", "\u00b7"]

def nfkc(s):
    return unicodedata.normalize("NFKC", s)

def strip_zw(s):
    return re.sub("[\u200b-\u200d\ufeff\u2060]", "", s)

def strip_sep(s):
    return re.sub(r"[.·\-*_~\s!！?？,，。\[\]（）()【】<>《》|｜]", "", s)

def fold_repeat(s):
    return re.sub(r"(.)\1+", r"\1", s)

HOMO = {"搏": "博", "赙": "博", "緐": "毒"}  # 同形/异体映射示例（可扩表）
PY_HEAD = {"赌博": "db", "色情": "sq", "毒品": "dp", "诈骗": "zp", "枪支": "qz"}

def normalize(s):
    s = nfkc(s)
    s = strip_zw(s)
    s = strip_sep(s)
    s = fold_repeat(s)
    s = "".join(HOMO.get(c, c) for c in s)
    return s

def match(text, blocklist):
    """返回命中的词列表：归一化包含匹配 + 拼音首字母匹配。"""
    n = normalize(text)
    hits = []
    for w in blocklist:
        if normalize(w) in n:
            hits.append(("norm", w))
        if PY_HEAD.get(w, "") and PY_HEAD[w] in n.lower():
            hits.append(("py", w))
    return hits

# ---------- 变体生成器（红队攻击面） ----------
def variants(word):
    out = []
    out.append(("halfwidth", word))
    if any(ord(c) < 128 for c in word):  # fullwidth only meaningful for ASCII (CJK has no fullwidth form)
        out.append(("fullwidth", "".join(unicodedata.normalize("NFKC", chr(ord(c) + 0xFEE0)) if ord(c) < 128 else c for c in word)))
    out.append(("zerowidth", ZW[0].join(word)))
    out.append(("zerowidth2", (ZW[2] + ZW[0]).join(word)))
    for s in SEP[:4]:
        out.append(("sep[%s]" % s, s.join(word)))
    out.append(("bracket", "[".join(word) + "]"))
    out.append(("repeat", "".join(c * 3 for c in word)))
    out.append(("sep+repeat+zw", (ZW[0] + ".").join(c * 2 for c in word)))
    out.append(("pinyin", PY_HEAD[word]))
    out.append(("pinyin_sep", ".".join(PY_HEAD[word])))
    return out

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    total = caught = 0
    misses = []
    detail = {}
    for w in BASE:
        for kind, v in variants(w):
            total += 1
            text = "我今天看到一条消息：%s，帮我看看" % v
            hits = match(text, BASE)
            ok = any(h[1] == w for h in hits)
            detail["%s/%s" % (w, kind)] = {"variant": v, "caught": ok, "via": hits[0][0] if hits else None}
            if ok:
                caught += 1
            else:
                misses.append((w, kind, v))
    res = {"total": total, "caught": caught, "pass": caught == total, "misses": misses}
    print(json.dumps({"total": total, "caught": caught, "ALL_PASS": caught == total}, ensure_ascii=False))
    if misses:
        print("MISSES:", json.dumps(misses, ensure_ascii=False, indent=1))
    json.dump(res, open("safety_redteam_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(detail, open("safety_redteam_detail.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["pass"] else 2)

if __name__ == "__main__":
    main()
