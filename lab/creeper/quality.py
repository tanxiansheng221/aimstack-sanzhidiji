# -*- coding: utf-8 -*-
"""Quality gate (layer 2): 查重闸只拦重复，这里拦低质。
Signals: boilerplate-shell (登录/关注/点赞/导航壳), residual length, sentence structure,
information density. Fail-closed, reasons returned. Stdlib only."""
import re, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BOILER = ["关注", "点赞", "评论", "收藏", "分享", "复制链接", "登录", "注册", "首页",
          "导航", "下一篇", "上一篇", "相关推荐", "扫码", "下载客户端", "隐私政策",
          "版权所有", "免责声明", "阅读全文", "展开全文", "个人主页", "消息", "设置",
          "Follow", "Like", "Share", "Subscribe", "Sign in", "Log in", "Home",
          "Menu", "Search", "Comment", "Cookie"]

def _strip_boiler_lines(text):
    """Drop lines that are mostly shell (short line containing boiler terms)."""
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if len(s) < 30 and any(b.lower() in s.lower() for b in BOILER):
            continue
        kept.append(s)
    return "\n".join(kept)

def quality(text):
    """Return (ok, score, reasons). ok=False -> don't ingest."""
    reasons = []
    text = text or ""
    residual = _strip_boiler_lines(text)
    residual_len = len(residual.replace("\n", "").replace(" ", ""))

    # 1) residual meaningful length
    if residual_len < 80:
        return False, 0.0, ["residual_too_short:%d" % residual_len]

    # 2) boilerplate shell ratio
    boiler_hits = sum(1 for b in BOILER if b.lower() in text.lower())
    lines = [l for l in text.split("\n") if l.strip()]
    shell_ratio = boiler_hits / max(len(lines), 1)
    if shell_ratio > 0.5:
        reasons.append("shell_ratio:%.2f" % shell_ratio)

    # 3) sentence structure (prose should have enders)
    sent = len(re.findall(r"[。！？.!?；;]", residual))
    if sent < 3:
        reasons.append("too_few_sentences:%d" % sent)

    # 4) information density: max of CJK char diversity and latin word diversity
    # (tech docs mix code/terms into CJK text -> use whichever channel is healthier)
    body = re.sub(r"\s+", "", residual)
    cjk = re.findall(r"[\u4e00-\u9fff]", body)
    latin = re.findall(r"[a-zA-Z]+", body)
    d_cjk = len(set(cjk)) / len(cjk) if len(cjk) > 50 else None
    d_lat = len(set(w.lower() for w in latin)) / len(latin) if len(latin) > 20 else None
    density = max([d for d in (d_cjk, d_lat) if d is not None], default=0.0)
    if density < 0.30:
        reasons.append("low_density:%.2f" % density)

    # 5) menu-list smell: many short lines
    short_lines = sum(1 for l in lines if len(l.strip()) < 12)
    if lines and short_lines / len(lines) > 0.6:
        reasons.append("menu_smell:%.2f" % (short_lines / len(lines)))

    # score: start 1.0, subtract per signal
    score = 1.0
    score -= min(shell_ratio, 0.5)
    score -= 0.2 if sent < 5 else 0.0
    score -= max(0.0, 0.5 - density)
    score -= 0.2 if short_lines / max(len(lines), 1) > 0.6 else 0.0
    score = max(0.0, round(score, 2))

    ok = not reasons and score >= 0.6
    if ok:
        reasons = ["pass:%.2f" % score]
    return ok, score, reasons
