# -*- coding: utf-8 -*-
"""Render kernel: real-browser fetch via system Edge (Playwright channel=msedge).
For JS-shell pages where plain urllib gets an empty shell. Same pipeline after:
html_to_text -> quality gate -> KB."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.ingest import html_to_text

def _edge_content(url, wait_ms=2500, timeout_ms=25000):
    """Launch system Edge, return (html, title) or (None, reason)."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                                       "Chrome/120.0 Safari/537.36")
            try:
                pg.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                pg.wait_for_timeout(wait_ms)  # let SPA hydrate
                html = pg.content()
                title = pg.title()
            finally:
                b.close()
    except PWTimeout:
        return None, "render_timeout"
    except Exception as e:
        return None, "render_error:" + str(e)[:100]
    return html, title


def render_html(url, wait_ms=2500, timeout_ms=25000):
    """SERP/JS 页用: 保留完整 HTML (要 href 链接)。Return (html, title) or (None, reason)."""
    return _edge_content(url, wait_ms, timeout_ms)


def render_fetch(url, wait_ms=2500, timeout_ms=25000):
    """Return (title, text) or (None, reason). Uses system Edge, no browser download."""
    html, title = _edge_content(url, wait_ms, timeout_ms)
    if html is None:
        return None, title
    t, text = html_to_text(html)
    if len(text.strip()) < 40:
        return None, "render_empty_shell"
    return {"title": t or title or url, "text": text}, "ok"

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://juejin.cn/post/7139740920259407902"
    doc, st = render_fetch(url)
    if doc:
        print("RENDER OK:", doc["title"][:60])
        print("text len:", len(doc["text"]), "| head:", doc["text"][:120].replace("\n", " "))
    else:
        print("FAIL:", st)
