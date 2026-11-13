# -*- coding: utf-8 -*-
"""Rich-format file ingest -> KB. Stdlib only.

Supported: .txt .md / .html .htm / .docx (zip+xml) / .pdf (zlib stream, best-effort)
Every format funnels into plain text then kb.add (same gate as crawler).
"""
import io, json, os, re, sys, zipfile, zlib
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ---------- html ----------
class _TextX(HTMLParser):
    SKIP = {"script", "style", "noscript", "head"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []; self._skip = 0; self._title = ""
        self._in_title = False
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP: self._skip += 1
        if tag == "title": self._in_title = True
    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip: self._skip -= 1
        if tag == "title": self._in_title = False
    def handle_data(self, data):
        if self._in_title:
            self._title += data
        elif not self._skip:
            d = data.strip()
            if d: self.parts.append(d)

def html_to_text(html):
    p = _TextX()
    try:
        p.feed(html)
    except Exception:
        pass
    return p._title.strip(), "\n".join(p.parts)


# ---------- docx ----------
def docx_to_text(path):
    """docx = zip of xml; pull w:t runs, paragraphs on w:p."""
    out = []
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for para in root.iter(ns + "p"):
        runs = [t.text or "" for t in para.iter(ns + "t")]
        line = "".join(runs).strip()
        if line:
            out.append(line)
    return "\n".join(out)


# ---------- pdf (best-effort: FlateDecode text streams) ----------
_TJ = re.compile(r"\((?:\\.|[^\\()])*\)")
_PDFTEXT = re.compile(r"[\u0020-\u007e\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+")

def pdf_to_text(path):
    """Extract readable runs from decompressed content streams. Works for many text PDFs;
    CJK-embedded-font PDFs may yield little (honest limit)."""
    raw = open(path, "rb").read()
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        data = m.group(1)
        try:
            data = zlib.decompress(data)
        except Exception:
            pass
        try:
            s = data.decode("latin-1", "ignore")
        except Exception:
            continue
        for tok in _TJ.findall(s):
            inner = tok[1:-1]
            inner = re.sub(r"\\([()\\])", r"\1", inner)
            chunks.append(inner)
    txt = " ".join(chunks)
    return " ".join(_PDFTEXT.findall(txt))


# ---------- dispatch ----------
def load_any(path):
    """Return (title, text). Raises on unsupported."""
    ext = os.path.splitext(path)[1].lower()
    base = os.path.splitext(os.path.basename(path))[0]
    if ext in (".txt", ".md", ".markdown"):
        raw = io.open(path, encoding="utf-8", errors="ignore").read()
        t = ""
        m = re.match(r"^#\s+(.+)", raw)
        if m: t = m.group(1)
        return t or base, raw
    if ext in (".html", ".htm"):
        html = io.open(path, encoding="utf-8", errors="ignore").read()
        title, text = html_to_text(html)
        return title or base, text
    if ext == ".docx":
        return base, docx_to_text(path)
    if ext == ".pdf":
        return base, pdf_to_text(path)
    raise ValueError("unsupported format: " + ext)
