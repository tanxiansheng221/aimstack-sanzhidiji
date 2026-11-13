# -*- coding: utf-8 -*-
"""Gate rejection test + docx ingest test (hand-built docx via zipfile)."""
import io, json, os, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.kb_core import KB
from kb.ingest import docx_to_text, load_any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BT = r"H:\AimCode\build-tools"
tmp = os.path.join(BT, "_kbtest")
os.makedirs(tmp, exist_ok=True)

# ---- 1. gate rejection: fresh KB, add same text twice ----
kbp = os.path.join(tmp, "cards.jsonl")
if os.path.exists(kbp):
    os.remove(kbp)
kb = KB(tmp)
r1 = kb.add("测试文档", "这是一段足够长的测试正文，用来验证知识库闸门的查重能力，必须超过二十个字才算有效卡片内容。", "t1")
r2 = kb.add("测试文档", "这是一段足够长的测试正文，用来验证知识库闸门的查重能力，必须超过二十个字才算有效卡片内容。", "t2")
st1 = [st for _, st in r1]; st2 = [st for _, st in r2]
print("first add:", st1, "| second add:", st2)
assert any(s == "added" for s in st1) and any(s.startswith("rejected") for s in st2), "GATE FAIL"
print("GATE OK: duplicate rejected")

# ---- 2. docx ingest: hand-build minimal docx ----
docx_path = os.path.join(tmp, "test.docx")
WNS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
paras = "".join("<w:p><w:r><w:t>%s</w:t></w:r></w:p>" % p for p in
                ["自研知识库文档测试", "第一段：外置知识库是模型的知识外挂，问答秒回全靠它。",
                 "第二段：用户在聊天里手动发文档，模型自动清洗转化入库。"])
doc_xml = ('<?xml version="1.0"?><w:document %s><w:body>%s</w:body></w:document>' % (WNS, paras))
ct_xml = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="xml" ContentType="application/xml"/></Types>')
with zipfile.ZipFile(docx_path, "w") as z:
    z.writestr("[Content_Types].xml", ct_xml)
    z.writestr("word/document.xml", doc_xml)
title, text = load_any(docx_path)
print("docx title:", title, "| text len:", len(text))
assert "外挂" in text and "清洗" in text, "DOCX FAIL"
print("DOCX OK")
print("ALL KB TESTS PASS")
