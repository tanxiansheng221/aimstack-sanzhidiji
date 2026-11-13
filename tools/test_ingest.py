# -*- coding: utf-8 -*-
"""增量入库验收: 3 格式用户文档 (md/html/docx) 走 kb.add -> 查回验证 -> 去重验证
   + web.ingest 活网爬取入库"""
import os, re, subprocess, sys, time, zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
MINI = r"H:\AimCode\aimp\build\mini_llama.gguf"
CWD = r"H:\AimCode\aimp"
DOC = r"H:\AimCode\aimp\testdocs"
MARK = "AIMPKBTEST2026"
total = npass = 0

def run(args, timeout=300):
    t0 = time.time()
    p = subprocess.run(args, cwd=CWD, capture_output=True, timeout=timeout)
    return p.stdout.decode("utf-8", "ignore"), p.returncode, time.time() - t0

def rec(name, ok, notes):
    global total, npass
    total += 1
    npass += bool(ok)
    print("[%s] %-16s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

# ---------- 造 3 格式测试文档 (合成占位内容, 非私有文档) ----------
os.makedirs(DOC, exist_ok=True)
body_cn = "这是 aimp 插件总线增量入库功能验收用的合成测试文档，讲述 Thunderbird 通信协议的三层封装。"
with open(os.path.join(DOC, "test_md.md"), "w", encoding="utf-8") as f:
    f.write("# %s 文档\n\n%s\nMarkdown 格式直读测试段。\n" % (MARK, body_cn))
with open(os.path.join(DOC, "test_html.html"), "w", encoding="utf-8") as f:
    f.write("<html><head><title>%s 页面</title></head><body><p>%s</p>"
            "<p>HTML 抽取测试段。</p></body></html>" % (MARK, body_cn))
docx_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>%s 文档</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>%s</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>DOCX 解析测试段。</w:t></w:r></w:p></w:body></w:document>'
            ) % (MARK, body_cn)
with zipfile.ZipFile(os.path.join(DOC, "test_doc.docx"), "w") as z:
    z.writestr("word/document.xml", docx_xml)
print("docs ready: md/html/docx @ %s" % DOC, flush=True)

# ---------- 1. 三格式逐个走 kb.add ----------
for name, fn in [("MD", "test_md.md"), ("HTML", "test_html.html"), ("DOCX", "test_doc.docx")]:
    out, rc, dt = run([EXE, MINI, "--cap", "kb.add", os.path.join(DOC, fn)], 300)
    m = re.search(r'"added":\s*(\d+)', out)
    ok = rc == 0 and m and int(m.group(1)) > 0
    rec("ADD_%s" % name, ok, "%ds added=%s%s" % (
        dt, m.group(1) if m else "-",
        " err=" + out[:80] if not ok else ""))

# ---------- 2. 去重闸: 同一文档再喂一遍必须被拒 ----------
out, rc, dt = run([EXE, MINI, "--cap", "kb.add", os.path.join(DOC, "test_md.md")], 300)
m = re.search(r'"added":\s*(\d+)', out)
rj = re.search(r'"rejected":\s*(\d+)', out)
ok = rc == 0 and m and int(m.group(1)) == 0 and rj and int(rj.group(1)) > 0
rec("DEDUP_REJECT", ok, "added=%s rejected=%s (fail-closed 查重门)"
    % (m.group(1) if m else "-", rj.group(1) if rj else "-"))

# ---------- 3. 入库即检索: 用标记词查回 ----------
for fmt in ["md", "html", "docx"]:
    out, rc, dt = run([EXE, MINI, "--cap", "kb.search", MARK + " " + fmt, "3"])
    ok = rc == 0 and '"results": [{' in out
    rec("SEARCHBACK_%s" % fmt.upper(), ok, "%ds %s" % (dt, fmt))

# ---------- 4. web.ingest: 搜索 -> 抓取 -> 过闸 -> 入库 ----------
out, rc, dt = run([EXE, MINI, "--cap", "web.ingest", "BitNet b1.58 推理", "2"], 600)
cw = re.search(r'"crawled":\s*(\d+)', out)
ad = re.search(r'"fts_indexed":\s*(\d+)', out)
ok = rc == 0 and cw and int(cw.group(1)) > 0 and ad
rec("WEB_INGEST", ok, "%ds crawled=%s fts=%s details=%s" % (
    dt, cw.group(1) if cw else "-", ad.group(1) if ad else "-",
    (re.search(r'"details":\s*(\[.*?\])', out).group(1)[:150] if re.search(r'"details":\s*(\[.*?\])', out) else "-")))

print("\n=== SUMMARY: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
