# -*- coding: utf-8 -*-
"""终极全量回归 (万无一失版): 对话三方式 x3 / 生成+确定性 / 分词往返 /
   五插件 / 双库合并 / 整包换库 / 诚实边界 — 全部 x3 遍"""
import os, re, subprocess, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXE = r"H:\AimCode\aimp\build\aimp_probe.exe"
W = r"H:\AimCode\aimp\data\bitnet2b\m2b_tern.aimw"
TOK = r"H:\AimCode\aimp\data\m2b_llama3_head.gguf"
MINI = r"H:\AimCode\aimp\build\mini_llama.gguf"
CWD = r"H:\AimCode\aimp"
ALT_DYN = r"H:\AimCode\aimp\testdocs\kb_alt_dyn"
MARK = "AIMPFINAL%s" % str(int(time.time()))[-6:]   # 每次运行唯一标记

def run(args, env_extra=None, timeout=600):
    t0 = time.time()
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(args, cwd=CWD, capture_output=True, timeout=timeout, env=env)
    return p.stdout.decode("utf-8", "ignore"), p.returncode, time.time() - t0

total = npass = 0
def rec(name, ok, notes):
    global total, npass
    total += 1
    npass += bool(ok)
    print("[%s] %-20s %s" % ("PASS" if ok else "FAIL", name, notes), flush=True)

# ============ A. 对话三方式 x 3 遍 ============
print("== A 对话: 闲聊直答 / RAG 引答 / 无料拒答 ==", flush=True)
for i, q in enumerate(["在吗", "你好呀", "嗨，有人吗"]):
    out, rc, dt = run([EXE, W, TOK, "--gt", q, "32"])
    g = re.search(r"GEN_TEXT:(.*)", out)
    rec("CHAT_GT_%d" % (i + 1), rc == 0 and g and g.group(1).strip(),
        "%ds %s" % (dt, g.group(1).strip()[:16] if g else "-"))

for i, q in enumerate(["Textadept 是什么",
                       "介绍一下 Textadept 编辑器",
                       "Textadept 编辑器用 Lua 扩展"]):
    out, rc, dt = run([EXE, W, TOK, "--ask", q, "32"])
    h = re.search(r"KB_HITS:(\d+)", out)
    g = re.search(r"GEN_TEXT:(.*)", out)
    rec("CHAT_RAG_%d" % (i + 1),
        rc == 0 and h and int(h.group(1)) > 0 and g,
        "%ds hits=%s %s" % (dt, h.group(1) if h else "-",
                            g.group(1).strip()[:16] if g else "-"))

for i, q in enumerate(["M11 步长调度 复现判决",
                       "微核 M11 步长调度复现怎么判的",
                       "microcore M11 步长调度 复现失败"]):
    out, rc, dt = run([EXE, W, TOK, "--ask", q, "32"])
    hm = "HONEST_MISS:1" in out
    rec("CHAT_REJECT_%d" % (i + 1), rc == 0 and hm and "GEN_TEXT" not in out,
        "%ds honest=%s gen_absent=%s" % (dt, hm, "GEN_TEXT" not in out))

# ============ B. 纯 token 生成 x3 + 确定性 ============
print("== B 生成: 固定 ids + 确定性 ==", flush=True)
IDS = "128000,1502,25,74662,103054,128009,72803,25,220"
ids_seen = set()
for i in range(3):
    out, rc, dt = run([EXE, W, "--gen", IDS, "32"])
    m = re.search(r"([\d.]+) tok/s", out)
    ids = re.search(r"GEN_IDS:(.*)", out)
    if ids:
        ids_seen.add(ids.group(1).strip())
    rec("GEN_%d" % (i + 1), rc == 0 and m and ids, "%ds %s tok/s" % (dt, m.group(1) if m else "-"))
rec("GEN_DETERMINISM", len(ids_seen) == 1, "3 遍输出逐位一致=%s" % (len(ids_seen) == 1))

# ============ C. 分词往返 x3 ============
print("== C 分词器往返 (空格分隔 ids) ==", flush=True)
for i in range(3):
    out, rc, dt = run([EXE, TOK, "--enc", "hello world 你好"])
    m = re.search(r"ENC_IDS:(.*)", out)
    ok = rc == 0 and m
    if ok:
        out2, rc2, dt2 = run([EXE, TOK, "--dec", m.group(1).strip()])
        ok = rc2 == 0 and "DEC_TEXT:hello world 你好" in out2
    rec("TOKRT_%d" % (i + 1), ok, "%.1fs" % (dt + dt2 if ok else dt))

# ============ D. 插件五件套 ============
print("== D 插件: demo / kb.search / web.search / kb.add / web.ingest ==", flush=True)
for i in range(3):
    out, rc, dt = run([EXE, MINI, "--cap", "demo.time", "ping"])
    rec("P_DEMO_%d" % (i + 1), rc == 0 and '"cap": "demo.time"' in out, "%ds" % dt)

for i, q in enumerate(["文艺复兴", "BitNet 三值权重", "ZZZNOSUCHMARK999"]):
    out, rc, dt = run([EXE, MINI, "--cap", "kb.search", q, "3"])
    exp = (q != "ZZZNOSUCHMARK999")   # 前两个在库, 第三个诚实空
    got = '"results": [{' in out
    rec("P_KBSEARCH_%d" % (i + 1), rc == 0 and got == exp, "%ds %s hit=%s" % (dt, q[:10], got))

for i, q in enumerate(["BitNet b1.58 inference", "SpaceX 星舰 最新试飞", "Python 3.13 新特性"]):
    out, rc, dt = run([EXE, MINI, "--cap", "web.search", q, "3"])
    rec("P_WEBSEARCH_%d" % (i + 1), rc == 0 and '"results": [{' in out, "%ds %s" % (dt, q[:10]))

# kb.add x3: 每遍唯一标记文档 (md/html/docx)
import zipfile
os.makedirs(r"H:\AimCode\aimp\testdocs", exist_ok=True)
for i, (fn, kind) in enumerate([("final_a.md", "Markdown"), ("final_b.html", "HTML"), ("final_c.docx", "DOCX")]):
    path = os.path.join(r"H:\AimCode\aimp\testdocs", fn)
    body = "%s 验收文档 %s 格式第 %d 遍。" % (MARK, kind, i + 1)
    if kind == "Markdown":
        open(path, "w", encoding="utf-8").write("# %s\n\n%s\n" % (MARK, body))
    elif kind == "HTML":
        open(path, "w", encoding="utf-8").write("<html><body><p>%s %s</p></body></html>" % (MARK, body))
    else:
        xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
               '<w:body><w:p><w:r><w:t>%s %s</w:t></w:r></w:p></w:body></w:document>' % (MARK, body))
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("word/document.xml", xml)
    out, rc, dt = run([EXE, MINI, "--cap", "kb.add", path])
    m = re.search(r'"added":\s*(\d+)', out)
    s = '"store": "dyn"' in out
    rec("P_KBADD_%s" % kind, rc == 0 and m and int(m.group(1)) == 1 and s,
        "%ds added=%s dyn=%s" % (dt, m.group(1) if m else "-", s))

# 入库即查回 (唯一标记)
out, rc, dt = run([EXE, MINI, "--cap", "kb.search", MARK, "3"])
rec("P_KBADD_SEARCHBACK", rc == 0 and out.count('"id": "kb-') >= 3, "%ds hits=%d" % (dt, out.count('"id": "kb-')))

# web.ingest x3 (不同查询, topn=1 轻量; 判据 crawled>=1, added 可为 0=已在库)
for i, q in enumerate(["BitNet b1.58 推理", "Textadept editor", "线性注意力 大模型"]):
    out, rc, dt = run([EXE, MINI, "--cap", "web.ingest", q, "1"], timeout=600)
    cw = re.search(r'"crawled":\s*(\d+)', out)
    rec("P_WEBINGEST_%d" % (i + 1), rc == 0 and cw and int(cw.group(1)) >= 1,
        "%ds crawled=%s %s" % (dt, cw.group(1) if cw else "-", q[:10]))

# ============ E. 双库 + 换库 ============
print("== E 双库合并 / 整包换库 ==", flush=True)
for i, (q, exp) in enumerate([("文艺复兴", "base"), ("BitNet 三值权重", "dyn"),
                              ("BitNet CPU inference", "dyn")]):
    out, rc, dt = run([EXE, MINI, "--cap", "kb.search", q, "3"])
    st = sorted(set(re.findall(r'"store": "(\w+)"', out)))
    rec("STORE_MERGE_%d" % (i + 1), rc == 0 and st == [exp], "%ds stores=%s" % (dt, st))

os.makedirs(ALT_DYN, exist_ok=True)
for i in range(3):
    mk = "%s_SWAP%d" % (MARK, i)
    doc = os.path.join(r"H:\AimCode\aimp\testdocs", "swap_%d.md" % i)
    open(doc, "w", encoding="utf-8").write("# %s\n\n%s 备用动态库专属内容。\n" % (mk, mk))
    out, rc, dt = run([EXE, MINI, "--cap", "kb.add", doc], {"AIM_KB_DYN_ROOT": ALT_DYN})
    m = re.search(r'"added":\s*(\d+)', out)
    rec("SWAP_ADD_%d" % (i + 1), rc == 0 and m and int(m.group(1)) == 1,
        "%ds added=%s" % (dt, m.group(1) if m else "-"))
    out, rc, dt = run([EXE, MINI, "--cap", "kb.search", mk, "3"], {"AIM_KB_DYN_ROOT": ALT_DYN})
    rec("SWAP_ON_%d" % (i + 1), rc == 0 and '"results": [{' in out, "%ds 换库可检索" % dt)
    out, rc, dt = run([EXE, MINI, "--cap", "kb.search", mk, "3"])
    rec("SWAP_OFF_%d" % (i + 1), rc == 0 and '"results": []' in out, "%ds 库隔离" % dt)

print("\n=== FINAL: %d/%d PASS ===" % (npass, total), flush=True)
sys.exit(0 if npass == total else 1)
