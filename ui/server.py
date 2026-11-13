# -*- coding: utf-8 -*-
"""aimp 加载器本地 UI 服务 (纯 stdlib, 绑定 127.0.0.1)
对话(auto/direct/kb) / 知识库检索 / 搜索引擎 / 文档入库 / 网页入库 / 库统计
启动: python ui/server.py   ->  http://127.0.0.1:8600"""
import io, json, os, re, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = r"H:\AimCode\aimp"
EXE = os.path.join(BASE, "build", "aimp_probe.exe")
W = os.path.join(BASE, "data", "bitnet2b", "m2b_tern.aimw")
TOKF = os.path.join(BASE, "data", "m2b_llama3_head.gguf")
MINI = os.path.join(BASE, "build", "mini_llama.gguf")
UPLOADS = os.path.join(BASE, "testdocs", "uploads")
LOG_PATH = os.path.join(BASE, "ui", "ui_log.jsonl")
PORT = 8600


def logj(entry):
    """运行日志: 每次请求一行 JSONL (时间/端点/问题/路径/轨迹/耗时)"""
    try:
        entry["ts"] = time.strftime("%m-%d %H:%M:%S")
        with io.open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from kb.fts_index import FtsIndex

KB_BASE = os.environ.get("AIM_KB_BASE", r"H:\AimCode\lab\cpu-native\kb_store")
KB_DYN = os.environ.get("AIM_KB_DYN_ROOT", r"H:\AimCode\lab\cpu-native\kb_dyn_store")

GEN_LOCK = threading.Lock()   # 模型同一时刻只进一个生成请求
# 推理引擎线程池: 本机实测 W2 甜点 (串行 1.99 -> W2 3.20 tok/s, W4 2.63 反降);
# 引擎内有自校准护栏 (并行不显著更快自动降级串行), 沙箱/限额机器不受影响
os.environ.setdefault("AIM_WORKERS", "2")


def run_exe(args, timeout=420):
    t0 = time.time()
    p = subprocess.run(args, cwd=BASE, capture_output=True, timeout=timeout)
    return p.stdout.decode("utf-8", "ignore"), p.returncode, time.time() - t0


def parse_gen(out):
    g = re.search(r"GEN_TEXT:(.*)", out, re.S)
    t = re.search(r"TIME ([\d.]+)s\s+([\d.]+) tok/s", out)
    p = re.search(r"PROMPT_TOKS:(\d+)", out)
    h = re.search(r"KB_HITS:(\d+)", out)
    return {
        "answer": g.group(1).strip() if g else "",
        "time_s": float(t.group(1)) if t else 0,
        "toks": float(t.group(2)) if t else 0,
        "prompt_toks": int(p.group(1)) if p else 0,
        "kb_hits": int(h.group(1)) if h else None,
        "honest_miss": "HONEST_MISS:1" in out,
    }


def cap_json(out):
    i = out.find("{")
    return json.loads(out[i:]) if i >= 0 else {}


def kb_search(q, topk=4):
    out, rc, dt = run_exe([EXE, MINI, "--cap", "kb.search", q, str(topk)], 120)
    if rc:
        return None, dt
    j = cap_json(out)
    j["_bus_s"] = round(dt, 2)
    return j, dt


RE_WEB = re.compile(r"搜索|搜一下|搜一搜|搜搜|查一下|查查|帮我查|上网查|新闻|最新消息|如今的?情况|现在的?情况")
RE_ING = re.compile(r"入库|爬下来|抓下来|爬进库|收进库|存进库")
RE_CLEAN = re.compile(
    r"^(帮我|请|麻烦|劳驾)?(在网上|网上|用搜索)?(搜索|搜一下|搜一搜|查一下|查查|帮我查)(一下|一下下)?(，|,|:|：)?\s*")
# 闲聊白名单: 只有这些才允许纯对话 (其它一切 auto 问题查库不到就走联网)
RE_CHAT = re.compile(
    r"^(在吗|你好|您好|嗨+|哈喽|hello|hi|谢谢|多谢|辛苦|再见|拜拜|晚安|你是谁|你叫什么|你是干什么的|你能做什么)[!！。？?~，,\s]*$",
    re.I)


def agent_web(q):
    """联网全链: 搜索引擎 -> 抓取过闸入库 -> 查库 -> 引答 (每步落 trace)"""
    trace = []
    clean = RE_CLEAN.sub("", q).strip() or q
    out, rc, dt = run_exe([EXE, MINI, "--cap", "web.search", clean, "4"], 120)
    serp = cap_json(out).get("results", []) if rc == 0 else []
    trace.append({"title": "🌐 搜索引擎", "summary": "%d 条结果" % len(serp),
                  "detail": " | ".join((h.get("title") or "")[:40] for h in serp[:4])})
    out, rc, dt = run_exe([EXE, MINI, "--cap", "web.ingest", clean, "2"], 600)
    ing = cap_json(out)
    trace.append({"title": "🕸 抓取过闸", "summary": "抓 %s 页, 拒 %s 页, 入库 +%s 卡"
                  % (ing.get("crawled"), ing.get("rejected"), ing.get("fts_indexed")),
                  "detail": " ; ".join("%s %s" % (d.get("url", "")[:44],
                             d.get("status") or ("+%s" % d.get("added", 0)))
                             for d in ing.get("details", []))})
    kb_j, _ = kb_search(clean, 4)
    hits = kb_j.get("results") if kb_j else []
    trace.append({"title": "📚 知识库", "summary": "%d 张卡命中" % len(hits),
                  "detail": ""})
    if not hits:
        # 摘要兜底: SERP snippet 是搜索页合法可见文本, 作参考喂模型 (标注来源等级)
        snips = [h for h in serp if h.get("snippet")][:4]
        if snips:
            ref = "\n".join("[%d] %s\n%s" % (i + 1, h["title"][:60],
                                            h["snippet"][:150])
                            for i, h in enumerate(snips))
            with GEN_LOCK:
                out, rc, dt = run_exe([EXE, W, TOKF, "--gt",
                    "以下是网络搜索摘要:\n%s\n问题: %s" % (ref, clean), "32"], 420)
            if rc:
                return {"ok": False, "error": "gen rc=%d" % rc}
            info = parse_gen(out)
            trace.append({"title": "📝 搜索摘要兜底",
                          "summary": "正文爬不到, %d 条搜索摘要作参考 (来源等级低于全文)" % len(snips),
                          "detail": ""})
            return {"ok": True, "path": "digest", "trace": trace,
                    "sources": [{"title": h["title"], "url": h["url"],
                                 "snippet": h["snippet"][:150],
                                 "store": "serp", "origin": "serp"} for h in snips],
                    "answer": info["answer"], "time_s": info["time_s"],
                    "toks": info["toks"], "prompt_toks": info["prompt_toks"],
                    "honest_miss": False}
        return {"ok": True, "path": "honest", "trace": trace, "sources": [],
                "answer": "[联网失败] 搜索到了 %d 条结果，但目标站不可达/被 robots 拒抓/未过质量闸，连搜索摘要都没拿到。如实告知：现在答不了，不编造。" % len(serp),
                "honest_miss": True, "time_s": 0, "toks": 0}
    with GEN_LOCK:
        out, rc, dt = run_exe([EXE, W, TOKF, "--ask", clean, "32"], 420)
    if rc:
        return {"ok": False, "error": "gen rc=%d" % rc}
    info = parse_gen(out)
    return {"ok": True, "path": "web", "trace": trace, "sources": hits,
            "answer": info["answer"], "time_s": info["time_s"],
            "toks": info["toks"], "prompt_toks": info["prompt_toks"],
            "honest_miss": info["honest_miss"]}


def agent_ingest(q):
    """网页入库流: 搜索 -> 抓取过闸 -> 查回报数"""
    trace = []
    clean = RE_CLEAN.sub("", q).strip() or q
    out, rc, dt = run_exe([EXE, MINI, "--cap", "web.ingest", clean, "2"], 600)
    ing = cap_json(out)
    trace.append({"title": "🕸 抓取过闸", "summary": "抓 %s 页入库, 拒 %s 页, FTS +%s 卡"
                  % (ing.get("crawled"), ing.get("rejected"), ing.get("fts_indexed")),
                  "detail": " ; ".join("%s %s" % (d.get("url", "")[:44],
                             d.get("status") or ("+%s" % d.get("added", 0)))
                             for d in ing.get("details", []))})
    kb_j, _ = kb_search(clean, 3)
    hits = kb_j.get("results") if kb_j else []
    trace.append({"title": "📚 查回验证", "summary": "%d 张卡可检索" % len(hits),
                  "detail": ""})
    return {"ok": True, "path": "ingest", "trace": trace, "sources": hits,
            "answer": "入库完成：本次抓取 %s 页入库、%s 页被拒（robots/质量闸），动态库新增 %s 卡，已在上方列出可查回的卡。"
                      % (ing.get("crawled"), ing.get("rejected"), ing.get("fts_indexed")),
            "honest_miss": False, "time_s": 0, "toks": 0}


def api_chat(q, mode):
    # 意图路由 (确定性规则; 2B 模型不做 function calling)
    if RE_ING.search(q):
        return agent_ingest(q)
    if RE_WEB.search(q):
        return agent_web(q)
    # 闲聊白名单: 任何模式都直答 (在吗/你好这种永远不该吃"知识库不知道")
    if mode == "direct" or RE_CHAT.match(q.strip()):
        # 闲聊白名单 / 用户点名纯对话 -> 直答
        with GEN_LOCK:
            out, rc, dt = run_exe([EXE, W, TOKF, "--gt", q, "48"], 420)
        if rc:
            return {"ok": False, "error": "gen rc=%d" % rc}
        info = parse_gen(out)
        return {"ok": True, "path": "direct", "answer": info["answer"],
                "sources": [], "time_s": info["time_s"], "toks": info["toks"],
                "prompt_toks": info["prompt_toks"], "honest_miss": False}
    # auto / kb: 先查库
    sources, path = [], ""
    kb_j, t1 = kb_search(q, 4)
    if kb_j and kb_j.get("results"):
        sources = kb_j["results"]
    if mode == "kb" and not sources:
        return {"ok": True, "path": "honest", "answer":
                "[未检索到] 知识库中没有与问题相关的内容，如实告知：不知道。",
                "sources": [], "honest_miss": True, "time_s": round(t1, 2)}
    if sources:
        with GEN_LOCK:
            out, rc, dt = run_exe([EXE, W, TOKF, "--ask", q, "48"], 420)
        if rc:
            return {"ok": False, "error": "gen rc=%d" % rc}
        info = parse_gen(out)
        path = "kb"
        if info["honest_miss"]:
            path = "honest"
            info["answer"] = ("[未检索到] 知识库中没有相关内容，如实告知：不知道。"
                              if mode == "kb" else info["answer"])
        return {"ok": True, "path": path, "answer": info["answer"],
                "sources": sources, "time_s": round(info["time_s"], 2),
                "toks": info["toks"], "prompt_toks": info["prompt_toks"],
                "kb_hits": info["kb_hits"], "honest_miss": info["honest_miss"]}
    # auto: 库里没有 -> 联网全链 (搜->爬->查库->引答/摘要/诚实), 不给纯对话编造的机会
    return agent_web(q)


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>aimp 加载器</title><style>
:root{--bg:#0f1117;--panel:#171a23;--line:#262b38;--tx:#dbe2f0;--dim:#8b94a8;
--acc:#5b8cff;--ok:#3fbf7f;--warn:#e0a83f;--bad:#e05c5c}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font:14px/1.6 "Segoe UI",system-ui,sans-serif;display:flex;height:100vh}
nav{width:220px;background:var(--panel);border-right:1px solid var(--line);padding:16px;display:flex;flex-direction:column;gap:6px}
nav h1{font-size:15px;margin-bottom:12px;color:var(--acc)}
nav button{background:none;border:1px solid var(--line);color:var(--tx);padding:9px 12px;border-radius:8px;text-align:left;cursor:pointer;font-size:13px}
nav button:hover,nav button.on{border-color:var(--acc);color:var(--acc)}
nav .foot{margin-top:auto;font-size:11px;color:var(--dim)}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
header .st{font-size:12px;color:var(--dim)}
#view{flex:1;overflow-y:auto;padding:20px}
#barslot{flex-shrink:0;padding:0 20px 16px}
.mode{display:flex;gap:8px;margin-bottom:14px;align-items:center;flex-wrap:wrap}
.mode button{border:1px solid var(--line);background:none;color:var(--tx);padding:5px 12px;border-radius:14px;cursor:pointer;font-size:12px}
.mode button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.mode .hint{font-size:12px;color:var(--dim);flex-basis:100%}
.bar{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap}
.bar textarea{flex:1 1 220px;min-width:200px;background:var(--panel);border:1px solid var(--line);color:var(--tx);border-radius:10px;padding:10px 14px;resize:none;height:64px;font:inherit}
.bar button{background:var(--acc);border:0;color:#fff;border-radius:10px;padding:0 22px;cursor:pointer;font-size:14px;flex:0 0 auto;height:44px}
.bar button:disabled{opacity:.5}
#capsel{flex:0 0 auto;background:var(--panel);border:1px solid var(--line);color:var(--tx);border-radius:10px;padding:0 8px;height:44px;font:inherit;font-size:12px;width:130px}
#capgo{flex:0 0 auto;background:var(--panel);border:1px solid var(--acc);color:var(--acc);border-radius:10px;padding:0 14px;cursor:pointer;font-size:13px;height:44px}
.bub{max-width:88%;margin:10px 0;padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
.bub.u{background:#233052;margin-left:auto}
.bub.a{background:var(--panel);border:1px solid var(--line)}
.meta{font-size:11px;color:var(--dim);margin-top:6px}
.src{border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:8px 10px;margin:6px 0;font-size:12px;cursor:pointer}
.src .t{color:var(--acc)}.src .x{display:none;color:var(--dim);margin-top:4px}
.src.open .x{display:block}
.tag{display:inline-block;font-size:10px;padding:1px 7px;border-radius:8px;margin-right:6px;vertical-align:1px}
.tag.base{background:#2a3b2e;color:#7fd79f}.tag.dyn{background:#3b2f1e;color:#e8b46a}
.spin{color:var(--warn);font-size:12px}
input[type=text],input[type=number]{background:var(--panel);border:1px solid var(--line);color:var(--tx);border-radius:8px;padding:8px 12px;font:inherit}
.row{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin:8px 0;font-size:12px}
.card b{color:var(--acc)}
pre{white-space:pre-wrap;word-break:break-word;color:var(--tx)}
</style></head><body>
<nav>
<h1>⚡ aimp 加载器</h1>
<button data-v=chat class=on>💬 对话</button>
<button data-v=kb>📚 知识库检索</button>
<button data-v=web>🌐 搜索引擎</button>
<button data-v=add>📄 文档入库</button>
<button data-v=ingest>🕸 网页入库</button>
<button data-v=stats>📊 库存统计</button>
<button data-v=caps>🔌 插件总线</button>
<button data-v=logs>📜 运行日志</button>
<div class=foot>自研 C 引擎 + 插件总线<br>AIMW 三值权重 · 本地 127.0.0.1</div>
</nav>
<main><header><b id=title>💬 对话</b><span class=st id=hstat></span></header>
<div id=view></div>
<div id=barslot></div></main>
<script>
let VIEW='chat',MODE='auto',BUSY=false;
const $=s=>document.querySelector(s);
const views={chat:'💬 对话',kb:'📚 知识库检索',web:'🌐 搜索引擎',add:'📄 文档入库',ingest:'🕸 网页入库',caps:'🔌 插件总线',stats:'📊 库存统计',logs:'📜 运行日志'};
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{switchView(b.dataset.v)});
function switchView(v){VIEW=v;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
$('#title').textContent=views[v];$('#view').innerHTML='';renderBar(v);
if(v==='stats')loadStats();
if(v==='caps')loadCaps();
if(v==='logs')loadLogs();}
async function loadLogs(){
 const r=await api('/api/logs',{});
 const rows=(r.entries||[]).map(e=>{
  const tr=(e.trace||[]).map(t=>' ⚙'+t[0]+':'+t[1]).join('');
  return `<div class=card><b>${esc(e.ts||'')}</b> [${esc(e.api||'')}] ${esc(e.q||e.file||'')}<br>
  <span class=meta>${e.path?('路径='+e.path+' '):''}${e.hits!=null?('命中='+e.hits+' '):''}${e.added!=null?('入库='+e.added+' '):''}${e.crawled!=null?('抓取='+e.crawled+' '):''}${e.honest?'HONEST ':''}耗时=${e.secs}s${e.error?(' 错误='+esc(e.error)):''}${tr}</span></div>`}).join('');
 $('#view').innerHTML=rows||'<div class=card>还没有日志 — 去对话/检索/入库跑一轮再回来看</div>'}
async function loadCaps(){
 const r=await fetch('/api/caps').then(x=>x.json());
 if(r.error){$('#view').innerHTML='<div class=card>读 caps.json 失败: '+esc(r.error)+'</div>';return}
 $('#view').innerHTML='<div class=card><b>插件总线</b> = C 引擎读 plugins/caps.json 注册表直调 Python 插件 (--cap, 不加载模型秒回)。<br>新插件两步: ① plugins/ 下写 py (argv[1] 收查询 JSON 文件, stdout 吐 {"cap":...}) ② caps.json 登记一行</div>'+
  r.caps.map(c=>`<div class=card><b>🔌 ${esc(c.name)}</b><div class=meta>${esc(c.cmd)}</div>
  <div class=bar style=margin-top:8px><input type=text data-q="${esc(c.name)}" placeholder='查询内容 (装进总线 JSON 信封的 "query" 字段)' style=flex:1>
  <input type=number data-k="${esc(c.name)}" value=5 min=1 max=20 style=width:64px>
  <button data-cap="${esc(c.name)}">调用</button></div>
  <div class=out data-out="${esc(c.name)}"></div></div>`).join('');
 document.querySelectorAll('button[data-cap]').forEach(b=>b.onclick=async()=>{
  if(BUSY)return;BUSY=true;
  const cap=b.dataset.cap,out=document.querySelector('[data-out="'+cap+'"]');
  const q=document.querySelector('[data-q="'+cap+'"]').value.trim();
  const k=+document.querySelector('[data-k="'+cap+'"]').value||5;
  out.innerHTML='<span class=spin>总线调用中…</span>';
  try{const r=await api('/api/capcall',{cap,q,topk:k});
   out.innerHTML='<div class=meta>耗时 '+r._bus_s+'s</div><pre style="white-space:pre-wrap;font-size:12px;color:#9ab;margin:4px 0 0">'+esc(JSON.stringify(r,null,1))+'</pre>'}
  catch(e){out.textContent='调用失败: '+e}
  BUSY=false});
}
function bubble(cls,html){const d=document.createElement('div');d.className='bub '+cls;d.innerHTML=html;
$('#view').appendChild(d);$('#view').scrollTop=1e9;return d}
function barHTML(kind){
 if(kind==='chat')return `<div class=mode>
  <button data-m=auto class=on>自动</button><button data-m=kb>知识库优先</button><button data-m=direct>纯对话</button>
  <span class=hint>自动=闲聊直答; 知识题查库引答; 查不到自动联网(搜→爬→答); 全灭如实拒答, 绝不编</span></div>
  <div class=bar><textarea id=inp placeholder="问点什么… 试试: 帮我搜索一下 Textadept"></textarea><select id=capsel></select><button id=capgo title="用选中插件直调输入框内容 (不加载模型)">🔌直调</button><button id=go>发送</button></div>`;
 if(kind==='kb'||kind==='web')return `<div class=bar><input type=text id=inp style="flex:1" placeholder="${kind==='kb'?'在两个库中检索…':'搜索引擎查询…'}">
  <input type=number id=k value=5 min=1 max=10 style="width:64px"><button id=go>查询</button></div>`;
 if(kind==='add')return `<div class=bar><input type=file id=f style="flex:1">
  <button id=go>入库 (txt/md/html/docx/pdf)</button></div>`;
 if(kind==='ingest')return `<div class=bar><input type=text id=inp style="flex:1" placeholder="主题, 搜到就爬, 过质量闸入库">
  <input type=number id=k value=2 min=1 max=3 style="width:64px"><button id=go>爬取入库</button></div>`;
 return ''}
function renderBar(v){$('#barslot').innerHTML=barHTML(v);bindBar(v)}
function bindBar(v){
 document.querySelectorAll('.mode button').forEach(b=>b.onclick=()=>{MODE=b.dataset.m;
  document.querySelectorAll('.mode button').forEach(x=>x.classList.toggle('on',x===b))});
 const go=$('#go'),inp=$('#inp');if(!go)return;
 go.onclick=()=>{if(BUSY)return;doAction(v)};
 if(inp)inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();if(!BUSY)doAction(v)}});
 if(v==='chat'){fetch('/api/caps').then(x=>x.json()).then(r=>{const s=$('#capsel');
  if(s)(r.caps||[]).forEach(c=>{const o=document.createElement('option');o.value=c.name;o.textContent=c.name;s.appendChild(o)})}).catch(()=>{});
  const cg=$('#capgo');if(cg)cg.onclick=()=>{if(!BUSY)capCall()}}}
async function capCall(){
 const inp=$('#inp'),cg=$('#capgo'),sel=$('#capsel');if(!inp||!sel)return;
 const q=inp.value.trim();if(!q)return;
 const cap=sel.value;BUSY=true;if(cg)cg.disabled=true;
 bubble('u','🔌 <b>'+esc(cap)+'</b> '+esc(q.replace(/</g,'&lt;')));
 const sp=bubble('a','<span class=spin>总线调用中…</span>');
 try{const r=await api('/api/capcall',{cap,q,topk:5});sp.remove();
  const res=r.results||[];
  bubble('a',(res.length?res.map(srcHTML).join(''):'<span class=meta>无 results 字段, 原始返回:</span><pre style="white-space:pre-wrap;font-size:12px;color:#9ab">'+esc(JSON.stringify(r,null,1))+'</pre>')+
   '<div class=meta>[插件直调 '+esc(cap)+'] 总线 '+r._bus_s+'s</div>')}
 catch(e){sp.remove();bubble('a','调用失败: '+e)}
 BUSY=false;if(cg)cg.disabled=false}
async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json()}
async function doAction(v){
 const inp=$('#inp'),go=$('#go');BUSY=true;go.disabled=true;
 try{
  if(v==='chat'){const q=inp.value.trim();if(!q)return reset();
   bubble('u',q.replace(/</g,'&lt;'));
   const sp=bubble('a','<span class=spin>路由中… (联网全链可达 1-3 分钟, 本机 CPU 推理)</span>');
   const r=await api('/api/chat',{q,mode:MODE});sp.remove();renderChat(r);}
  if(v==='kb'||v==='web'){const q=inp.value.trim();const k=+$('#k').value||5;if(!q)return reset();
   const sp=bubble('a','<span class=spin>查询中…</span>');
   const r=await api('/api/'+(v==='kb'?'kbsearch':'websearch'),{q,topk:k});sp.remove();renderSearch(r);}
  if(v==='add'){const f=$('#f').files[0];if(!f)return reset();
   const sp=bubble('a','<span class=spin>解析+入库中…</span>');
   const buf=await f.arrayBuffer();
   const r=await fetch('/api/kbadd',{method:'POST',headers:{'X-Filename':encodeURIComponent(f.name)},body:buf});
   const j=await r.json();sp.remove();renderAdd(j);}
  if(v==='ingest'){const q=inp.value.trim();const n=+$('#k').value||2;if(!q)return reset();
   const sp=bubble('a','<span class=spin>搜索→抓取→过闸→入库… (可达 1-2 分钟)</span>');
   const r=await api('/api/webingest',{q,topn:n});sp.remove();renderIngest(r);}
 }catch(e){bubble('a','请求失败: '+e)}reset()}
function reset(){BUSY=false;const g=$('#go');if(g)g.disabled=false}
function esc(s){return (s||'').replace(/</g,'&lt;')}
function srcHTML(s){return `<div class=src onclick=this.classList.toggle('open')>
 <span class="tag ${s.store||'dyn'}">${s.store||'?'}·${esc(s.origin)||'?'}</span>
 <span class=t>${esc((s.title||'').slice(0,46))}</span> <span style=float:right>分 ${s.score!=null?s.score:''}</span>
 <div class=x>${esc((s.text||s.snippet||'').slice(0,300))}</div></div>`}
function renderChat(r){
 if(!r.ok){bubble('a','出错: '+esc(r.error));return}
 const pathName={direct:'纯对话',kb:'知识库引答',honest:'诚实拒答',web:'联网全文引答',digest:'搜索摘要引答',ingest:'网页入库'}[r.path]||r.path;
 const tr=(r.trace||[]).map(t=>`<div class=src onclick=this.classList.toggle('open')>⚙ ${esc(t.title)} — ${esc(t.summary)}${t.detail?`<div class=x>${esc(t.detail)}</div>`:''}</div>`).join('');
 const src=r.sources&&r.sources.length?('<div class=meta>参考 '+r.sources.length+' 张卡:</div>'+r.sources.map(srcHTML).join('')):'';
 bubble('a',`${tr}${esc(r.answer)}${src}<div class=meta>[${pathName}] ${r.time_s!=null?('生成 '+r.time_s+'s'):''}${r.toks!=null?('· '+r.toks+' tok/s'):''}${r.prompt_toks!=null?('· prompt '+r.prompt_toks+' tok'):''}${r.honest_miss?'· HONEST':''}</div>`)}
function renderSearch(r){
 if(r.error||!r.results){bubble('a','出错或无结果');return}
 bubble('a',(r.results.map(srcHTML).join('')||'<span class=meta>两库均无结果, 如实返回空</span>')+
  `<div class=meta>base ${r.cards.base} 卡 + dyn ${r.cards.dyn} 卡 · ${r._bus_s}s</div>`)}
function renderAdd(j){
 bubble('a',`<b>${esc(j.file||j.error)}</b><br>入库 ${j.added} 张, 查重拒 ${j.rejected} 张, FTS 索引 ${j.fts_indexed}<br>
 <span class=meta>动态库现有 ${j.dyn_total} 卡 · ${j.store}</span>`)}
function renderIngest(j){
 const det=(j.details||[]).map(d=>`<div class=meta>${esc(d.url)} — ${d.added!=null?('入库 '+d.added+' / 拒 '+d.gated+' / 质量 '+d.quality):esc(d.status)}</div>`).join('');
 bubble('a',`<b>${esc(j.query)}</b><br>抓取 ${j.crawled} 页入库, ${j.rejected} 页被拒, FTS +${j.fts_indexed}${det}`)}
async function loadStats(){
 const r=await api('/api/stats',{});
 $('#view').innerHTML=`<div class=card><b>固定底座</b> (wiki, 不可换)<br>${r.base} 卡<br>路径 ${esc(r.base_path)}</div>
 <div class=card><b>动态库</b> (user+crawl, 可整包更换)<br>${r.dyn} 卡<br>路径 ${esc(r.dyn_path)}</div>
 <div class=card><b>模型</b><br>BitNet b1.58 2B · AIM-W 0.87GB · 串行内核 ${r.tokps} tok/s (沙箱实测偏低, 亲手 CMD ~2.3)</div>`}
switchView('chat');
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (str, bytes)):
            body = json.dumps(body, ensure_ascii=False)   # dict 直传也安全
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/logs":
            entries = []
            try:
                with io.open(LOG_PATH, encoding="utf-8") as f:
                    entries = [json.loads(l) for l in f if l.strip()][-60:]
            except Exception:
                pass
            self._send(200, json.dumps({"entries": list(reversed(entries))},
                                       ensure_ascii=False))
        elif self.path == "/api/caps":
            # 插件总线注册表 (caps.json 原样透出)
            caps = []
            try:
                with io.open(os.path.join(BASE, "plugins", "caps.json"),
                             encoding="utf-8") as f:
                    caps = [{"name": k, "cmd": v}
                            for k, v in json.load(f).items()]
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)[:120]},
                                           ensure_ascii=False))
                return
            self._send(200, json.dumps({"caps": caps}, ensure_ascii=False))
        else:
            self._send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        t0 = time.time()
        try:
            if self.path == "/api/chat":
                j = json.loads(raw.decode("utf-8"))
                r = api_chat(j["q"], j.get("mode", "auto"))
                logj({"api": "chat", "q": j["q"][:80], "mode": j.get("mode", "auto"),
                      "path": r.get("path"), "honest": r.get("honest_miss", False),
                      "trace": [(t.get("title", ""), t.get("summary", "")) for t in r.get("trace", [])],
                      "sources": len(r.get("sources", [])),
                      "secs": round(time.time() - t0, 1), "ok": r.get("ok", False),
                      "error": r.get("error", "")})
                self._send(200, json.dumps(r, ensure_ascii=False))
            elif self.path == "/api/kbsearch":
                j = json.loads(raw.decode("utf-8"))
                r, dt = kb_search(j["q"], int(j.get("topk", 5)))
                logj({"api": "kbsearch", "q": j["q"][:80],
                      "hits": len((r or {}).get("results", [])),
                      "secs": round(time.time() - t0, 1)})
                self._send(200, json.dumps(r or {"error": "kb.search fail"},
                                           ensure_ascii=False))
            elif self.path == "/api/websearch":
                j = json.loads(raw.decode("utf-8"))
                out, rc, dt = run_exe([EXE, MINI, "--cap", "web.search",
                                       j["q"], str(j.get("topk", 5))], 120)
                r = cap_json(out) if not rc else {"error": "web.search rc=%d" % rc}
                logj({"api": "websearch", "q": j["q"][:80],
                      "hits": len(r.get("results", [])),
                      "secs": round(time.time() - t0, 1)})
                self._send(200, json.dumps(r, ensure_ascii=False))
            elif self.path == "/api/kbadd":
                fn = urllib.parse.unquote(self.headers.get("X-Filename", "upload.bin"))
                os.makedirs(UPLOADS, exist_ok=True)
                path = os.path.join(UPLOADS, os.path.basename(fn))
                with open(path, "wb") as f:
                    f.write(raw)
                out, rc, dt = run_exe([EXE, MINI, "--cap", "kb.add", path], 300)
                r = cap_json(out) if not rc else {"error": "kb.add rc=%d" % rc}
                logj({"api": "kbadd", "file": os.path.basename(fn)[:60],
                      "added": r.get("added"), "rejected": r.get("rejected"),
                      "secs": round(time.time() - t0, 1)})
                self._send(200, json.dumps(r, ensure_ascii=False))
            elif self.path == "/api/webingest":
                j = json.loads(raw.decode("utf-8"))
                out, rc, dt = run_exe([EXE, MINI, "--cap", "web.ingest",
                                       j["q"], str(j.get("topn", 2))], 600)
                r = cap_json(out) if not rc else {"error": "web.ingest rc=%d" % rc}
                logj({"api": "webingest", "q": j["q"][:80],
                      "crawled": r.get("crawled"), "rejected": r.get("rejected"),
                      "fts": r.get("fts_indexed"),
                      "secs": round(time.time() - t0, 1)})
                self._send(200, json.dumps(r, ensure_ascii=False))
            elif self.path == "/api/capcall":
                # 插件直调: --cap 走 C 总线 (不加载模型); cap 名必须已在 caps.json 注册
                j = json.loads(raw.decode("utf-8"))
                cap = j.get("cap", "")
                reg = {}
                try:
                    with io.open(os.path.join(BASE, "plugins", "caps.json"),
                                 encoding="utf-8") as f:
                        reg = json.load(f)
                except Exception:
                    pass
                if cap not in reg:
                    self._send(200, json.dumps({"error": "未注册插件: %s" % cap},
                                               ensure_ascii=False))
                    return
                q = j.get("q", "")
                topk = int(j.get("topk", 5))
                out, rc, dt = run_exe([EXE, MINI, "--cap", cap, q, str(topk)], 300)
                r = cap_json(out) if rc == 0 else {"error": "bus rc=%d" % rc}
                r["_bus_s"] = round(dt, 2)
                logj({"api": "capcall", "cap": cap, "q": q[:80],
                      "secs": round(time.time() - t0, 1),
                      "ok": rc == 0, "error": r.get("error", "")})
                self._send(200, json.dumps(r, ensure_ascii=False))
            elif self.path == "/api/stats":
                b = FtsIndex(KB_BASE).count()
                d = FtsIndex(KB_DYN).count()
                self._send(200, json.dumps({
                    "base": b, "dyn": d, "base_path": KB_BASE, "dyn_path": KB_DYN,
                    "tokps": 2.3}, ensure_ascii=False))
            else:
                self._send(404, "{}")
        except Exception as e:
            logj({"api": self.path, "error": str(e)[:160],
                  "secs": round(time.time() - t0, 1)})
            self._send(200, json.dumps({"error": str(e)[:200]}, ensure_ascii=False))


if __name__ == "__main__":
    os.makedirs(UPLOADS, exist_ok=True)
    print("aimp UI http://127.0.0.1:%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
