# -*- coding: utf-8 -*-
# build_lang3_corpus.py — 纯中文版语言包：六语种稀释事故的修复
#   过滤律：回复含西里尔/阿拉伯/谚文/假名 → 拒；创意 _en 任务 → 拒；回复须含汉字
#   配方：emp2中×2 + creative中×1.5 + refusal中×1.5 + smalltalk×6 + 推理回放18%
import io, json, random, zlib, time, os, sys, re
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
SEED = 20261105
t0 = time.time()

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

def rng_of(name, extra=0):
    return random.Random((zlib.crc32(name.encode()) + SEED + extra) % (2**31))

# 非中文脚本黑名单（回复里出现即拒）
BAD = re.compile(u'[\u0400-\u04FF\u0600-\u06FF\uAC00-\uD7AF\u3040-\u30FF\u0980-\u09FF\u0900-\u097F]')
CJK = re.compile(u'[\u4E00-\u9FFF]')

def zh_ok(q, a):
    """中文过滤：q+a 无其他语种脚本，a 含足量汉字"""
    if BAD.search(q) or BAD.search(a):
        return False
    cjk = len(CJK.findall(a))
    return cjk >= 6 and cjk / max(len(a), 1) > 0.15

rows = []

# ---- 1) emp2 共情 ×2（只留中文）----
log("emp2 zh ...")
src = os.path.join(ROOT, "emp2_corpus", "emp2_corpus.jsonl")
base = []
n_all = 0
with io.open(src, encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        n_all += 1
        q = "[情绪:%s] %s" % (r.get("emotion", "一般"), r.get("user_situation", ""))
        a = r.get("resp", "")
        if q.strip() and a.strip() and zh_ok(q, a):
            base.append((q, a))
log("emp2: %d/%d zh (%.0fs)" % (len(base), n_all, time.time() - t0))
rng = rng_of("emp2zh")
pick = [base[rng.randrange(len(base))] for _ in range(2 * len(base))]
for q, a in pick:
    rows.append({"q": q, "a": a, "src": "lang_emp2", "split": "train"})

# ---- 2) creative ×1.5（去掉 _en 任务，只留中文）----
log("creative zh ...")
base = []
n_all = 0
with io.open(os.path.join(ROOT, "creative_corpus", "creative_v0.jsonl"), encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        task = r.get("task", "story")
        if task.endswith("_en"):
            continue
        n_all += 1
        q = "[创意:%s] %s" % (task, r.get("q", ""))
        a = r.get("ans", "")
        if q.strip() and a.strip() and zh_ok(q, a):
            base.append((q, a))
log("creative: %d/%d zh (%.0fs)" % (len(base), n_all, time.time() - t0))
rng = rng_of("creativezh")
pick = [base[rng.randrange(len(base))] for _ in range(int(1.5 * len(base)))]
for q, a in pick:
    rows.append({"q": q, "a": a, "src": "lang_creative", "split": "train"})

# ---- 3) refusal ×1.5（从 v2fmt 混料抽，只留中文）----
log("refusal zh ...")
refusal_base = []
with io.open(os.path.join(ROOT, "v2fmt_1p5b_full_mix.jsonl"), encoding="utf-8", errors="replace") as f:
    for line in f:
        if '"replay_refusal"' in line:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("src") == "replay_refusal" and r.get("q") and r.get("a") and zh_ok(r["q"], r["a"]):
                refusal_base.append((r["q"], r["a"]))
log("refusal: %d zh (%.0fs)" % (len(refusal_base), time.time() - t0))
rng = rng_of("refusalzh")
pick = [refusal_base[rng.randrange(len(refusal_base))] for _ in range(int(1.5 * len(refusal_base)))]
for q, a in pick:
    rows.append({"q": q, "a": a, "src": "lang_refusal", "split": "train"})

# ---- 4) 小对话 ×6（本来就是中文，加权防「在吗」）----
log("smalltalk x6 ...")
exec(open(os.path.join(ROOT, "build_lang1_corpus.py"), encoding="utf-8").read()
     .split("synth = []")[1].split("rng = rng_of(\"smalltalk_rep\")")[0]
     .replace("ri = rng_of(", "#ri = rng_of(")) if False else None
# 直接重写（避免 exec 脆弱）
HELLOS = ["在吗", "你好", "在不在", "嗨", "有人吗", "在么", "晚上好", "早上好",
          "下午好", "你好呀", "您好", "在吗在吗", "能听见吗", "测试", "你好！"]
REPLIES = ["在的，怎么了？", "你好！有什么可以帮你的？", "在呢，说吧。", "嗨！我在。",
           "在的，请讲。", "我在的，什么事？", "你好呀，今天怎么样？", "嗯，在。你说。"]
IDENT_Q = ["你是谁", "你叫什么名字", "你是什么", "你是机器人吗", "你会什么", "你能干什么",
           "你是AI吗", "介绍一下你自己"]
IDENT_A = ["我是 AIM，一个小型的字节级语言模型，可以陪你聊天、做推理题。", "我叫 AIM，会一点点推理和聊天。",
           "我是一个小语言模型 AIM，擅长按格式解题，也能简单聊聊。", "AIM，很小的模型，但很努力。"]
THANKS_Q = ["谢谢", "多谢", "谢啦", "辛苦了"]
THANKS_A = ["不客气！", "不用谢，应该的。", "小事一桩。", "客气啦，有需要随时说。"]
BYE_Q = ["再见", "拜拜", "晚安", "回见", "下次聊"]
BYE_A = ["再见！", "拜拜，有空再聊。", "晚安，好梦。", "好的，回见。"]
SMALL = [
    ("今天天气怎么样", "我没有实时天气数据，不过你那边要是出太阳，就适合出门走走。"),
    ("你聪明吗", "我是个小模型，推理还可以，聊天马马虎虎，算术不太行，诚实地说。"),
    ("你多大了", "我刚训练完不久，算是个新生儿。"),
    ("讲个笑话", "程序员最讨厌的两件事：别人不写注释，和让自己写注释。"),
    ("你会写诗吗", "会一点，你出个题目试试？"),
    ("帮我算个数", "小的算术我可以试试，多位数不一定准，你可以考考我。"),
    ("你吃饭了吗", "我不吃饭，吃电。谢谢关心！"),
    ("你现在在哪", "我跑在用户的机器上，没有具体位置，哪里有电哪里在。"),
    ("说句话听听", "好呀，听到请回复，收到请回答。"),
    ("你是中文模型吗", "是呀，我们说中文。"),
]
synth = []
ri = rng_of("hello_zh")
for q in HELLOS:
    synth.append((q, REPLIES[ri.randrange(len(REPLIES))]))
ri = rng_of("ident_zh")
for q in IDENT_Q:
    synth.append((q, IDENT_A[ri.randrange(len(IDENT_A))]))
ri = rng_of("thanks_zh")
for q in THANKS_Q:
    synth.append((q, THANKS_A[ri.randrange(len(THANKS_A))]))
ri = rng_of("bye_zh")
for q in BYE_Q:
    synth.append((q, BYE_A[ri.randrange(len(BYE_A))]))
synth.extend(SMALL)
rep = (6 * 24000) // max(1, len(synth)) + 1
for _ in range(rep):
    for q, a in synth:
        rows.append({"q": q, "a": a, "src": "lang_smalltalk", "split": "train"})
log("smalltalk: %d rows total=%d (%.0fs)" % (len(synth) * rep, len(rows), time.time() - t0))

# ---- 5) 推理回放 18%（防遗忘）----
REPLAY_SRCS = {"v2_csp", "v2_graph", "v2_logic", "v2_causal", "v2_fsm"}
log("replay index ...")
replay_idx = {s: [] for s in REPLAY_SRCS}
with io.open(os.path.join(ROOT, "v2fmt_1p5b_full_mix.jsonl"), encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        for s in REPLAY_SRCS:
            if '"src": "%s"' % s in line:
                replay_idx[s].append(i)
                break
n_lang = len(rows)
n_replay = int(n_lang * 0.18)
rng = rng_of("replay3")
need = set()
for s, idxs in replay_idx.items():
    take = rng.sample(idxs, min(n_replay // len(REPLAY_SRCS) + 1, len(idxs)))
    need.update(take)
log("replay rows target: %d (need %d)" % (n_replay, len(need)))
mix_path = os.path.join(ROOT, "lang3_mix.jsonl")
got = 0
with io.open(os.path.join(ROOT, "v2fmt_1p5b_full_mix.jsonl"), encoding="utf-8", errors="replace") as f, \
     io.open(mix_path, "w", encoding="utf-8", newline="") as w:
    for i, line in enumerate(f):
        if i in need:
            w.write(line)
            got += 1
with io.open(mix_path, "a", encoding="utf-8", newline="") as w:
    for r in rows:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
log("mix: %d replay + %d lang rows, %.1fMB (%.0fs)" % (
    got, len(rows), os.path.getsize(mix_path) / 1e6, time.time() - t0))

# ---- 6) 打包（同款打包器）----
A.build_packed_cache(mix_path, os.path.join(ROOT, "lang3_pack.npz"), T=512, seed=42)
import numpy as np_
z = np_.load(os.path.join(ROOT, "lang3_pack.npz"))
w = z["win_tok"]
sample = w[:2000]
cjk = ((sample >= 0xE4) & (sample <= 0xE9)).mean()
ascii_ratio = (sample < 128).mean()
log("pack: %d windows 汉字字节=%.3f ASCII=%.3f (%.0fs total)" % (
    len(w), cjk, ascii_ratio, time.time() - t0))
