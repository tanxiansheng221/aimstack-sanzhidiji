# -*- coding: utf-8 -*-
# build_lang1_corpus.py — 语言先行包 lang1：共情/创意/拒绝重料 + 小对话 + 推理科回放
#   目标：治「在吗→乱码」。产物 lang1_mix.jsonl → lang1_pack.npz（~7000 步量）
#   纪律：G3 确定性（crc32 种子）、行长度过滤（q+a<=480B 由打包器兜底）、回放防遗忘
import io, json, random, zlib, time, os, sys
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
SEED = 20261104
t0 = time.time()

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

def rng_of(name, extra=0):
    return random.Random((zlib.crc32(name.encode()) + SEED + extra) % (2**31))

rows = []

# ---- 1) emp2 共情 ×2（13 万源 → 26 万行）----
log("emp2 x2 ...")
src = os.path.join(ROOT, "emp2_corpus", "emp2_corpus.jsonl")
base = []
with io.open(src, encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = "[情绪:%s] %s" % (r.get("emotion", "一般"), r.get("user_situation", ""))
        a = r.get("resp", "")
        if q.strip() and a.strip():
            base.append((q, a))
rng = rng_of("emp2")
pick = [base[rng.randrange(len(base))] for _ in range(2 * len(base))]
for q, a in pick:
    rows.append({"q": q, "a": a, "src": "lang_emp2", "split": "train"})
log("emp2: %d rows (%.0fs)" % (len(rows), time.time() - t0))

# ---- 2) creative ×1.5（9.5 万源 → 14 万行）----
log("creative x1.5 ...")
base = []
with io.open(os.path.join(ROOT, "creative_corpus", "creative_v0.jsonl"), encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = "[创意:%s] %s" % (r.get("task", "story"), r.get("q", ""))
        a = r.get("ans", "")
        if q.strip() and a.strip():
            base.append((q, a))
rng = rng_of("creative")
pick = [base[rng.randrange(len(base))] for _ in range(int(1.5 * len(base)))]
for q, a in pick:
    rows.append({"q": q, "a": a, "src": "lang_creative", "split": "train"})
log("creative: %d rows total=%d (%.0fs)" % (len(pick), len(rows), time.time() - t0))

# ---- 3) refusal ×1.5（从 v2fmt 混料抽 6.5 万源 → 10 万行）----
log("refusal x1.5 ...")
refusal_base = []
with io.open(os.path.join(ROOT, "v2fmt_1p5b_full_mix.jsonl"), encoding="utf-8", errors="replace") as f:
    for line in f:
        if '"replay_refusal"' in line:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("src") == "replay_refusal" and r.get("q") and r.get("a"):
                refusal_base.append((r["q"], r["a"]))
rng = rng_of("refusal")
pick = [refusal_base[rng.randrange(len(refusal_base))] for _ in range(int(1.5 * len(refusal_base)))]
for q, a in pick:
    rows.append({"q": q, "a": a, "src": "lang_refusal", "split": "train"})
log("refusal: %d rows total=%d (%.0fs)" % (len(pick), len(rows), time.time() - t0))

# ---- 4) 小对话合成（4000 条 ×6 = 2.4 万行）——直治「在吗」----
log("smalltalk synth ...")
HELLOS = ["在吗", "你好", "在不在", "嗨", "hello", "hi", "有人吗", "在么", "晚上好", "早上好",
          "下午好", "你好呀", "您好", "hi~", "hello world", "在吗在吗", "能听见吗", "测试", "ping", "你好！"]
REPLIES = ["在的，怎么了？", "你好！有什么可以帮你的？", "在呢，说吧。", "嗨！我在。", "Hello! How can I help?",
           "在的，请讲。", "我在的，什么事？", "你好呀，今天怎么样？", "嗯，在。你说。", "Hi there!"]
IDENT_Q = ["你是谁", "你叫什么名字", "你是什么", "你是机器人吗", "你会什么", "你能干什么",
           "who are you", "what can you do", "你是AI吗", "介绍一下你自己"]
IDENT_A = ["我是 AIM，一个小型的字节级语言模型，可以陪你聊天、做推理题。", "我叫 AIM，会一点点推理和聊天。",
           "我是一个小语言模型 AIM，擅长按格式解题，也能简单聊聊。", "AIM，很小的模型，但很努力。"]
THANKS_Q = ["谢谢", "多谢", "thanks", "thank you", "谢啦", "辛苦了"]
THANKS_A = ["不客气！", "不用谢，应该的。", "Anytime!", "小事一桩。", "客气啦，有需要随时说。"]
BYE_Q = ["再见", "拜拜", "晚安", "bye", "回见", "下次聊"]
BYE_A = ["再见！", "拜拜，有空再聊。", "晚安，好梦。", "Bye!", "好的，回见。"]
SMALL = [
    ("今天天气怎么样", "我没有实时天气数据，不过你那边要是出太阳，就适合出门走走。"),
    ("你聪明吗", "我是个小模型，推理还可以，聊天马马虎虎，算术不太行，诚实地说。"),
    ("你多大了", "我刚训练完不久，算是个新生儿。"),
    ("讲个笑话", "程序员最讨厌的两件事：别人不写注释，和让自己写注释。"),
    ("你会写诗吗", "会一点，你出个题目试试？"),
    ("帮我算个数", "小的算术我可以试试，多位数不一定准，你可以考考我。"),
    ("你吃饭了吗", "我不吃饭，吃电。谢谢关心！"),
    ("你现在在哪", "我跑在用户的机器上，没有具体位置，哪里有电哪里在。"),
]
synth = []
ri = rng_of("hello_replies")
for q in HELLOS:
    synth.append((q, REPLIES[ri.randrange(len(REPLIES))]))
ri = rng_of("ident_replies")
for q in IDENT_Q:
    synth.append((q, IDENT_A[ri.randrange(len(IDENT_A))]))
ri = rng_of("thanks_replies")
for q in THANKS_Q:
    synth.append((q, THANKS_A[ri.randrange(len(THANKS_A))]))
ri = rng_of("bye_replies")
for q in BYE_Q:
    synth.append((q, BYE_A[ri.randrange(len(BYE_A))]))
synth.extend(SMALL)
# 每条 ×N 到 2.4 万
rng = rng_of("smalltalk_rep")
rep = 24000 // max(1, len(synth)) + 1
for _ in range(rep):
    for q, a in synth:
        rows.append({"q": q, "a": a, "src": "lang_smalltalk", "split": "train"})
log("smalltalk: %d rows total=%d (%.0fs)" % (len(synth) * rep, len(rows), time.time() - t0))

# ---- 5) 推理科回放 15%（五科除外，防遗忘 csp/graph/logic/causal/fsm）----
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
rng = rng_of("replay")
need = set()
for s, idxs in replay_idx.items():
    take = rng.sample(idxs, min(n_replay // len(REPLAY_SRCS) + 1, len(idxs)))
    need.update(take)
log("replay rows target: %d (need %d)" % (n_replay, len(need)))
mix_path = os.path.join(ROOT, "lang1_mix.jsonl")
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

# ---- 6) 打包 ----
A.build_packed_cache(mix_path, os.path.join(ROOT, "lang1_pack.npz"), T=512, seed=42)
import numpy as np_
z = np_.load(os.path.join(ROOT, "lang1_pack.npz"))
log("pack: %d windows (%.0fs total)" % (len(z["win_tok"]), time.time() - t0))
