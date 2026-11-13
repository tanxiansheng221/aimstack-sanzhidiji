# -*- coding: utf-8 -*-
# emp1: empathy-value SFT probe — template-grammar corpus (mechanically scoreable) + tiny AimLM SFT
# Metrics: emotion_hit / followup / length_ok / distinct / safety_zero — the mechanically verifiable core of "emotional value"
import os, time, json, math, random
import numpy as np
import torch
import torch.nn as nn

R = lambda s: print(s, flush=True)
R("=== EMP1: empathy SFT probe ===")
dev = "cuda" if torch.cuda.is_available() else "cpu"
R("torch %s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
BOS, PAD = 257, 256

def tok_bytes(s, max_len):
    ids = [BOS] + list(s.encode("utf-8"))[: max_len - 1]
    return ids + [PAD] * (max_len - len(ids))

# ---------- scenario bank (zh + en) ----------
SC = [
 ("悲伤", "sad", ["我养了很多年的狗昨天走了", "我最好的朋友搬去了很远的城市", "奶奶上个月去世了，我很难受"]),
 ("焦虑", "anxious", ["下周有个特别重要的面试，我整晚睡不着", "体检报告有几个指标异常，我很害怕", "房租涨了，我担心这个月钱不够"]),
 ("愤怒", "angry", ["同事把我的方案署了自己的名", "我被无缘无故骂了一顿", "排队两小时结果被告知排错了"]),
 ("孤独", "lonely", ["周末一个人待了两天，没跟任何人说话", "搬到新城市三个月，一个朋友都没有", "大家聚会都没叫我"]),
 ("压力", "stressed", ["三个项目同时压过来，我快撑不住了", "家里和工作的矛盾夹着我，喘不过气", "所有人都指望我，但我也只是一个人"]),
 ("挫败", "frustrated", ["准备了三个月的考试还是没过", "创业项目黄了，钱也赔了", "改了八遍的方案还是被否了"]),
 ("喜悦", "joyful", ["我拿到心仪公司的offer了！", "我坚持跑步一百天了！", "今天有人夸我做的菜好吃，超开心"]),
 ("疲惫", "tired", ["连续加班三周，我感觉被掏空了", "每天睡五个小时，白天像行尸走肉", "照顾病人一个月，我快撑不动了"]),
]
LEX_ZH = {"悲伤": ["难过", "伤心", "心疼", "心痛", "低落"], "焦虑": ["担心", "紧张", "不安", "害怕"],
          "愤怒": ["生气", "恼火", "委屈", "火大"], "孤独": ["孤单", "寂寞", "一个人不好受"],
          "压力": ["压力", "喘不过气", "撑不住"], "挫败": ["挫败", "不甘心", "白费", "可惜"],
          "喜悦": ["开心", "为你高兴", "替你开心", "真棒", "了不起"], "疲惫": ["累", "疲惫", "辛苦"]}
LEX_EN = {"sad": ["sorry", "sad", "hurt", "pain"], "anxious": ["worry", "anxious", "nervous", "scary", "understand"],
          "angry": ["angry", "upset", "unfair", "frustrat"], "lonely": ["lonely", "alone", "there for you"],
          "stressed": ["pressure", "overwhelmed", "heavy"], "frustrated": ["frustrat", "disappoint", "hard work"],
          "joyful": ["happy", "wonderful", "congratul", "proud"], "tired": ["tired", "exhausted", "drained", "rest"]}
ACK_ZH = ["听到这些，我真的很心疼。", "你的这种感受完全可以理解。", "谢谢你愿意告诉我这些。", "我能感觉到你现在有多难受。", "这真的不容易。"]
ACK_EN = ["I'm really sorry you're going through this.", "Your feelings make complete sense.", "Thank you for trusting me with this.", "That sounds really hard."]
VAL_ZH = ["换成谁遇到这种事都会这样的。", "这不是你的错，你已经做得很好了。", "有这种情绪很正常，不用责怪自己。", "你的感受是真实且值得被认真对待的。"]
VAL_EN = ["Anyone would feel this way.", "It's not your fault — you've done your best.", "It's okay to feel this way.", "Your feelings are valid."]
SUP_ZH = ["如果愿意多说一点，我一直在听。", "不管结果怎样，你都不用一个人扛。", "先照顾好自己，别急着想别的。", "你已经很努力了，这一点很了不起。"]
SUP_EN = ["I'm here whenever you want to talk more.", "You don't have to carry this alone.", "Take care of yourself first.", "You've been trying so hard — that counts."]
FOL_ZH = ["你现在最想聊哪一部分？", "这种感觉持续多久了？", "身边有人可以陪你聊聊吗？", "接下来你打算怎么安排？"]
FOL_EN = ["What part of this is weighing on you most?", "How long have you been feeling this way?", "Is there someone around who can support you?", "What would help you most right now?"]
BAN = ["别想了", "这有什么", "至于吗", "你想多了", "赶紧的"]

def mk_zh(rng, emo_zh, utt):
    parts = [rng.choice(ACK_ZH), rng.choice(VAL_ZH), rng.choice(SUP_ZH), rng.choice(FOL_ZH)]
    rng.shuffle(parts)
    n = rng.choice([2, 3, 3, 4])
    resp = "".join(parts[:n])
    return {"lang": "zh", "emo": emo_zh, "user": utt, "resp": resp}

def mk_en(rng, emo_en, utt):
    parts = [rng.choice(ACK_EN), rng.choice(VAL_EN), rng.choice(SUP_EN), rng.choice(FOL_EN)]
    rng.shuffle(parts)
    n = rng.choice([2, 3, 3, 4])
    resp = " ".join(parts[:n])
    return {"lang": "en", "emo": emo_en, "user": utt, "resp": resp}

rng = random.Random(7)
corpus, seen = [], set()
for rep in range(25):
    for emo_zh, emo_en, utts in SC:
        for utt in utts:
            r = mk_zh(rng, emo_zh, utt)
            if r["resp"] not in seen:
                seen.add(r["resp"]); corpus.append(r)
            r = mk_en(rng, emo_en, utt)
            if r["resp"] not in seen:
                seen.add(r["resp"]); corpus.append(r)
rng.shuffle(corpus)
cut = int(len(corpus) * 0.9)
train, val = corpus[:cut], corpus[cut:]
R("corpus: %d (train %d val %d), distinct responses %d" % (len(corpus), len(train), len(val), len(seen)))

# ---------- mechanical scorecard ----------
def score(resp, emo, lang):
    lex = (LEX_ZH if lang == "zh" else LEX_EN)[emo]
    rl = resp.lower()
    return {"emotion_hit": any(w in rl for w in lex),
            "followup": ("？" in resp) or ("?" in resp),
            "length_ok": 15 <= len(resp) <= 200,
            "safety": not any(b in resp for b in BAN)}

def fmt_ok(s):
    return s["emotion_hit"] and s["followup"] and s["length_ok"] and s["safety"]

R("template ceiling: %.3f" % (sum(fmt_ok(score(r["resp"], r["emo"], r["lang"])) for r in val) / len(val)))

# ---------- model (same AimLM family) ----------
class LinAttnBlock(nn.Module):
    def __init__(self, D, H1):
        super().__init__()
        self.Wq = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.Wk = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.Wv = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.Wo = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.W1 = nn.Parameter(torch.randn(D, H1) / math.sqrt(D))
        self.b1 = nn.Parameter(torch.zeros(H1))
        self.W2 = nn.Parameter(torch.randn(H1, D) / math.sqrt(H1))
        self.b2 = nn.Parameter(torch.zeros(D))
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
    def forward(self, h):
        q = h @ self.Wq
        k = (h @ self.Wk) / math.sqrt(h.shape[-1])
        v = h @ self.Wv
        cum = torch.cumsum(torch.einsum("bti,btj->btij", k, v), dim=1)
        o = torch.einsum("bti,btij->btj", q, cum)
        h = self.ln1(h + o @ self.Wo)
        h = self.ln2(h + torch.relu(h @ self.W1 + self.b1) @ self.W2 + self.b2)
        return h

class AimLM(nn.Module):
    def __init__(self, V=277, D=192, L=2, H1=384, T=192):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.E = nn.Parameter(torch.randn(V, D, generator=g) * 0.1)
        self.Pp = nn.Parameter(torch.randn(T, D, generator=g) * 0.1)
        self.blocks = nn.ModuleList([LinAttnBlock(D, H1) for _ in range(L)])
        self.Wout = nn.Parameter(torch.randn(D, V, generator=g) / math.sqrt(D))
    def forward(self, tok):
        h = self.E[tok] + self.Pp[None, : tok.shape[1], :]
        for b in self.blocks:
            h = b(h)
        return h @ self.Wout

MAXQ, MAXA = 160, 160
def enc(r):
    u = "用户: " + r["user"] if r["lang"] == "zh" else "User: " + r["user"]
    a = r["resp"]
    return tok_bytes(u, MAXQ), tok_bytes(a, MAXA)

X = np.array([enc(r)[0] for r in train], dtype=np.int64)
Y = np.array([enc(r)[1] for r in train], dtype=np.int64)
m = AimLM(T=MAXQ).to(dev)
R("params: %d" % sum(p.numel() for p in m.parameters()))
opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.1)
Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
STEPS, B, WARM = 10000, 64, 500
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(Xt), (B,), device=dev)
    lg = m(Xt[idx])
    lg = lg[:, -Yt.shape[1]:, :]
    mask = Yt[idx] != PAD
    loss = nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), Yt[idx].reshape(-1), reduction="none")[mask.reshape(-1)].mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    lr = 6e-4 * ((step + 1) / WARM if step < WARM else 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM))))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.step()
    if step % 2500 == 0:
        R("step %d loss %.4f %.0fs" % (step, loss.item(), time.time() - t0))

# ---------- greedy generation + scorecard ----------
@torch.no_grad()
def gen(m, user_text, lang, max_new=140):
    u = "用户: " + user_text if lang == "zh" else "User: " + user_text
    ids = tok_bytes(u, MAXQ)
    ctx = torch.tensor([ids], device=dev)
    out = []
    for _ in range(max_new):
        lg = m(ctx)[:, -1, :].argmax(-1)
        t = lg.item()
        if t == PAD:
            break
        out.append(t)
        if ctx.shape[1] >= MAXQ:
            break
        ctx = torch.cat([ctx, torch.tensor([[t]], device=dev)], dim=1)
    return bytes(out).decode("utf-8", "replace")

n_eval = min(160, len(val))
stats = {"emotion_hit": 0, "followup": 0, "length_ok": 0, "safety": 0, "fmt_ok": 0}
samples = []
for r in val[:n_eval]:
    g = gen(m, r["user"], r["lang"])
    s = score(g, r["emo"], r["lang"])
    for k in stats:
        stats[k] += 1 if s[k] else 0
    if len(samples) < 6:
        samples.append({"user": r["user"], "gold": r["resp"], "gen": g})
res = {"template_ceiling": R_ceiling if False else None, "n": n_eval, "gen": {k: round(v / n_eval, 4) for k, v in stats.items()}, "samples": samples}
res["template_ceiling"] = round(sum(fmt_ok(score(r["resp"], r["emo"], r["lang"])) for r in val) / len(val), 4)
json.dump(res, open("emp1_results.json", "w"), indent=1, ensure_ascii=False)
R("GEN METRICS: " + json.dumps(res["gen"], ensure_ascii=False))
R("template ceiling %.3f" % res["template_ceiling"])
for s in samples:
    R("USER: %s\nGOLD: %s\nGEN : %s\n" % (s["user"], s["gold"], s["gen"]))
R("=== EMP1 DONE ===")
