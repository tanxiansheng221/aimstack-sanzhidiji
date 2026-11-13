# -*- coding: utf-8 -*-
# lang1: multilingual FSM probe — 6 languages (en/zh/es/ar/ru/ja) mixed training, per-language eval
# Question: does byte-level tokenization + mixed training give uniform multilingual task ability?
import os, time, json, math, random
import numpy as np
import torch
import torch.nn as nn

R = lambda s: print(s, flush=True)
R("=== LANG1: multilingual FSM probe ===")
R("torch %s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- data gen: FSM, 6 languages ----------
BOS, PAD = 257, 256

def tok_bytes(s, max_len):
    ids = [BOS] + list(s.encode("utf-8"))[: max_len - 1]
    return ids + [PAD] * (max_len - len(ids))

T_TPL = {
 "en": {"final": "Machine transitions: {t}. Input: {i}. Start at S0. After the whole input, which state? Answer S<k>.",
        "visit": "Machine transitions: {t}. Input: {i}. Start at S0. How many times is state S{k} visited (including start)? Answer a number.",
        "hit":   "Machine transitions: {t}. Input: {i}. Start at S0. Is state S{k} visited at any point? Answer yes or no.",
        "yes": "yes", "no": "no"},
 "zh": {"final": "机器转移：{t}。输入：{i}。从 S0 开始。整个输入结束后处于哪个状态？回答 S<k>。",
        "visit": "机器转移：{t}。输入：{i}。从 S0 开始。状态 S{k} 被访问多少次（包括起点）？回答一个数字。",
        "hit":   "机器转移：{t}。输入：{i}。从 S0 开始。状态 S{k} 是否被访问过？回答 {y} 或 {n}。",
        "yes": "是", "no": "否"},
 "es": {"final": "Transiciones de la máquina: {t}. Entrada: {i}. Empieza en S0. ¿En qué estado está tras toda la entrada? Responde S<k>.",
        "visit": "Transiciones de la máquina: {t}. Entrada: {i}. Empieza en S0. ¿Cuántas veces se visita el estado S{k} (incluido el inicio)? Responde un número.",
        "hit":   "Transiciones de la máquina: {t}. Entrada: {i}. Empieza en S0. ¿Se visita el estado S{k} en algún momento? Responde {y} o {n}.",
        "yes": "sí", "no": "no"},
 "ar": {"final": "تحولات الآلة: {t}. المدخل: {i}. يبدأ من S0. في أي حالة بعد نهاية المدخل بالكامل؟ أجب S<k>.",
        "visit": "تحولات الآلة: {t}. المدخل: {i}. يبدأ من S0. كم مرة تُزار الحالة S{k} (بما فيها البداية)? أجب برقم.",
        "hit":   "تحولات الآلة: {t}. المدخل: {i}. يبدأ من S0. هل تُزار الحالة S{k} في أي لحظة؟ أجب {y} أو {n}.",
        "yes": "نعم", "no": "لا"},
 "ru": {"final": "Переходы машины: {t}. Вход: {inp}. Начало в S0. В каком состоянии машина после всего входа? Ответь S<k>.",
        "visit": "Переходы машины: {t}. Вход: {inp}. Начало в S0. Сколько раз посещается состояние S{k} (включая начало)? Ответь числом.",
        "hit":   "Переходы машины: {t}. Вход: {inp}. Начало в S0. Посещается ли состояние S{k} хотя бы раз? Ответь {y} или {n}.",
        "yes": "да", "no": "нет"},
 "ja": {"final": "マシンの遷移: {t}。入力: {i}。S0 から開始。入力全体の後、どの状態にいる？ S<k> で答えて。",
        "visit": "マシンの遷移: {t}。入力: {i}。S0 から開始。状態 S{k} は何回訪問される（開始含む）？ 数字で答えて。",
        "hit":   "マシンの遷移: {t}。入力: {i}。S0 から開始。状態 S{k} は一度でも訪問される？ {y} か {n} で答えて。",
        "yes": "はい", "no": "いいえ"},
}
LANGS = list(T_TPL.keys())

def gen_row(rng):
    ns = rng.choice([3, 4, 5]); sym_n = 3
    trans = [[rng.randrange(ns) for _ in range(sym_n)] for _ in range(ns)]
    seq = [rng.randrange(sym_n) for _ in range(rng.choice([4, 8, 12, 16]))]
    s = 0; trace = [0]
    for a in seq:
        s = trans[s][a]; trace.append(s)
    sym = "abc"
    t_txt = "; ".join("S%d%s>S%d" % (si, sym[a], trans[si][a]) for si in range(ns) for a in range(sym_n))
    i_txt = ",".join(sym[a] for a in seq)
    lang = rng.choice(LANGS)
    T = T_TPL[lang]
    qt = rng.choice(["final", "visit", "hit"])
    k = rng.randrange(ns)
    if qt == "final":
        q = T["final"].format(t=t_txt, i=i_txt); ans = "S%d" % s
    elif qt == "visit":
        q = T["visit"].format(t=t_txt, i=i_txt, k=k); ans = str(trace.count(k))
    else:
        q = T["hit"].format(t=t_txt, i=i_txt, k=k, y=T["yes"], n=T["no"])
        ans = T["yes"] if k in trace else T["no"]
    cot = " -> ".join("S%d" % x for x in trace)
    return {"lang": lang, "qt": qt, "q": q, "cot": cot, "ans": ans}

rng = random.Random(42)
train = [gen_row(rng) for _ in range(2400)]
val_by_lang = {L: [gen_row(rng) for _ in range(100)] for L in LANGS}
R("data: train %d, val %d/lang" % (len(train), len(val_by_lang[LANGS[0]])))

def batch_rows(rows, max_q=256, max_a=128):
    X = np.array([tok_bytes(r["q"], max_q) for r in rows], dtype=np.int64)
    Y = np.array([tok_bytes(r["cot"] + " || " + r["ans"], max_a) for r in rows], dtype=np.int64)
    return X, Y

# ---------- model: AimLM (linear attention, same family) ----------
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
    def __init__(self, V=277, D=192, L=2, H1=384, T=256):
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

def aloss(logits, y):
    lg = logits[:, -y.shape[1]:, :]
    mask = y != PAD
    return nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none")[mask.reshape(-1)].mean()

@torch.no_grad()
def eval_rows(m, rows):
    accs, apreds = [], []
    for cs in range(0, len(rows), 64):
        X, Y = batch_rows(rows[cs:cs + 64])
        xb, yb = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
        lg = m(xb)
        pred = lg[:, -yb.shape[1]:, :].argmax(-1)
        mask = yb != PAD
        accs.append(((pred == yb) | ~mask).all(dim=1).float())
        apos = mask.sum(1) - 1
        apreds.append((pred.gather(1, apos[:, None]) == yb.gather(1, apos[:, None])).float())
    return torch.cat(accs).mean().item(), torch.cat(apreds).mean().item()

m = AimLM().to(dev)
R("params: %d" % sum(p.numel() for p in m.parameters()))
opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.1)
STEPS, B, WARM = 12000, 64, 600
Xtr, Ytr = batch_rows(train)
Xt, Yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
curve = []
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(Xt), (B,), device=dev)
    loss = aloss(m(Xt[idx]), Yt[idx])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    lr = 6e-4 * ((step + 1) / WARM if step < WARM else 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM))))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.step()
    if step % 2000 == 0 or step == STEPS:
        per = {}
        for L in LANGS:
            ve, ae = eval_rows(m, val_by_lang[L])
            per[L] = {"val_exact": round(ve, 4), "ans_exact": round(ae, 4)}
        R("step %d loss %.4f %s %.0fs" % (step, loss.item(), json.dumps(per), time.time() - t0))
        curve.append({"step": step, "loss": round(loss.item(), 4), "per_lang": per})

res = {"curve": curve, "wall": round(time.time() - t0, 1)}
json.dump(res, open("lang1_results.json", "w"), indent=1, ensure_ascii=False)
R("FINAL " + json.dumps(curve[-1]["per_lang"], ensure_ascii=False))
R("=== LANG1 DONE ===")
