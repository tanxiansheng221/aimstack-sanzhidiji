# -*- coding: utf-8 -*-
# smoke_804m_v2.py — 804M v2 本机 CPU 生成冒烟：生成器出全新题 + 验证器拿真值 + 模型闭卷作答对账
import io, sys, json, time, random, math, os
import numpy as np
import torch

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
os.chdir(r"H:\AimCode\lab\cpu-native")

_KEEPALIVE = [sys.stdout]  # 旧 stdout 持引用防 GC：GC 会 close 共享 buffer，杀死后续包装
import aim_train_real as A  # 它自带 stdout 重包装
import curriculum_graph_lab as G
_KEEPALIVE.append(sys.stdout)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CKPT = r"H:\AimCode\lab\cpu-native\dcu_harvest\v2_804m.pt"
BOS, SEP, EOS, PAD = 257, 10, 258, 256

print("loading checkpoint (cpu) ...", flush=True)
t0 = time.time()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = A.AimLMReal(V=cfg["V"], D=cfg["D"], L=cfg["L"], H1=cfg["H1"], chunk=cfg["chunk"], T=cfg["T"])
model.load_state_dict(ck["model"])
del ck  # 释放 opt 状态（3.2GB）
model.eval()
print("loaded in %.0fs, cfg=%s" % (time.time() - t0, cfg), flush=True)

@torch.no_grad()
def generate(q_text, max_new=120, temperature=0.0):
    """贪心解码：BOS + q + \n 起头，生成到 EOS 或 ' || ' 或预算尽。"""
    qb = q_text.encode("utf-8")[:1600]
    tok = [BOS] + list(qb) + [SEP]
    out = []
    hit_oo = False
    for _ in range(max_new):
        inp = torch.tensor([tok[-cfg["T"]:]], dtype=torch.long)
        logits = model(inp)[0, -1]  # [V]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        else:
            nxt = int(logits.argmax())
        if nxt == EOS:
            break
        out.append(nxt)
        tok.append(nxt)
        tail = bytes(out[-4:]).decode("utf-8", "ignore")
        if tail.endswith(" || "):
            hit_oo = True
            break
    text = bytes(out).decode("utf-8", "replace")
    ans = text.split(" || ")[-1].strip() if (" || " in text) else text.strip()
    return text, ans, hit_oo

# ---------- 冒烟 1：图论科全新题（生成器出题，验证器拿真值） ----------
rng = random.Random(777)
n_graph, g_hit = 8, 0
print("\n=== 图论科全新题（闭卷） ===", flush=True)
for i in range(n_graph):
    sm = G.gen_sample(rng)
    v = G.verify(sm)
    if not v["ok"]:
        continue
    text, ans, hit_oo = generate(sm["q"])
    ok = ans == sm["ans"]
    g_hit += ok
    print("[%d] %-9s gold=%-6s gen=%-10s %s | cot前40: %s" % (
        i, sm["fam"], sm["ans"], ans[:10], "HIT" if ok else "MISS", text[:40].replace("\n", " ")), flush=True)
print("graph: %d/%d" % (g_hit, n_graph), flush=True)

# ---------- 冒烟 2：混料 val 里的真题（十科各抽风格样例） ----------
print("\n=== 混料 val 真题抽查 ===", flush=True)
val_rows = []
with io.open(r"H:\AimCode\lab\cpu-native\curriculum_mix_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") == "val":
            val_rows.append(r)
by_src = {}
for r in val_rows:
    by_src.setdefault(r["subject"], []).append(r)
rng2 = random.Random(42)
n_val, v_hit, v_oo = 10, 0, 0
shown = 0
for s in sorted(by_src):
    r = rng2.choice(by_src[s])
    a_true = ((r.get("cot") or "") + " || " + r["ans"]).strip()
    q_text = r["q"]
    text, ans, hit_oo = generate(q_text, max_new=100)
    v_oo += hit_oo
    ok = ans == r["ans"]
    v_hit += ok
    shown += 1
    if shown > 10:
        break
    print("[%-9s] gold=%-8s gen=%-10s %s" % (s, r["ans"][:8], ans[:10], "HIT" if ok else "MISS"), flush=True)
print("val: %d/%d exact, ' || ' 命中 %d/10" % (v_hit, shown, v_oo), flush=True)
print("=== SMOKE DONE ===", flush=True)
