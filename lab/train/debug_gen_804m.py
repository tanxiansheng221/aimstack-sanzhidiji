# -*- coding: utf-8 -*-
# debug_gen_804m.py — 生成空输出诊断：mmap 加载 + 首 token 分布 + 原始 token 流
import io, sys, json, time, random, os
import torch

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
os.chdir(r"H:\AimCode\lab\cpu-native")

_KEEPALIVE = [sys.stdout]
import aim_train_real as A
_KEEPALIVE.append(sys.stdout)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CKPT = r"H:\AimCode\lab\cpu-native\dcu_harvest\v2_804m.pt"
BOS, SEP, EOS, PAD = 257, 10, 258, 256

t0 = time.time()
ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
print("loaded %.0fs" % (time.time() - t0), flush=True)
cfg = ck["cfg"]
model = A.AimLMReal(V=cfg["V"], D=cfg["D"], L=cfg["L"], H1=cfg["H1"], chunk=cfg["chunk"], T=cfg["T"])
model.load_state_dict(ck["model"])
model.eval()
print("model ready, step=%s" % ck.get("step"), flush=True)

# 短题：math val 一条
row = None
with io.open("curriculum_mix_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") == "val" and r.get("subject") == "math":
            row = r
            break
q = row["q"]
print("Q:", q[:80], flush=True)
print("GOLD:", row["ans"], flush=True)

tok = [BOS] + list(q.encode("utf-8")) + [SEP]
print("prompt len:", len(tok), flush=True)

@torch.no_grad()
def step(toks):
    inp = torch.tensor([toks], dtype=torch.long)
    logits = model(inp)
    print("logits shape:", tuple(logits.shape), "last-vec mean %.4f std %.4f" % (
        logits[0, -1].mean(), logits[0, -1].std()), flush=True)
    last = logits[0, -1]
    top = torch.topk(last, 5)
    for sc, idx in zip(top.values.tolist(), top.indices.tolist()):
        print("  top: id=%d byte=%r score=%.3f" % (idx, chr(idx) if idx < 256 else "<spec>", sc), flush=True)
    return last

step(tok[:700])
last = step(tok)
gen = []
for i in range(40):
    nxt = int(last.argmax())
    gen.append(nxt)
    if nxt == EOS:
        print("HIT EOS at %d" % i, flush=True)
        break
    last = model(torch.tensor([tok + gen], dtype=torch.long))[0, -1]
print("gen ids:", gen[:20], flush=True)
print("gen text: %r" % bytes([b for b in gen if b < 256]).decode("utf-8", "replace"), flush=True)
