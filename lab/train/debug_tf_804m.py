# -*- coding: utf-8 -*-
# debug_tf_804m.py — teacher forcing 诊断：答案区逐位 argmax 命中率（区分模型问题 vs 自由生成跑偏）
import io, sys, json, time, os
import numpy as np
import torch

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
os.chdir(r"H:\AimCode\lab\cpu-native")

_KEEPALIVE = [sys.stdout]
import aim_train_real as A
_KEEPALIVE.append(sys.stdout)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CKPT = r"H:\AimCode\lab\cpu-native\dcu_harvest\v2_804m.pt"
BOS, SEP, EOS, PAD = 257, 10, 258, 256

ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
cfg = ck["cfg"]
model = A.AimLMReal(V=cfg["V"], D=cfg["D"], L=cfg["L"], H1=cfg["H1"], chunk=cfg["chunk"], T=cfg["T"])
model.load_state_dict(ck["model"])
model.eval()

# 训练见过的样本（混料全是 train 行）：TF 命中应极高，不高=加载/前向断链
rows = []
with io.open(r"H:\AimCode\lab\cpu-native\v2_804m_mix.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rows.append(r)
rng = np.random.RandomState(42)
picks = rng.choice(len(rows), 12, replace=False)

@torch.no_grad()
def teacher_force(r):
    q = r["q"].encode("utf-8")[:1600]
    a = r["a"].encode("utf-8")[:1600]
    seq = [BOS] + list(q) + [SEP] + list(a) + [EOS]
    if len(seq) > cfg["T"]:
        return None
    tok = torch.tensor([seq], dtype=torch.long)
    amask = torch.zeros(1, len(seq), dtype=torch.bool)
    amask[0, len(q) + 2:] = True  # a + EOS 区
    logits = model(tok)
    lg, tg = logits[0, :-1], tok[0, 1:]
    mk = amask[0, 1:]
    n_tok = int(mk.sum())
    hit = int((lg.argmax(-1)[mk] == tg[mk]).sum())
    gold = bytes(int(b) for b in tg[mk][:60] if b < 256).decode("utf-8", "replace")
    pred = bytes(int(b) for b in lg.argmax(-1)[mk][:60] if b < 256).decode("utf-8", "replace")
    ce = torch.nn.functional.cross_entropy(lg.float(), tg, reduction="none")
    return hit, n_tok, ce[mk].mean().item(), gold, pred

tot_hit = tot_tok = 0
ces = []
i = 0
for pi in picks:
    r = rows[int(pi)]
    res = teacher_force(r)
    if res is None:
        continue
    hit, n_tok, ce, gold, pred = res
    tot_hit += hit; tot_tok += n_tok; ces.append(ce)
    print("[%2d] tok_hit %3d/%-3d ce %.3f" % (i, hit, n_tok, ce), flush=True)
    print("     gold: %r" % gold[:50], flush=True)
    print("     pred: %r" % pred[:50], flush=True)
    i += 1
print("TF: argmax %d/%d = %.1f%%, mean ce %.3f" % (tot_hit, tot_tok, tot_hit / tot_tok * 100, sum(ces) / len(ces)), flush=True)
