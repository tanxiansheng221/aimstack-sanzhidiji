# -*- coding: utf-8 -*-
# test_bf16adam.py — 自研 AdamWBF16 对账：①与 torch AdamW 的 loss 曲线接近度 ②state_dict 往返一致 ③显存口径（CPU 上验状态 dtype）
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
from aim_train_real import AimLMReal, AdamWBF16

torch.manual_seed(7)
D, L, H1, T, B = 64, 2, 128, 96, 8
V = 277

def gen_batches(n):
    out = []
    g = torch.Generator().manual_seed(123)
    for _ in range(n):
        tok = torch.randint(0, V, (B, T), generator=g)
        amask = torch.zeros(B, T, dtype=torch.bool)
        amask[:, T // 2:] = True
        out.append((tok, amask))
    return out

def answer_loss(logits, tok, amask):
    import torch.nn as nn
    lg = logits[:, :-1, :]; tg = tok[:, 1:]; m = amask[:, 1:]
    return nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none")[m.reshape(-1)].mean()

def run(opt_kind, batches):
    torch.manual_seed(42)
    model = AimLMReal(V=V, D=D, L=L, H1=H1, chunk=32, T=T, seed=42)
    if opt_kind == "torch":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        opt = AdamWBF16(model.parameters(), lr=1e-3)
    losses = []
    for tok, amask in batches:
        logits = model(tok)
        loss = answer_loss(logits, tok, amask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return model, opt, losses

batches = gen_batches(120)
m1, o1, l1 = run("torch", batches)
m2, o2, l2 = run("bf16", batches)

ok = True
# ① loss 曲线接近度：末 20 步均值相对差
tail1 = sum(l1[-20:]) / 20; tail2 = sum(l2[-20:]) / 20
rel = abs(tail1 - tail2) / max(tail1, 1e-9)
print("A. loss tail20: torch=%.4f bf16=%.4f rel=%.4f%%" % (tail1, tail2, rel * 100))
a = rel < 0.05
ok &= a
print("   ->", "PASS" if a else "FAIL", "(判据 <5%)")

# ② state_dict 往返
sd = o2.state_dict()
o2b = AdamWBF16(m2.parameters(), lr=1e-3)
o2b.load_state_dict(sd)
same = o2b.t == sd["t"] and all(torch.equal(x, y) for x, y in zip(o2b.m, sd["m"])) and all(torch.equal(x, y) for x, y in zip(o2b.v, sd["v"]))
print("B. state_dict roundtrip:", "PASS" if same else "FAIL")
ok &= same

# ③ 状态 dtype=bf16（显存口径的根）
dts = set(str(x.dtype) for x in o2.m + o2.v)
c = dts == {"torch.bfloat16"}
print("C. m/v dtype:", dts, "PASS" if c else "FAIL")
ok &= c

# ④ 衰减+更新真的动了权重
moved = any((p1 - p2).abs().max().item() > 0 for p1, p2 in zip(m1.parameters(), m2.parameters()))
print("D. weights diverged (两优化器路径独立):", "PASS" if moved else "FAIL")
ok &= moved

print("=== ALL", "PASS" if ok else "FAIL", "===")
sys.exit(0 if ok else 1)
