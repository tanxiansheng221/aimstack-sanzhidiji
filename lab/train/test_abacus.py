# -*- coding: utf-8 -*-
# test_abacus.py - Abacus 位嵌入冒烟：位序正确性 / 梯度流 / resume 兼容 / 默认关闭零影响
import io, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from aim_train_real import AimLMReal, PAD

def digpos_ref(tok, max_pos):
    """参考实现：逐 token 从右往左数连续数字距离"""
    B, T = tok.shape
    out = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        run = 0
        for i in range(T - 1, -1, -1):
            c = tok[b, i].item()
            if 48 <= c <= 57:
                out[b, i] = min(run, max_pos - 1)
                run += 1
            else:
                run = 0
    return out

fails = []
# T1: 位序正确性（向量化 vs 参考实现）
torch.manual_seed(0)
m = AimLMReal(V=277, D=16, L=1, H1=32, chunk=8, T=24, seed=0, abacus=True, ab_pos=32)
tok = torch.randint(0, 277, (2, 24)); tok[0, :3] = PAD
seq = "x1234y56+789"
tok[1] = torch.tensor([ord(c) for c in seq] + [PAD] * (24 - len(seq)))
d = (tok >= 48) & (tok <= 57)
idx = torch.arange(tok.shape[1])
t = torch.where(~d, idx, torch.full_like(idx, tok.shape[1]))
nxt = torch.flip(torch.cummin(torch.flip(t, [-1]), -1)[0], [-1])
dpos = (nxt - idx - 1).clamp(min=0, max=31)
ref = digpos_ref(tok, 32)
if not torch.equal(dpos, ref):
    fails.append("T1 dpos mismatch:\n vec=%s\n ref=%s" % (dpos.tolist(), ref.tolist()))
else:
    print("T1 dpos OK | x1234y56+789 ->", dpos[1].tolist()[:12])

# T2: 梯度流（AB 有梯度，非数字 token 不受 AB 影响）
m.train()
tokg = tok.clone().requires_grad_(False)
logits = m(tokg)
loss = logits.float().pow(2).mean()
loss.backward()
if m.AB.grad is None or m.AB.grad.abs().sum() == 0:
    fails.append("T2 AB grad is zero/None")
else:
    print("T2 AB grad OK, |g|=%.4f" % m.AB.grad.abs().sum().item())

# T3: AB=0 时 forward 与无 abacus 模型逐位一致（默认关闭零影响）
m0 = AimLMReal(V=277, D=16, L=1, H1=32, chunk=8, T=24, seed=0, abacus=False, ab_pos=32)
m0.load_state_dict({k: v for k, v in m.state_dict().items() if k != "AB"})
with torch.no_grad():
    m.AB.zero_()
    o1 = m(tokg)
    o2 = m0(tokg)
if torch.allclose(o1, o2, atol=1e-6):
    print("T3 AB=0 等价 OK")
else:
    fails.append("T3 mismatch max=%s" % (o1 - o2).abs().max().item())

# T4: resume 兼容（旧 ckpt 无 AB 键 → strict=False 路径）
torch.save({"model": m0.state_dict(), "opt": {}, "step": 123}, "_abacus_fake_ckpt.pt")
ck = torch.load("_abacus_fake_ckpt.pt", map_location="cpu")
missing, unexpected = m.load_state_dict(ck["model"], strict=False)
os.remove("_abacus_fake_ckpt.pt")
if missing == ["AB"] and not unexpected:
    print("T4 resume strict=False OK, missing=%s" % missing)
else:
    fails.append("T4 missing=%s unexpected=%s" % (missing, unexpected))

# T5: 参数计数增量
n_ab = sum(p.numel() for p in m.parameters())
n_0 = sum(p.numel() for p in m0.parameters())
print("T5 params: base=%d abacus=%d delta=%d (=32x16)" % (n_0, n_ab, n_ab - n_0))
if n_ab - n_0 != 32 * 16:
    fails.append("T5 delta=%d" % (n_ab - n_0))

print("=" * 40)
if fails:
    print("FAIL %d:" % len(fails))
    for f in fails:
        print(" -", f)
    sys.exit(2)
print("ABACUS SMOKE ALL PASS")
