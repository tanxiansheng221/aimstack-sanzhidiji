# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

seq = "x1234y56+789"
tok = torch.tensor([ord(c) for c in seq] + [0] * (24 - len(seq))).unsqueeze(0)
d = (tok >= 48) & (tok <= 57)
idx = torch.arange(tok.shape[1])
t = torch.where(~d, idx, torch.full_like(idx, tok.shape[1]))
print("d:  ", d[0].tolist())
print("t:  ", t[0].tolist())
nxt = torch.flip(torch.cummin(torch.flip(t, [0]), 0)[0], [0])
print("nxt:", nxt[0].tolist())
print("dpos:", (nxt - idx - 1).clamp(max=31)[0].tolist())
