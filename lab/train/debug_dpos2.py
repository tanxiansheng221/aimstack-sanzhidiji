# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
print("torch", torch.__version__)
t = torch.tensor([0, 24, 24, 24, 24, 5, 24, 24, 8, 24, 24, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
rev = torch.flip(t, [0])
print("rev:    ", rev.tolist())
c = torch.cummin(rev, 0)[0]
print("cummin: ", c.tolist())
nxt = torch.flip(c, [0])
print("nxt:    ", nxt.tolist())
