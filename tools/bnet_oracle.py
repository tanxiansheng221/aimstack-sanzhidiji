# -*- coding: utf-8 -*-
# PyTorch 对账工具 v2：分阶段 flush，手写贪心循环
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DIR = r"H:\AimCode\aimp\data\bitnet2b"
PROMPT_IDS = [128000, 791, 7438, 315, 2324, 374]

def p(*a):
    print(*a); sys.stdout.flush()

p("STAGE1 tokenizer")
tok = AutoTokenizer.from_pretrained(DIR)
p("STAGE2 model load")
model = AutoModelForCausalLM.from_pretrained(DIR, dtype=torch.bfloat16)
model.eval()
p("STAGE3 forward")
ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
try:
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
except Exception as e:
    p("FORWARD_FAIL:", repr(e)); sys.exit(1)
p("STAGE4 top10")
top = torch.topk(logits, 10)
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    p(f"  {i}\t{v:.4f}\t{tok.decode([i])!r}")

p("STAGE5 greedy16")
cur = list(PROMPT_IDS)
out_ids = []
for _ in range(16):
    with torch.no_grad():
        lg = model(torch.tensor([cur], dtype=torch.long)).logits[0, -1].float()
    nid = int(torch.argmax(lg))
    out_ids.append(nid); cur.append(nid)
p("GREEDY_16:", out_ids)
p("GREEDY_TEXT:", repr(tok.decode(out_ids)))
p("DONE")
