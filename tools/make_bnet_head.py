# -*- coding: utf-8 -*-
"""从 BitNet 真·llama3 tokenizer.json 重铸 tokenizer-only GGUF
(修复 m2b_head.gguf 错源词表: 那是 M2.5 对账用合成词表, 不是 BitNet 的)
用法: python make_bnet_head.py -> data/m2b_llama3_head.gguf"""
import json, struct, sys, os

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TJ = os.path.join(ROOT, 'data', 'bitnet2b', 'tokenizer.json')
OUT = os.path.join(ROOT, 'data', 'm2b_llama3_head.gguf')
ALIGN = 64

d = json.load(open(TJ, encoding='utf-8'))
vocab = d['model']['vocab']                       # {token: id}
merges = d['model']['merges']                     # ["a b", ...] or [[a,b],...]
added = d.get('added_tokens', [])

nt = len(vocab) + len(added)
tokens = [None] * nt
for tk, i in vocab.items():
    tokens[i] = tk
for at in added:
    tokens[at['id']] = at['content']
assert all(t is not None for t in tokens), 'id 有空洞'
print(f'tokens={nt} merges={len(merges)} added={len(added)}')

def s(v):
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def kv_str(k, v):
    return s(k) + struct.pack('<I', 8) + s(v)

def kv_u32(k, v):
    return s(k) + struct.pack('<I', 4) + struct.pack('<I', v)

def kv_arr_str(k, vs):
    b = s(k) + struct.pack('<I', 9) + struct.pack('<I', 8) + struct.pack('<Q', len(vs))
    for v in vs:
        b += s(v)
    return b

meta = b''
meta += kv_str('general.architecture', 'llama')
meta += kv_u32('general.alignment', ALIGN)
meta += kv_str('tokenizer.ggml.model', 'gpt2')
meta += kv_arr_str('tokenizer.ggml.tokens', tokens)
mg = [m if isinstance(m, str) else m[0] + ' ' + m[1] for m in merges]
meta += kv_arr_str('tokenizer.ggml.merges', mg)

header = (struct.pack('<I', 0x46554747) + struct.pack('<I', 3) +
          struct.pack('<Q', 0) + struct.pack('<Q', 5) + meta)  # n_tensors=0, n_kv=5
# 对齐填充到 64
pad = (-len(header)) % ALIGN
header += b'\x00' * pad
open(OUT, 'wb').write(header)
print(f'written {OUT} {os.path.getsize(OUT)} bytes')
print('BNET_HEAD_DONE')
