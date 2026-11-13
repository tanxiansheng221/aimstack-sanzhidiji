# -*- coding: utf-8 -*-
# M2.5: C tokenizer vs Python refbpe 对账
import subprocess, sys
sys.path.insert(0, 'tools')
from refbpe import read_gguf_tokenizer, RefBPE

kv = read_gguf_tokenizer('data/m2b_head.gguf')
bp = RefBPE(kv)

SAMPLES = [
    "hello world",
    "The quick brown fox jumps over the lazy dog",
    "你好，世界",
    "今天天气不错",
    "def foo(x): return x+1",
    "42 is the answer",
    "人工智能改变世界",
    "x = 3.14 * 2",
    "a",
    "     spaces   here",
    "Mixed 中文 and English text 123",
]

ok = True
for t in SAMPLES:
    ref = bp.encode(t)
    r = subprocess.run(['build/aimp_probe.exe', 'data/m2b_head.gguf',
                        '--enc', t], capture_output=True, text=True,
                       errors='replace')
    got = []
    for line in r.stdout.splitlines():
        if line.startswith('ENC_IDS'):
            got = [int(x) for x in line.replace('ENC_IDS:', '').split()]
    m = (got == ref)
    ok = ok and m
    print('%s  %-40r ref=%s got=%s' % ('PASS' if m else 'FAIL', t[:36],
                                       ref[:8], got[:8]))
print('==== M2.5 %s ====' % ('ALL_PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
