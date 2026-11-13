# -*- coding: utf-8 -*-
"""用自研加载器(aimp_probe.exe)对 BitNet AIM-W 发「在吗」: 编码(查词表)->生成(计时)->解码"""
import json, subprocess, time, os, sys

ROOT = r'H:\AimCode\aimp'
EXE = os.path.join(ROOT, 'build', 'aimp_probe.exe')
AIMW = os.path.join(ROOT, 'data', 'bitnet2b', 'm2b_tern.aimw')
TOKJ = os.path.join(ROOT, 'data', 'bitnet2b', 'tokenizer.json')
sys.stdout.reconfigure(encoding='utf-8')

tj = json.load(open(TOKJ, encoding='utf-8'))
vocab = tj['model']['vocab']
rev = {v: k for k, v in vocab.items()}
BOS = vocab.get('<|begin_of_text|>', 128000)

def enc(text):
    ids = [BOS]
    for ch in text:
        if ch in vocab:
            ids.append(vocab[ch])
        else:
            b = ch.encode('utf-8')
            for bb in b:
                key = '<0x%02X>' % bb
                ids.append(vocab.get(key, 0))
    return ids

def dec(ids):
    out = []
    for i in ids:
        p = rev.get(i, '')
        if p.startswith('<0x') and p.endswith('>'):
            try:
                out.append(bytes([int(p[3:-1], 16)]))
                continue
            except ValueError:
                pass
        if p.startswith('<|') and p.endswith('|>'):
            continue
        out.append(p.encode('utf-8'))
    return b''.join(out).decode('utf-8', errors='replace')

prompt = '在吗'
ids = enc(prompt)
id_str = ','.join(str(i) for i in ids)
print('PROMPT:', prompt, '-> ids:', id_str, '| roundtrip:', dec(ids[1:]))

def gen(n, spec):
    args = [EXE, AIMW, '--gen', id_str, str(n)] + ([str(spec)] if spec > 1 else [])
    # 预热一次 (页缓存), 不计
    subprocess.run(args, capture_output=True)
    t0 = time.perf_counter()
    r = subprocess.run(args, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    line = [l for l in r.stdout.splitlines() if l.startswith('GEN_IDS')][0]
    out_ids = [int(x) for x in line.split(':')[1].split()]
    return out_ids, dt

for tag, n, spec in (('serial', 64, 1), ('spec k=2', 64, 2)):
    out, dt = gen(n, spec)
    print(f'[{tag}] {dt:.2f}s  {n/dt:.2f} tok/s')
    print(f'[{tag}] reply:', dec(out))
