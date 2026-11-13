# -*- coding: utf-8 -*-
# aimp M2 对账: 合成迷你 llama GGUF(Q8_0) + NumPy 基线 -> C 引擎 argmax 对账
import struct, subprocess, sys, os
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # cwd 自锚定 aimp 根

ALIGN = 32
NL, NH, NKV, HD = 2, 2, 1, 16
D = NH * HD
FF = 32
VOC = 32
CTX = 64
EPS = 1e-5
ROPE_BASE = 10000.0
PROMPT = [3, 7, 11, 5]
N_NEXT = 3
rng = np.random.default_rng(42)

def s(v):
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def kv_str(k, v):   return s(k) + struct.pack('<I', 8) + s(v)
def kv_u32(k, v):   return s(k) + struct.pack('<I', 4) + struct.pack('<I', v)
def kv_f32(k, v):   return s(k) + struct.pack('<I', 6) + struct.pack('<f', v)

def kv_arr_str(k, vs):
    b = s(k) + struct.pack('<I', 9) + struct.pack('<I', 8) + struct.pack('<Q', len(vs))
    for v in vs: b += s(v)
    return b

def q8_quant(x):
    # GGUF Q8_0: 2B f16 scale + 32B int8 = 34B/?
    out = bytearray(); deq = np.zeros(len(x), np.float32)
    nb = (len(x) + 31) // 32
    xp = np.concatenate([x, np.zeros(nb*32 - len(x), np.float32)])
    for b in range(nb):
        blk = xp[b*32:(b+1)*32]
        amax = float(np.abs(blk).max())
        sc16 = np.float16(amax / 127.0) if amax > 0 else np.float16(1.0)
        q = np.clip(np.round(blk / np.float32(sc16)), -127, 127).astype(np.int8)
        out += struct.pack('<e', float(sc16)) + q.tobytes()
        deq[b*32:(b+1)*32] = (np.float32(sc16) * q.astype(np.float32))[:len(deq)-b*32]
    return bytes(out), deq

def tensor_info(name, dims, qt, offset):
    b = s(name) + struct.pack('<I', len(dims))
    for d in dims: b += struct.pack('<Q', d)
    return b + struct.pack('<I', qt) + struct.pack('<Q', offset)

def f32_arr(x):
    return struct.pack('<%df' % len(x), *x)

W = {}
W['token_embd'] = rng.normal(0, 1, D*VOC).astype(np.float32)
for l in range(NL):
    W['blk%d.attn_norm' % l] = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.attn_q' % l]      = rng.normal(0, .5, D*D).astype(np.float32)
    W['blk%d.attn_k' % l]      = rng.normal(0, .5, D*NKV*HD).astype(np.float32)
    W['blk%d.attn_v' % l]      = rng.normal(0, .5, D*NKV*HD).astype(np.float32)
    W['blk%d.attn_output' % l] = rng.normal(0, .3, D*D).astype(np.float32)
    W['blk%d.ffn_norm' % l]    = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.ffn_gate' % l]    = rng.normal(0, .5, D*FF).astype(np.float32)
    W['blk%d.ffn_up' % l]      = rng.normal(0, .5, D*FF).astype(np.float32)
    W['blk%d.ffn_down' % l]    = rng.normal(0, .3, FF*D).astype(np.float32)
W['output_norm'] = (1 + rng.normal(0, .1, D)).astype(np.float32)
W['output'] = rng.normal(0, .5, D*VOC).astype(np.float32)

q8b, DEQ = {}, {}
for k, v in W.items():
    b, d = q8_quant(v)
    q8b[k] = b; DEQ[k] = d
F32_KEYS = ['output_norm'] + ['blk%d.%s' % (l, nk) for l in range(NL)
                              for nk in ('attn_norm', 'ffn_norm')]
for k in F32_KEYS:
    q8b[k] = f32_arr(W[k]); DEQ[k] = W[k].astype(np.float32)

# ---- GGUF 组装 ----
NAMES = [('token_embd.weight', 'token_embd', [D, VOC], 8)]
for l in range(NL):
    NAMES += [
        ('blk.%d.attn_norm.weight' % l, 'blk%d.attn_norm' % l, [D], 0),
        ('blk.%d.attn_q.weight' % l, 'blk%d.attn_q' % l, [D, D], 8),
        ('blk.%d.attn_k.weight' % l, 'blk%d.attn_k' % l, [NKV*HD, D], 8),
        ('blk.%d.attn_v.weight' % l, 'blk%d.attn_v' % l, [NKV*HD, D], 8),
        ('blk.%d.attn_output.weight' % l, 'blk%d.attn_output' % l, [D, D], 8),
        ('blk.%d.ffn_norm.weight' % l, 'blk%d.ffn_norm' % l, [D], 0),
        ('blk.%d.ffn_gate.weight' % l, 'blk%d.ffn_gate' % l, [FF, D], 8),
        ('blk.%d.ffn_up.weight' % l, 'blk%d.ffn_up' % l, [FF, D], 8),
        ('blk.%d.ffn_down.weight' % l, 'blk%d.ffn_down' % l, [D, FF], 8),
    ]
NAMES += [('output_norm.weight', 'output_norm', [D], 0),
          ('output.weight', 'output', [D, VOC], 8)]

meta = (kv_str('general.architecture', 'llama') +
        kv_str('general.name', 'aimp-mini') +
        kv_u32('general.alignment', ALIGN) +
        kv_u32('llama.block_count', NL) +
        kv_u32('llama.context_length', CTX) +
        kv_u32('llama.embedding_length', D) +
        kv_u32('llama.attention.head_count', NH) +
        kv_u32('llama.attention.head_count_kv', NKV) +
        kv_u32('llama.attention.key_length', HD) +
        kv_u32('llama.feed_forward_length', FF) +
        kv_u32('llama.vocab_size', VOC) +
        kv_f32('llama.attention.layer_norm_rms_epsilon', EPS) +
        kv_f32('llama.rope.freq_base', ROPE_BASE) +
        kv_str('tokenizer.ggml.model', 'gpt2') +
        kv_arr_str('tokenizer.ggml.tokens',
                   ['<pad>', '<s>', '</s>'] +
                   ['t%d' % i for i in range(VOC - 3)]))

# STUB3
# ---- tensor layout ----
body, off = b'', 0
for gname, key, dims, qt in NAMES:
    nb = len(q8b[key])
    pad = (-off) % ALIGN
    body += b'\x00' * pad
    body += q8b[key]
    meta += tensor_info(gname, dims, qt, off)
    off += nb + pad

header = (struct.pack('<I', 0x46554747) + struct.pack('<I', 3) +
          struct.pack('<Q', len(NAMES)) + struct.pack('<Q', 15) + meta)
data_start = (len(header) + ALIGN - 1) // ALIGN * ALIGN

OUT = 'build/mini_llama.gguf'
with open(OUT, 'wb') as f:
    f.write(header)
    f.write(b'\x00' * (data_start - len(header)))
    f.write(body)
print('mini:', OUT, 'tensors=%d size=%d' % (len(NAMES), data_start + len(body)))

# ---- M2.7: AIM-W export (same weights) ----
# 容器 v1 (spec: aim-kernel/spec/aimw-format.md)；架构元数据以 fp32 标量张量携带
DT_FP32, DT_TERNP = 0, 2

def tern_quant(x):
    scale = np.float32(np.abs(x).max())
    codes = np.clip(np.round(x.astype(np.float64) / max(float(scale), 1e-12)),
                    -1, 1).astype(np.int8)
    return codes, scale

def pack_tern(codes):
    # spec 码表: 0→0, 1→+1, 2→-1, 3=保留；填充位必须 0；低位在前
    n = len(codes)
    m = np.where(codes > 0, 1, np.where(codes < 0, 2, 0)).astype(np.uint8)
    packed = np.zeros((n + 3) // 4, np.uint8)
    for i in range(n):
        packed[i >> 2] |= np.uint8(m[i] << np.uint8((i & 3) * 2))
    return packed.tobytes()

def write_aimw(path, entries):
    buf = bytearray(b'AIMW')
    buf += struct.pack('<II', 1, len(entries))
    for name, dt, dims, data in entries:
        nb = name.encode('utf-8')
        buf += struct.pack('<I', len(nb)) + nb
        buf += struct.pack('<I', dt) + struct.pack('<I', len(dims))
        for d in dims:
            buf += struct.pack('<I', int(d))
        buf += struct.pack('<I', len(data)) + data
    with open(path, 'wb') as f:
        f.write(bytes(buf))

TERN_KEYS = {'token_embd', 'output'}
for l in range(NL):
    TERN_KEYS |= {'blk%d.attn_q' % l, 'blk%d.attn_k' % l, 'blk%d.attn_v' % l,
                  'blk%d.attn_output' % l, 'blk%d.ffn_gate' % l,
                  'blk%d.ffn_up' % l, 'blk%d.ffn_down' % l}

def aimw_dims(gname, dims):
    # AIM-W dims 行主序 = C 侧内存解读:
    # token_embd/output 的 GGUF ne=[D,VOC] 行主=[VOC][D] -> reverse;
    # 其余矩阵 mini 约定 ne[0]=rows(=out) -> 原序
    if gname in ('token_embd.weight', 'output.weight'):
        return list(reversed(dims))
    return list(dims)

def aimw_entries(ternary):
    ents = []
    for gname, key, dims, _qt in NAMES:
        dims_aw = aimw_dims(gname, dims)
        if ternary and key in TERN_KEYS:
            codes, sc = tern_quant(W[key])
            ents.append((gname, DT_TERNP, dims_aw,
                         pack_tern(codes) + np.float32(sc).tobytes()))
        else:
            ents.append((gname, DT_FP32, dims_aw, W[key].tobytes()))
    meta = [('meta.nlayers', NL), ('meta.n_heads', NH), ('meta.n_kv_heads', NKV),
            ('meta.head_dim', HD), ('meta.n_ff', FF), ('meta.n_vocab', VOC),
            ('meta.ctx_len', CTX)]
    for k, v in meta:
        ents.append((k, DT_FP32, [1], np.float32(v).tobytes()))
    ents.append(('meta.eps', DT_FP32, [1], np.float32(EPS).tobytes()))
    ents.append(('meta.rope_base', DT_FP32, [1], np.float32(ROPE_BASE).tobytes()))
    return ents

write_aimw('build/mini_fp.aimw', aimw_entries(False))
write_aimw('build/mini_tern.aimw', aimw_entries(True))
print('aimw: build/mini_fp.aimw + build/mini_tern.aimw written')

# STUB4
# ---- numpy baseline (scalar contract, mirrors C sum order) ----
def rmsnorm(x, w):
    ss = np.float32(0)
    for i in range(len(x)):
        ss = np.float32(ss + np.float32(x[i] * x[i]))
    inv = np.float32(1.0) / np.float32(np.sqrt(np.float32(ss / len(x) + EPS)))
    return np.array([np.float32(np.float32(x[i] * inv) * w[i])
                     for i in range(len(x))], np.float32)

def matvec(w, h, rows, indim):
    w = np.asarray(w, np.float32).ravel()
    y = np.zeros(rows, np.float32)
    for o in range(rows):
        row = w[o*indim:(o+1)*indim]
        acc = np.float32(0)
        for b in range(indim // 32):
            part = np.float32(0)
            for i in range(32):
                part = np.float32(part + np.float32(h[b*32+i] * row[b*32+i]))
            acc = np.float32(acc + part)
        y[o] = acc
    return y

def rope_vec(v, pos):
    half = HD // 2
    r = v.copy()
    for j in range(half):
        theta = np.float32(ROPE_BASE) ** np.float32(-2.0 * j / HD)
        ang = np.float32(pos) * theta
        c = np.float32(np.cos(ang)); sn = np.float32(np.sin(ang))
        x0 = v[j]; x1 = v[j + half]
        r[j] = np.float32(x0 * c - x1 * sn)
        r[j + half] = np.float32(x0 * sn + x1 * c)
    return r

# STUB5
def forward(ids):
    n = len(ids)
    EMB = DEQ['token_embd'].reshape(VOC, D)
    x = np.stack([EMB[ids[p]] for p in range(n)])
    kv_dim = NKV * HD
    kc = np.zeros((NL, n, kv_dim), np.float32)
    vc = np.zeros((NL, n, kv_dim), np.float32)
    scale = np.float32(1.0 / np.sqrt(np.float32(HD)))
    for l in range(NL):
        xn = np.stack([rmsnorm(x[p], DEQ['blk%d.attn_norm' % l])
                       for p in range(n)])
        q = np.zeros((n, D), np.float32)
        for p in range(n):
            q[p] = matvec(DEQ['blk%d.attn_q' % l].reshape(D, D), xn[p], D, D)
            kc[l, p] = matvec(DEQ['blk%d.attn_k' % l].reshape(kv_dim, D), xn[p], kv_dim, D)
            vc[l, p] = matvec(DEQ['blk%d.attn_v' % l].reshape(kv_dim, D), xn[p], kv_dim, D)
        for p in range(n):
            for h in range(NH):
                q[p, h*HD:(h+1)*HD] = rope_vec(q[p, h*HD:(h+1)*HD], p)
            for h in range(NKV):
                kc[l, p, h*HD:(h+1)*HD] = rope_vec(kc[l, p, h*HD:(h+1)*HD], p)
        ao = np.zeros((n, D), np.float32)
        for p in range(n):
            for h in range(NH):
                kvh = h // (NH // NKV)
                qh = q[p, h*HD:(h+1)*HD]
                sc = np.zeros(p+1, np.float32)
                for t in range(p+1):
                    s2 = np.float32(0)
                    for i in range(HD):
                        s2 = np.float32(s2 + np.float32(qh[i] * kc[l, t, kvh*HD+i]))
                    sc[t] = np.float32(s2 * scale)
                mx = sc.max()
                e = np.array([np.float32(np.exp(np.float32(sc[t] - mx)))
                              for t in range(p+1)], np.float32)
                inv = np.float32(1.0 / np.float32(e.sum()))
                oh = np.zeros(HD, np.float32)
                for t in range(p+1):
                    w = np.float32(e[t] * inv)
                    for i in range(HD):
                        oh[i] = np.float32(oh[i] + np.float32(w * vc[l, t, kvh*HD+i]))
                ao[p, h*HD:(h+1)*HD] = oh
        for p in range(n):
            proj = matvec(DEQ['blk%d.attn_output' % l].reshape(D, D), ao[p], D, D)
            x[p] = np.array([np.float32(x[p][i] + proj[i]) for i in range(D)], np.float32)

# STUB6
        for p in range(n):
            xn2 = rmsnorm(x[p], DEQ['blk%d.ffn_norm' % l])
            gate = matvec(DEQ['blk%d.ffn_gate' % l].reshape(FF, D), xn2, FF, D)
            up = matvec(DEQ['blk%d.ffn_up' % l].reshape(FF, D), xn2, FF, D)
            act = np.zeros(FF, np.float32)
            for i in range(FF):
                g = gate[i]
                act[i] = np.float32(np.float32(g / np.float32(1.0 + np.exp(-g))) * up[i])
            dn = matvec(DEQ['blk%d.ffn_down' % l].reshape(D, FF), act, D, FF)
            x[p] = np.array([np.float32(x[p][i] + dn[i]) for i in range(D)], np.float32)
    xn3 = rmsnorm(x[n-1], DEQ['output_norm'])
    logits = matvec(DEQ['output'].reshape(VOC, D), xn3, VOC, D)
    return logits

def gen(ids, n_next):
    buf = list(ids)
    outs = []
    for _ in range(n_next):
        lg = forward(buf)
        outs.append(int(np.argmax(lg)))
        buf.append(outs[-1])
    return outs

# STUB7
# ---- C driver + assert ----
DEQ['token_embd'] = DEQ['token_embd'].reshape(VOC, D)
ref = gen(PROMPT, N_NEXT)
print('numpy ref :', ref)
r = subprocess.run(['build/aimp_probe.exe', OUT, '--gen',
                    ','.join(map(str, PROMPT)), str(N_NEXT)],
                   capture_output=True, text=True, errors='replace')
line = [l for l in r.stdout.splitlines() if l.startswith('GEN_IDS')]
ok = False
if line:
    got = [int(t) for t in line[0].replace('GEN_IDS:', '').split()]
    print('c engine  :', got)
    ok = (got == ref)
    if not ok:
        lg = forward(PROMPT)
        print('logit max diff check skipped (ids mismatch)')
print('==== M2 %s ====' % ('ALL_PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)