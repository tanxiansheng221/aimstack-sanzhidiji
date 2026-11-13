# -*- coding: utf-8 -*-
# aimp M2.7c: Q4_K 对账 - mini 放大版(行宽 256 倍数) GGUF(qtype=12) + numpy 基线
# 量化器(自洽简化版): per-superblock d_total = maxabs/8, 8 子块 6bit sc 相对细分,
# 对称无偏移 (dmin=0): w_deq = d_total*sc*(n-8)
# 求和序契约: 全局 i 升序逐位累加 (与 C matvec_q4k 逐位一致)
import os, struct, subprocess, sys
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ALIGN = 32
NL, NH, NKV, HD = 1, 2, 1, 128
D = NH * HD          # 256
FF = 256
VOC = 256
CTX = 64
EPS = 1e-5
ROPE_BASE = 10000.0
PROMPT = [3, 7, 11, 5]
N_NEXT = 3
rng = np.random.default_rng(77)

def s(v):
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def kv_str(k, v): return s(k) + struct.pack('<I', 8) + s(v)
def kv_u32(k, v): return s(k) + struct.pack('<I', 4) + struct.pack('<I', v)
def kv_f32(k, v): return s(k) + struct.pack('<I', 6) + struct.pack('<f', v)

def f32_arr(x):
    return struct.pack('<%df' % len(x), *x)

# ---- 权重 ----
W = {}
W['token_embd'] = rng.normal(0, 1, D * VOC).astype(np.float32)
for l in range(NL):
    W['blk%d.attn_norm' % l] = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.attn_q' % l]      = rng.normal(0, .5, D * D).astype(np.float32)
    W['blk%d.attn_k' % l]      = rng.normal(0, .5, (NKV * HD) * D).astype(np.float32)
    W['blk%d.attn_v' % l]      = rng.normal(0, .5, (NKV * HD) * D).astype(np.float32)
    W['blk%d.attn_output' % l] = rng.normal(0, .3, D * D).astype(np.float32)
    W['blk%d.ffn_norm' % l]    = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.ffn_gate' % l]    = rng.normal(0, .5, FF * D).astype(np.float32)
    W['blk%d.ffn_up' % l]      = rng.normal(0, .5, FF * D).astype(np.float32)
    W['blk%d.ffn_down' % l]    = rng.normal(0, .3, D * FF).astype(np.float32)
W['output_norm'] = (1 + rng.normal(0, .1, D)).astype(np.float32)
W['output'] = rng.normal(0, .5, D * VOC).astype(np.float32)

# ---- Q4_K 量化 (llama.cpp 非对称语义: w = d*sc*n - dmin*m, n∈[0,15]) ----
def q4k_quant(w):
    n = len(w)
    assert n % 256 == 0
    out = bytearray()
    deq = np.empty(n, np.float32)
    for s in range(n // 256):
        blk = w[s * 256:(s + 1) * 256]
        # 1) 每子块区间 [wmin, wmax] -> eff_d, off
        eff = np.zeros(8, np.float32)
        offs = np.zeros(8, np.float32)
        for i in range(8):
            sub = blk[i * 32:(i + 1) * 32]
            wmax = float(sub.max()); wmin = float(sub.min())
            if wmax > wmin:
                eff[i] = np.float32((wmax - wmin) / 15.0)
                offs[i] = np.float32(wmin)
            else:
                offs[i] = np.float32(wmin)   # 平坦子块: eff=0, 全部 == off
        d_total = np.float16(float(eff.max()))
        dmin_total = np.float16(float(offs.max()))
        # 2) 6bit 相对系数 (反推值以 C 同序公式为准)
        scs = [min(63, max(0, int(round(63.0 * float(eff[i]) / float(d_total)))))
               if d_total > 0 else 0 for i in range(8)]
        ms = [min(63, max(0, int(round(63.0 * float(offs[i]) / float(dmin_total)))))
              if dmin_total > 0 else 0 for i in range(8)]
        # 3) 反推 eff'/off' (C 序: f32(d_total)*sc) 并量化 nibble
        d32 = np.float32(d_total)
        dmin32 = np.float32(dmin_total)
        # scales 12B 布局 (由 get_scale_min_k4 反推):
        # q[0..3]  = sc[j]&63 | (sc[j+4]>>4)<<6
        # q[4..7]  = m[j]&63  | (m[j+4]>>4)<<6
        # q[8..11] = sc[j+4]&15 | (m[j+4]&15)<<4
        scb = bytearray(12)
        for j in range(4):
            scb[j] = (scs[j] & 63) | ((scs[4 + j] >> 4) << 6)
            scb[4 + j] = (ms[j] & 63) | ((ms[4 + j] >> 4) << 6)
            scb[8 + j] = (scs[4 + j] & 15) | ((ms[4 + j] & 15) << 4)
        qs = bytearray(128)
        for j in range(8):
            d1 = np.float32(d32 * np.float32(scs[j]))
            m1 = np.float32(dmin32 * np.float32(ms[j]))
            sub = blk[j * 32:(j + 1) * 32]
            if d1 > 0:
                nq = np.clip(np.round((sub - m1) / d1), 0, 15).astype(np.uint8)
            else:
                nq = np.zeros(32, np.uint8)
            g = j // 2
            half = j % 2
            base = g * 32
            for l in range(32):
                if half == 0:
                    qs[base + l] |= nq[l]
                else:
                    qs[base + l] |= np.uint8(nq[l] << 4)
            for l in range(32):
                deq[s * 256 + j * 32 + l] = np.float32(d1 * np.float32(int(nq[l])) - m1)
        out += struct.pack('<e', float(d_total))
        out += struct.pack('<e', float(dmin_total))
        out += bytes(scb)
        out += bytes(qs)
    return bytes(out), deq

Q4B, DEQ = {}, {}
F32_KEYS = ['output_norm'] + ['blk%d.%s' % (l, nk) for l in range(NL)
                              for nk in ('attn_norm', 'ffn_norm')]
for k, v in W.items():
    if k in F32_KEYS:
        Q4B[k] = f32_arr(v)
        DEQ[k] = v.astype(np.float32)
    else:
        b, d = q4k_quant(v)
        Q4B[k] = b
        DEQ[k] = d

# ---- GGUF 组装 ----
NAMES = [('token_embd.weight', 'token_embd', [D, VOC], 12)]
for l in range(NL):
    NAMES += [
        ('blk.%d.attn_norm.weight' % l, 'blk%d.attn_norm' % l, [D], 0),
        ('blk.%d.attn_q.weight' % l, 'blk%d.attn_q' % l, [D, D], 12),
        ('blk.%d.attn_k.weight' % l, 'blk%d.attn_k' % l, [NKV * HD, D], 12),
        ('blk.%d.attn_v.weight' % l, 'blk%d.attn_v' % l, [NKV * HD, D], 12),
        ('blk.%d.attn_output.weight' % l, 'blk%d.attn_output' % l, [D, D], 12),
        ('blk.%d.ffn_norm.weight' % l, 'blk%d.ffn_norm' % l, [D], 0),
        ('blk.%d.ffn_gate.weight' % l, 'blk%d.ffn_gate' % l, [FF, D], 12),
        ('blk.%d.ffn_up.weight' % l, 'blk%d.ffn_up' % l, [FF, D], 12),
        ('blk.%d.ffn_down.weight' % l, 'blk%d.ffn_down' % l, [D, FF], 12),
    ]
NAMES += [('output_norm.weight', 'output_norm', [D], 0),
          ('output.weight', 'output', [D, VOC], 12)]

def tensor_info(name, dims, qt, offset):
    b = s(name) + struct.pack('<I', len(dims))
    for d in dims:
        b += struct.pack('<Q', d)
    return b + struct.pack('<I', qt) + struct.pack('<Q', offset)

meta = (kv_str('general.architecture', 'llama') +
        kv_str('general.name', 'aimp-mini-q4k') +
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
        kv_f32('llama.rope.freq_base', ROPE_BASE))

body, off = b'', 0
for gname, key, dims, qt in NAMES:
    nb = len(Q4B[key])
    pad = (-off) % ALIGN
    body += b'\x00' * pad
    body += Q4B[key]
    meta += tensor_info(gname, dims, qt, off)
    off += nb + pad

header = (struct.pack('<I', 0x46554747) + struct.pack('<I', 3) +
          struct.pack('<Q', len(NAMES)) + struct.pack('<Q', 13) + meta)
data_start = (len(header) + ALIGN - 1) // ALIGN * ALIGN

OUT = 'build/mini_q4k.gguf'
with open(OUT, 'wb') as f:
    f.write(header)
    f.write(b'\x00' * (data_start - len(header)))
    f.write(body)
print('mini q4k:', OUT, 'tensors=%d size=%d' % (len(NAMES), data_start + len(body)))

# ---- numpy 基线 (求和序: 全局 i 升序) ----
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
        acc = np.float32(0)
        for i in range(indim):
            acc = np.float32(acc + np.float32(h[i] * w[o * indim + i]))
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

def forward(ids):
    n = len(ids)
    EMB = DEQ['token_embd'].reshape(VOC, D)
    x = np.stack([EMB[ids[p]] for p in range(n)])
    kv_dim = NKV * HD
    kc = np.zeros((NL, n, kv_dim), np.float32)
    vc = np.zeros((NL, n, kv_dim), np.float32)
    scale = np.float32(1.0 / np.sqrt(np.float32(HD)))
    for l in range(NL):
        xn = np.stack([rmsnorm(x[p], DEQ['blk%d.attn_norm' % l]) for p in range(n)])
        q = np.zeros((n, D), np.float32)
        for p in range(n):
            q[p] = matvec(DEQ['blk%d.attn_q' % l].reshape(D, D), xn[p], D, D)
            kc[l, p] = matvec(DEQ['blk%d.attn_k' % l].reshape(kv_dim, D), xn[p], kv_dim, D)
            vc[l, p] = matvec(DEQ['blk%d.attn_v' % l].reshape(kv_dim, D), xn[p], kv_dim, D)
        for p in range(n):
            for h in range(NH):
                q[p, h * HD:(h + 1) * HD] = rope_vec(q[p, h * HD:(h + 1) * HD], p)
            for h in range(NKV):
                kc[l, p, h * HD:(h + 1) * HD] = rope_vec(kc[l, p, h * HD:(h + 1) * HD], p)
        ao = np.zeros((n, D), np.float32)
        for p in range(n):
            for h in range(NH):
                kvh = h // (NH // NKV)
                qh = q[p, h * HD:(h + 1) * HD]
                sc = np.zeros(p + 1, np.float32)
                for t in range(p + 1):
                    s2 = np.float32(0)
                    for i in range(HD):
                        s2 = np.float32(s2 + np.float32(qh[i] * kc[l, t, kvh * HD + i]))
                    sc[t] = np.float32(s2 * scale)
                mx = sc.max()
                e = np.array([np.float32(np.exp(np.float32(sc[t] - mx)))
                              for t in range(p + 1)], np.float32)
                inv = np.float32(1.0 / np.float32(e.sum()))
                oh = np.zeros(HD, np.float32)
                for t in range(p + 1):
                    w = np.float32(e[t] * inv)
                    for i in range(HD):
                        oh[i] = np.float32(oh[i] + np.float32(w * vc[l, t, kvh * HD + i]))
                ao[p, h * HD:(h + 1) * HD] = oh
        for p in range(n):
            proj = matvec(DEQ['blk%d.attn_output' % l].reshape(D, D), ao[p], D, D)
            x[p] = np.array([np.float32(x[p][i] + proj[i]) for i in range(D)], np.float32)
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
    xn3 = rmsnorm(x[n - 1], DEQ['output_norm'])
    return matvec(DEQ['output'].reshape(VOC, D), xn3, VOC, D)

def gen(ids, n_next):
    buf = list(ids)
    outs = []
    for _ in range(n_next):
        lg = forward(buf)
        outs.append(int(np.argmax(lg)))
        buf.append(outs[-1])
    return outs

ref = gen(PROMPT, N_NEXT)
print('numpy q4k ref:', ref)
r = subprocess.run(['build/aimp_probe.exe', OUT, '--gen',
                    ','.join(map(str, PROMPT)), str(N_NEXT)],
                   capture_output=True, text=True, errors='replace')
line = [l for l in r.stdout.splitlines() if l.startswith('GEN_IDS')]
ok = False
if line:
    got = [int(t) for t in line[0].replace('GEN_IDS:', '').split()]
    print('c engine q4k :', got)
    ok = got == ref
print('==== M2.7c %s ====' % ('ALL_PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
