# -*- coding: utf-8 -*-
# aimp M2.8: AIM-W v2 mini 对账 - mini 模型 v2 量化 -> .aimw -> numpy 基线 vs C 引擎逐位
# 复用 make_mini_q4k 的权重分布/求和序基线; 量化走 gguf2aimw_v2.quant_v2 (与 C 逐位对齐语义)
import struct, subprocess, sys, os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw_v2 import quant_v2

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

# ---- 权重 (同 make_mini_q4k 分布) ----
W = {}
W['token_embd.weight'] = rng.normal(0, 1, D * VOC).astype(np.float32)
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
W['output.weight'] = rng.normal(0, .5, D * VOC).astype(np.float32)

def f32_arr(x):
    return struct.pack('<%df' % len(x), *x)

# ---- v2 量化 ----
PAY, DEQ = {}, {}
F32_KEYS = ['output_norm'] + ['blk%d.%s' % (l, nk) for l in range(NL)
                              for nk in ('attn_norm', 'ffn_norm')]
for k, v in W.items():
    if k in F32_KEYS:
        PAY[k] = f32_arr(v)
        DEQ[k] = v.astype(np.float32)
    else:
        payload, cb, deq, err = quant_v2(v)
        PAY[k] = payload
        DEQ[k] = deq
        print('  %-22s dt=3 rel_err=%.4f lut0=%.4f lut15=%.4f' % (k, err, cb[0], cb[-1]))

# ---- AIM-W v1 组装 ----
def s(v):
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def meta_kv(k, v):
    # AIM-W 名前缀是 u32 (GGUF 才是 u64, 曾用 s() 的 8B 前缀致 meta 段错位)
    kb = k.encode('utf-8')
    return struct.pack('<I', len(kb)) + kb \
         + struct.pack('<I', 0) + struct.pack('<I', 1) + struct.pack('<I', 1) \
         + struct.pack('<I', 4) + struct.pack('<f', float(v))

TENS = [('token_embd.weight', 'token_embd.weight', [VOC, D])]
for l in range(NL):
    TENS += [
        ('blk.%d.attn_norm.weight' % l, 'blk%d.attn_norm' % l, [D]),
        ('blk.%d.attn_q.weight' % l, 'blk%d.attn_q' % l, [D, D]),
        ('blk.%d.attn_k.weight' % l, 'blk%d.attn_k' % l, [NKV * HD, D]),
        ('blk.%d.attn_v.weight' % l, 'blk%d.attn_v' % l, [NKV * HD, D]),
        ('blk.%d.attn_output.weight' % l, 'blk%d.attn_output' % l, [D, D]),
        ('blk.%d.ffn_norm.weight' % l, 'blk%d.ffn_norm' % l, [D]),
        ('blk.%d.ffn_gate.weight' % l, 'blk%d.ffn_gate' % l, [FF, D]),
        ('blk.%d.ffn_up.weight' % l, 'blk%d.ffn_up' % l, [FF, D]),
        ('blk.%d.ffn_down.weight' % l, 'blk%d.ffn_down' % l, [D, FF]),
    ]
TENS += [('output_norm.weight', 'output_norm', [D]),
         ('output.weight', 'output.weight', [VOC, D])]

META = [('meta.nlayers', NL), ('meta.n_heads', NH), ('meta.n_kv_heads', NKV),
        ('meta.head_dim', HD), ('meta.n_ff', FF), ('meta.n_vocab', VOC),
        ('meta.ctx_len', CTX), ('meta.eps', EPS), ('meta.rope_base', ROPE_BASE)]

buf = bytearray(b'AIMW')
buf += struct.pack('<II', 1, len(TENS) + len(META))
for name, wk, dims in TENS:
    nb = name.encode('utf-8')
    buf += struct.pack('<I', len(nb)) + nb
    buf += struct.pack('<I', 3 if wk not in F32_KEYS else 0)
    buf += struct.pack('<I', len(dims))
    for dd in dims:
        buf += struct.pack('<I', dd)
    payload = PAY[wk]
    buf += struct.pack('<I', len(payload)) + payload
for k, v in META:
    buf += meta_kv(k, v)

OUT = 'build/mini_v2.aimw'
with open(OUT, 'wb') as f:
    f.write(bytes(buf))
print('mini v2 aimw:', OUT, 'size=%d' % len(buf))

# ---- numpy 基线 (求和序: 全局 i 升序, 与 C 逐位一致) ----
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
    EMB = DEQ['token_embd.weight'].reshape(VOC, D)
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
    return matvec(DEQ['output.weight'].reshape(VOC, D), xn3, VOC, D)

def gen(ids, n_next):
    buf = list(ids)
    outs = []
    for _ in range(n_next):
        lg = forward(buf)
        outs.append(int(np.argmax(lg)))
        buf.append(outs[-1])
    return outs

ref = gen(PROMPT, N_NEXT)
print('numpy v2 ref:', ref)
r = subprocess.run(['build/aimp_probe.exe', OUT, '--gen',
                    ','.join(map(str, PROMPT)), str(N_NEXT)],
                   capture_output=True, text=True, errors='replace')
line = [l for l in r.stdout.splitlines() if l.startswith('GEN_IDS')]
ok = False
if line:
    got = [int(t) for t in line[0].replace('GEN_IDS:', '').split()]
    print('c engine v2 :', got)
    ok = got == ref
else:
    print('probe stdout:', r.stdout[-500:])
    print('probe stderr:', r.stderr[-500:])
print('==== M2.8 %s ====' % ('ALL_PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
