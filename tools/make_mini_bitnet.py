# -*- coding: utf-8 -*-
# aimp M2.9: BitNet 架构 mini 对账 - absmean 三值(TERNP)+Q8 embed+SubLN x2+relu2
# numpy 基线 vs C 引擎逐位; 复用 make_mini_v2 的标量序基线框架
import struct, subprocess, sys, os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')

NL, NH, NKV, HD = 1, 2, 1, 128
D = NH * HD          # 256
FF = 256
VOC = 256
CTX = 64
EPS = 1e-5
ROPE_BASE = 10000.0
PROMPT = [3, 7, 11, 5]
N_NEXT = 3
rng = np.random.default_rng(99)

# ---- absmean 三值量化 (BitNet 语义) ----
def absmean_quant(w):
    """codes=round(w/mean|w|) clip; 返回 (codes, gamma=mean|w|)"""
    gamma = np.float32(max(float(np.abs(w).mean()), 1e-5))
    codes = np.clip(np.round(w / gamma), -1, 1).astype(np.int8)
    return codes, gamma

def ternp_pack(codes, scale):
    n = len(codes)
    m = np.where(codes > 0, 1, np.where(codes < 0, 2, 0)).astype(np.uint8)
    pad = (-n) % 4
    if pad:
        m = np.concatenate([m, np.zeros(pad, np.uint8)])
    mp = m.reshape(-1, 4)
    packed = (mp[:, 0] | (mp[:, 1] << 2) | (mp[:, 2] << 4) | (mp[:, 3] << 6))
    return packed.astype(np.uint8).tobytes() + np.float32(scale).tobytes()

def q8_pack(w):
    n = len(w)
    nb = n // 32
    B = w.reshape(nb, 32)
    amax = np.abs(B).max(axis=1)
    s16 = (amax / np.float32(127.0)).astype(np.float16)
    s32 = s16.astype(np.float32)
    q = np.where(s32[:, None] > 0,
                 np.clip(np.round(B / np.maximum(s32[:, None], 1e-12)), -128, 127), 0.0)
    out = np.zeros((nb, 34), np.uint8)
    out[:, 0:2] = s16.view(np.uint8).reshape(nb, 2)
    out[:, 2:] = q.astype(np.int8).view(np.uint8).reshape(nb, 32)
    return out.tobytes(), s16.astype(np.float32)

# ---- 权重 ----
W = {}
W['token_embd.weight'] = rng.normal(0, .3, D * VOC).astype(np.float32)
for l in range(NL):
    W['blk%d.attn_norm' % l]      = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.attn_sub_norm' % l]  = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.attn_q' % l]         = rng.normal(0, .5, D * D).astype(np.float32)
    W['blk%d.attn_k' % l]         = rng.normal(0, .5, (NKV * HD) * D).astype(np.float32)
    W['blk%d.attn_v' % l]         = rng.normal(0, .5, (NKV * HD) * D).astype(np.float32)
    W['blk%d.attn_output' % l]    = rng.normal(0, .3, D * D).astype(np.float32)
    W['blk%d.ffn_norm' % l]       = (1 + rng.normal(0, .1, D)).astype(np.float32)
    W['blk%d.ffn_gate' % l]       = rng.normal(0, .5, FF * D).astype(np.float32)
    W['blk%d.ffn_up' % l]         = rng.normal(0, .5, FF * D).astype(np.float32)
    W['blk%d.ffn_sub_norm' % l]   = (1 + rng.normal(0, .1, FF)).astype(np.float32)
    W['blk%d.ffn_down' % l]       = rng.normal(0, .3, D * FF).astype(np.float32)
W['output_norm'] = (1 + rng.normal(0, .1, D)).astype(np.float32)
# 无 output.weight: tied embeddings

def f32_arr(x):
    return struct.pack('<%df' % len(x), *x)

# ---- 量化 ----
PAY, DEQ = {}, {}
F32_KEYS = ['output_norm'] + ['blk%d.%s' % (l, nk) for l in range(NL)
                              for nk in ('attn_norm', 'attn_sub_norm',
                                         'ffn_norm', 'ffn_sub_norm')]
for k, v in W.items():
    if k in F32_KEYS:
        PAY[k] = f32_arr(v)
        DEQ[k] = v.astype(np.float32)
    elif k == 'token_embd.weight':
        PAY[k], s32 = q8_pack(v)
        nb = len(v) // 32
        DEQ[k] = (s32.reshape(nb, 1) * np.frombuffer(
            PAY[k], np.uint8, len(v) // 32 * 34).reshape(nb, 34)[:, 2:]
            .copy().view(np.int8).astype(np.float32)).reshape(-1).astype(np.float32)
        err = np.linalg.norm(v - DEQ[k]) / np.linalg.norm(v)
        print('  %-22s dt=4 rel_err=%.4f' % (k, err))
    else:
        codes, gamma = absmean_quant(v)
        PAY[k] = ternp_pack(codes, gamma)
        DEQ[k] = codes.astype(np.float32) * gamma
        err = np.linalg.norm(v - DEQ[k]) / np.linalg.norm(v)
        print('  %-22s dt=2 rel_err=%.4f gamma=%.4f' % (k, err, gamma))

# ---- AIM-W 组装 ----
TENS = [('token_embd.weight', 'token_embd.weight', [VOC, D], 4)]
for l in range(NL):
    TENS += [
        ('blk.%d.attn_norm.weight' % l, 'blk%d.attn_norm' % l, [D], 0),
        ('blk.%d.attn_q.weight' % l, 'blk%d.attn_q' % l, [D, D], 2),
        ('blk.%d.attn_k.weight' % l, 'blk%d.attn_k' % l, [NKV * HD, D], 2),
        ('blk.%d.attn_v.weight' % l, 'blk%d.attn_v' % l, [NKV * HD, D], 2),
        ('blk.%d.attn_sub_norm.weight' % l, 'blk%d.attn_sub_norm' % l, [D], 0),
        ('blk.%d.attn_output.weight' % l, 'blk%d.attn_output' % l, [D, D], 2),
        ('blk.%d.ffn_norm.weight' % l, 'blk%d.ffn_norm' % l, [D], 0),
        ('blk.%d.ffn_gate.weight' % l, 'blk%d.ffn_gate' % l, [FF, D], 2),
        ('blk.%d.ffn_up.weight' % l, 'blk%d.ffn_up' % l, [FF, D], 2),
        ('blk.%d.ffn_sub_norm.weight' % l, 'blk%d.ffn_sub_norm' % l, [FF], 0),
        ('blk.%d.ffn_down.weight' % l, 'blk%d.ffn_down' % l, [D, FF], 2),
    ]
TENS += [('output_norm.weight', 'output_norm', [D], 0)]

META = [('meta.nlayers', NL), ('meta.n_heads', NH), ('meta.n_kv_heads', NKV),
        ('meta.head_dim', HD), ('meta.n_ff', FF), ('meta.n_vocab', VOC),
        ('meta.ctx_len', CTX), ('meta.eps', EPS), ('meta.rope_base', ROPE_BASE)]

buf = bytearray(b'AIMW')
buf += struct.pack('<II', 1, len(TENS) + len(META))
for name, wk, dims, dt in TENS:
    nb = name.encode('utf-8')
    buf += struct.pack('<I', len(nb)) + nb
    buf += struct.pack('<I', dt) + struct.pack('<I', len(dims))
    for dd in dims:
        buf += struct.pack('<I', dd)
    payload = PAY[wk]
    buf += struct.pack('<I', len(payload)) + payload
for k, v in META:
    kb = k.encode('utf-8')
    buf += struct.pack('<I', len(kb)) + kb
    buf += struct.pack('<I', 0) + struct.pack('<I', 1) + struct.pack('<I', 1)
    buf += struct.pack('<I', 4) + struct.pack('<f', float(v))

OUT = 'build/mini_bitnet.aimw'
with open(OUT, 'wb') as f:
    f.write(bytes(buf))
print('mini bitnet aimw:', OUT, 'size=%d' % len(buf))

# ---- numpy 基线 (BitNet 前向: sub_norms + relu2, 求和序全局 i 升序) ----
def rmsnorm(x, w):
    ss = np.float32(0)
    for i in range(len(x)):
        ss = np.float32(ss + np.float32(x[i] * x[i]))
    inv = np.float32(1.0) / np.float32(np.sqrt(np.float32(ss / len(x) + EPS)))
    return np.array([np.float32(np.float32(x[i] * inv) * w[i])
                     for i in range(len(x))], np.float32)

def matvec(w, h, rows, indim, ternp=False):
    """ternp=1: 镜像 C 快路序. indim%32==0: 链4×8序(v2, 4 独立累加链拆延迟墙);
    否则: v1 8 车道序 —— 车道内 i 升序, 合并 ((a0+a1)+a2)+a3 后尾段串行"""
    w = np.asarray(w, np.float32).ravel()
    y = np.zeros(rows, np.float32)
    for o in range(rows):
        base = o * indim
        if ternp:
            if indim % 32 == 0:
                # v2 链4×8序: acc[c][j] 累计 i=8t+j (t≡c mod 4) 升序
                acc = [np.float32(0)] * 32
                for t in range(indim // 8):
                    c4 = (t % 4) * 8
                    for j in range(8):
                        i = t * 8 + j
                        acc[c4 + j] = np.float32(acc[c4 + j] + np.float32(h[i] * w[base + i]))
                s = [np.float32(0)] * 8
                for j in range(8):
                    v = np.float32(acc[j] + acc[8 + j])
                    v = np.float32(v + acc[16 + j])
                    s[j] = np.float32(v + acc[24 + j])
                accs = np.float32(s[0] + s[1])
                for j in range(2, 8):
                    accs = np.float32(accs + s[j])
                y[o] = accs
                continue
            a0 = a1 = a2 = a3 = np.float32(0)
            a4 = a5 = a6 = a7 = np.float32(0)
            i = 0
            while i + 8 <= indim:
                a0 = np.float32(a0 + np.float32(h[i] * w[base + i]))
                a1 = np.float32(a1 + np.float32(h[i + 1] * w[base + i + 1]))
                a2 = np.float32(a2 + np.float32(h[i + 2] * w[base + i + 2]))
                a3 = np.float32(a3 + np.float32(h[i + 3] * w[base + i + 3]))
                a4 = np.float32(a4 + np.float32(h[i + 4] * w[base + i + 4]))
                a5 = np.float32(a5 + np.float32(h[i + 5] * w[base + i + 5]))
                a6 = np.float32(a6 + np.float32(h[i + 6] * w[base + i + 6]))
                a7 = np.float32(a7 + np.float32(h[i + 7] * w[base + i + 7]))
                i += 8
            acc = np.float32(a0 + a1)
            acc = np.float32(acc + a2)
            acc = np.float32(acc + a3)
            acc = np.float32(acc + a4)
            acc = np.float32(acc + a5)
            acc = np.float32(acc + a6)
            acc = np.float32(acc + a7)
            for j in range(i, indim):
                acc = np.float32(acc + np.float32(h[j] * w[base + j]))
            y[o] = acc
        else:
            # Q8 (v2q 链4×8序): 镜像 mv_row_q8 AVX2 —— 链 ch 车道 j 累计
            # i=32b+8k+j (b≡ch mod 4, k=0..3) 升序, 车道内 ((a0+a1)+a2)+a3,
            # 横向左折; 每码 (q*scale 先舍入)=DEQ, ×h 后舍入
            nb = indim // 32
            if indim % 32 == 0 and nb % 4 == 0:
                acc = [np.float32(0)] * 32
                for b in range(0, nb, 4):
                    for ch in range(4):
                        for k in range(4):
                            for j in range(8):
                                i = (b + ch) * 32 + k * 8 + j
                                acc[ch * 8 + j] = np.float32(
                                    acc[ch * 8 + j] + np.float32(h[i] * w[base + i]))
                s = [np.float32(0)] * 8
                for j in range(8):
                    v = np.float32(acc[j] + acc[8 + j])
                    v = np.float32(v + acc[16 + j])
                    s[j] = np.float32(v + acc[24 + j])
                accs = np.float32(s[0] + s[1])
                for j in range(2, 8):
                    accs = np.float32(accs + s[j])
                y[o] = accs
                continue
            acc = np.float32(0)
            for i in range(indim):
                acc = np.float32(acc + np.float32(h[i] * w[base + i]))
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
            q[p] = matvec(DEQ['blk%d.attn_q' % l].reshape(D, D), xn[p], D, D, ternp=True)
            kc[l, p] = matvec(DEQ['blk%d.attn_k' % l].reshape(kv_dim, D), xn[p], kv_dim, D, ternp=True)
            vc[l, p] = matvec(DEQ['blk%d.attn_v' % l].reshape(kv_dim, D), xn[p], kv_dim, D, ternp=True)
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
            ao2 = rmsnorm(ao[p], DEQ['blk%d.attn_sub_norm' % l])  # BitNet SubLN
            proj = matvec(DEQ['blk%d.attn_output' % l].reshape(D, D), ao2, D, D, ternp=True)
            x[p] = np.array([np.float32(x[p][i] + proj[i]) for i in range(D)], np.float32)
        for p in range(n):
            xn2 = rmsnorm(x[p], DEQ['blk%d.ffn_norm' % l])
            gate = matvec(DEQ['blk%d.ffn_gate' % l].reshape(FF, D), xn2, FF, D, ternp=True)
            up = matvec(DEQ['blk%d.ffn_up' % l].reshape(FF, D), xn2, FF, D, ternp=True)
            act = np.zeros(FF, np.float32)
            for i in range(FF):
                g = gate[i]
                act[i] = np.float32((g * g if g > 0 else np.float32(0)) * up[i])  # relu2
            act = rmsnorm(act, DEQ['blk%d.ffn_sub_norm' % l])  # BitNet SubLN
            dn = matvec(DEQ['blk%d.ffn_down' % l].reshape(D, FF), act, D, FF, ternp=True)
            x[p] = np.array([np.float32(x[p][i] + dn[i]) for i in range(D)], np.float32)
    xn3 = rmsnorm(x[n - 1], DEQ['output_norm'])
    return matvec(DEQ['token_embd.weight'].reshape(VOC, D), xn3, VOC, D)  # tied

def gen(ids, n_next):
    buf = list(ids)
    outs = []
    for _ in range(n_next):
        lg = forward(buf)
        outs.append(int(np.argmax(lg)))
        buf.append(outs[-1])
    return outs

ref = gen(PROMPT, N_NEXT)
print('numpy bitnet ref:', ref)
r = subprocess.run(['build/aimp_probe.exe', OUT, '--gen',
                    ','.join(map(str, PROMPT)), str(N_NEXT)],
                   capture_output=True, text=True, errors='replace')
line = [l for l in r.stdout.splitlines() if l.startswith('GEN_IDS')]
ok = False
if line:
    got = [int(t) for t in line[0].replace('GEN_IDS:', '').split()]
    print('c engine bitnet :', got)
    ok = got == ref
else:
    print('probe stdout:', r.stdout[-300:])
    print('probe stderr:', r.stderr[-300:])
print('==== M2.9 %s ====' % ('ALL_PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
