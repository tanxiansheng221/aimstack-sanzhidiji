# -*- coding: utf-8 -*-
# aimp M2.7 对账: AIM-W 双档 (fp32 / ternary packed) C 引擎 vs NumPy 逐元素基线
# 求和序契约: out[o] = sum_i x[i]*w[o][i], i 升序, 全程 np.float32 逐次舍入
import os, subprocess, sys
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, r'H:\AimCode\aim-kernel\ref')
from aim_ref import load_aimw

EXE = 'build/aimp_probe.exe'
PROMPT = [3, 7, 11, 5]
N_NEXT = 3

def build_funcs(t):
    f32 = lambda k: t[k][2]
    NL  = int(round(float(f32('meta.nlayers')[0])))
    NH  = int(round(float(f32('meta.n_heads')[0])))
    NKV = int(round(float(f32('meta.n_kv_heads')[0])))
    HD  = int(round(float(f32('meta.head_dim')[0])))
    FF  = int(round(float(f32('meta.n_ff')[0])))
    EPS = float(f32('meta.eps')[0])
    ROPE_BASE = float(f32('meta.rope_base')[0])
    te_dims = t['token_embd.weight'][1]        # [VOC, D] 行主序
    VOC, D = te_dims[0], te_dims[1]
    kv_dim = NKV * HD

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
            row = w[o * indim:(o + 1) * indim]
            acc = np.float32(0)
            for i in range(indim):
                acc = np.float32(acc + np.float32(h[i] * row[i]))
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
        EMB = f32('token_embd.weight').reshape(VOC, D)
        x = np.stack([EMB[ids[p]] for p in range(n)])
        kc = np.zeros((NL, n, kv_dim), np.float32)
        vc = np.zeros((NL, n, kv_dim), np.float32)
        scale = np.float32(1.0 / np.sqrt(np.float32(HD)))
        for l in range(NL):
            xn = np.stack([rmsnorm(x[p], f32('blk.%d.attn_norm.weight' % l))
                           for p in range(n)])
            q = np.zeros((n, D), np.float32)
            for p in range(n):
                q[p] = matvec(f32('blk.%d.attn_q.weight' % l).reshape(D, D), xn[p], D, D)
                kc[l, p] = matvec(f32('blk.%d.attn_k.weight' % l).reshape(kv_dim, D), xn[p], kv_dim, D)
                vc[l, p] = matvec(f32('blk.%d.attn_v.weight' % l).reshape(kv_dim, D), xn[p], kv_dim, D)
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
                    for tt in range(p + 1):
                        s2 = np.float32(0)
                        for i in range(HD):
                            s2 = np.float32(s2 + np.float32(qh[i] * kc[l, tt, kvh * HD + i]))
                        sc[tt] = np.float32(s2 * scale)
                    mx = sc.max()
                    e = np.array([np.float32(np.exp(np.float32(sc[tt] - mx)))
                                  for tt in range(p + 1)], np.float32)
                    inv = np.float32(1.0 / np.float32(e.sum()))
                    oh = np.zeros(HD, np.float32)
                    for tt in range(p + 1):
                        w = np.float32(e[tt] * inv)
                        for i in range(HD):
                            oh[i] = np.float32(oh[i] + np.float32(w * vc[l, tt, kvh * HD + i]))
                    ao[p, h * HD:(h + 1) * HD] = oh
            for p in range(n):
                proj = matvec(f32('blk.%d.attn_output.weight' % l).reshape(D, D), ao[p], D, D)
                x[p] = np.array([np.float32(x[p][i] + proj[i]) for i in range(D)], np.float32)
            for p in range(n):
                xn2 = rmsnorm(x[p], f32('blk.%d.ffn_norm.weight' % l))
                gate = matvec(f32('blk.%d.ffn_gate.weight' % l).reshape(FF, D), xn2, FF, D)
                up = matvec(f32('blk.%d.ffn_up.weight' % l).reshape(FF, D), xn2, FF, D)
                act = np.zeros(FF, np.float32)
                for i in range(FF):
                    g = gate[i]
                    act[i] = np.float32(np.float32(g / np.float32(1.0 + np.exp(-g))) * up[i])
                dn = matvec(f32('blk.%d.ffn_down.weight' % l).reshape(D, FF), act, D, FF)
                x[p] = np.array([np.float32(x[p][i] + dn[i]) for i in range(D)], np.float32)
        xn3 = rmsnorm(x[n - 1], f32('output_norm.weight'))
        return matvec(f32('output.weight').reshape(VOC, D), xn3, VOC, D)

    def gen(ids, n_next):
        buf = list(ids)
        outs = []
        for _ in range(n_next):
            lg = forward(buf)
            outs.append(int(np.argmax(lg)))
            buf.append(outs[-1])
        return outs

    return gen

def run_case(path, tag):
    t = load_aimw(path)
    ref = build_funcs(t)(PROMPT, N_NEXT)
    print('%s numpy ref : %s' % (tag, ref))
    r = subprocess.run([EXE, path, '--gen', ','.join(map(str, PROMPT)), str(N_NEXT)],
                       capture_output=True, text=True, errors='replace')
    line = [l for l in r.stdout.splitlines() if l.startswith('GEN_IDS')]
    if not line:
        print('%s C engine  : FAIL rc=%d %s' % (tag, r.returncode, r.stderr.strip()))
        return False
    got = [int(x) for x in line[0].replace('GEN_IDS:', '').split()]
    print('%s C engine  : %s' % (tag, got))
    ok = got == ref
    print('%s %s' % (tag, 'PASS' if ok else 'MISMATCH'))
    return ok

ok1 = run_case('build/mini_fp.aimw', 'fp32  ')
ok2 = run_case('build/mini_tern.aimw', 'tern  ')
print('==== M2.7 %s ====' % ('ALL_PASS' if ok1 and ok2 else 'FAIL'))
sys.exit(0 if ok1 and ok2 else 1)
