# -*- coding: utf-8 -*-
# aimp M2.8: GGUF -> AIM-W v2 转换器 (dtype=3: Q4_K 容器 + 全张量 Lloyd 学习型点阵)
# 量化语义与 C 引擎逐位对齐:
#   d_stored=f16(max_eff/63), sc_j=clip(round(eff_j/d_stored),0,63)  -> C: d1=f32(d)*sc
#   dmin_stored=f16(max_offm/63), m_j 同理                            -> C: m1=f32(dmin)*m
#   z=(w+m1)/d1 -> Lloyd 16 中心(存 f16) -> n=最近中心索引 -> deq=d1*LUT[n]-m1
# 用法: python gguf2aimw_v2.py src.gguf dst.aimw [--uniform]
#   --uniform: LUT={0..15}, 退化为 Q4_K 语义 (对照基线)
import struct, sys, os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'tools')
from gguf2aimw import read_gguf, deq_q8

UNIFORM = '--uniform' in sys.argv

def f16(v):
    return np.float16(v)

def lloyd_1d(z, k=16, iters=12):
    """全局 1D Lloyd, 分块处理; 返回升序中心 (f32, 已 f16 圆整)"""
    qs = np.sort(np.quantile(z, np.linspace(0.01, 0.99, k))).astype(np.float32)
    CH = 1 << 23
    for _ in range(iters):
        mid = ((qs[:-1] + qs[1:]) * np.float32(0.5))
        sums = np.zeros(k, np.float64)
        cnt = np.zeros(k, np.int64)
        for s in range(0, len(z), CH):
            b = z[s:s + CH]
            idx = np.clip(np.searchsorted(mid, b), 0, k - 1)
            cnt += np.bincount(idx, minlength=k)
            sums += np.bincount(idx, weights=b.astype(np.float64), minlength=k)
        qs = np.sort(np.where(cnt > 0, sums / np.maximum(cnt, 1), qs).astype(np.float32))
    return np.sort(qs.astype(np.float16).astype(np.float32))  # f16 圆整后重排

def quant_v2(w):
    """单张量 -> (payload bytes, codebook f16 x16, deq values, rel_err)"""
    n = len(w)
    assert n % 256 == 0, 'dim not multiple of 256: %d' % n
    NB = n // 256
    B = w.reshape(NB, 8, 32)
    wmax = B.max(axis=2)
    wmin = B.min(axis=2)
    eff = (wmax - wmin) / np.float32(15.0)
    offm = np.maximum(-wmin, np.float32(0))
    # super scale (f16 圆整, C 端逐位一致)
    d_stored = f16(eff.max(axis=1) / np.float32(63.0))
    dmin_stored = f16(offm.max(axis=1) / np.float32(63.0))
    d32 = d_stored.astype(np.float32)
    dmin32 = dmin_stored.astype(np.float32)
    sc = np.clip(np.round(eff / np.maximum(d32[:, None], 1e-12)), 0, 63).astype(np.int32)
    ms = np.clip(np.round(offm / np.maximum(dmin32[:, None], 1e-12)), 0, 63).astype(np.int32)
    d1 = d32[:, None] * sc.astype(np.float32)      # (NB,8)
    m1 = dmin32[:, None] * ms.astype(np.float32)
    # z 域点阵学习 (C 反推值语义)
    def z_chunks():
        CH = 1 << 21
        for s in range(0, NB, CH):
            zz = (B[s:s + CH] + m1[s:s + CH, :, None]) / np.maximum(d1[s:s + CH, :, None], 1e-12)
            yield zz.reshape(-1).astype(np.float32)
    if UNIFORM:
        cb = np.arange(16, dtype=np.float32)
    else:
        zs = np.concatenate(list(z_chunks()))
        cb = lloyd_1d(zs)
    mid = ((cb[:-1] + cb[1:]) * np.float32(0.5))
    # 编码 + 反推值
    out = bytearray()
    deq = np.empty(n, np.float32)
    CH = 1 << 21
    for s in range(0, NB, CH):
        e = min(s + CH, NB)
        zz = ((B[s:e] + m1[s:e, :, None]) / np.maximum(d1[s:e, :, None], 1e-12)).reshape(-1).astype(np.float32)
        idx = np.clip(np.searchsorted(mid, zz), 0, 15).astype(np.uint8)
        nq = idx.reshape(e - s, 8, 32)
        val = (d1[s:e, :, None] * cb[idx].reshape(e - s, 8, 32) - m1[s:e, :, None]).reshape(-1)
        deq[s * 256:e * 256] = val.reshape(-1)
        # nibble 打包: 子块偶数->低 4bit, 奇数->高 4bit (与 C qg[l]&0xF / >>4 对齐)
        # C 端: qg = qs + (j/2)*32, j 偶读低 nibble, j 奇读高 nibble
        qbytes = np.zeros((e - s, 128), np.uint8)
        for j in range(8):
            gcol = (j // 2) * 32
            sub = nq[:, j, :]  # (nb, 32)
            if j % 2 == 0:
                qbytes[:, gcol:gcol + 32] |= sub
            else:
                qbytes[:, gcol:gcol + 32] |= (sub.astype(np.uint16) << 4).astype(np.uint8)
        # scales 12B 布局 (get_scale_min_k4 反推)
        scb = np.zeros((e - s, 12), np.uint8)
        scj = sc[s:e].astype(np.uint8)
        msj = ms[s:e].astype(np.uint8)
        for j in range(4):
            scb[:, j] = (scj[:, j] & 63) | ((scj[:, 4 + j] >> 4) << 6)
            scb[:, 4 + j] = (msj[:, j] & 63) | ((msj[:, 4 + j] >> 4) << 6)
            scb[:, 8 + j] = (scj[:, 4 + j] & 15) | ((msj[:, 4 + j] & 15) << 4)
        # 逐块交错 144B 记录 (C 按 blk = w + o*144 读, 不能平面写!)
        rec = np.zeros((e - s, 144), np.uint8)
        rec[:, 0:2] = d_stored[s:e].view(np.uint8).reshape(e - s, 2)
        rec[:, 2:4] = dmin_stored[s:e].view(np.uint8).reshape(e - s, 2)
        rec[:, 4:16] = scb
        rec[:, 16:144] = qbytes
        out += rec.tobytes()
    codebook = cb.astype(np.float16).tobytes()
    base = np.linalg.norm(w.astype(np.float64))
    err = np.linalg.norm((w - deq).astype(np.float64)) / base if base else 0.0
    return codebook + bytes(out), cb.astype(np.float16), deq, err

# ---- 装载 ----
def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    print('reading gguf header...')
    kvs, tensors, data_start, raw = read_gguf(src)
    g = lambda k: kvs.get(k, (None, None))[1]
    meta = [
        ('meta.nlayers',   g('llama.block_count')),
        ('meta.n_heads',   g('llama.attention.head_count')),
        ('meta.n_kv_heads', g('llama.attention.head_count_kv')),
        ('meta.head_dim',  g('llama.attention.key_length')),
        ('meta.n_ff',      g('llama.feed_forward_length')),
        ('meta.n_vocab',   g('llama.vocab_size')),
        ('meta.ctx_len',   min(g('llama.context_length') or 4096, 8192)),
        ('meta.eps',       g('llama.attention.layer_norm_rms_epsilon') or 1e-5),
        ('meta.rope_base', g('llama.rope.freq_base') or 1000000.0),
    ]
    for k, v in meta:
        assert v is not None, 'missing kv ' + k

    NORM_MARKS = ('attn_norm', 'ffn_norm', 'output_norm')
    buf = bytearray(b'AIMW')
    buf += struct.pack('<II', 1, len(tensors) + len(meta))
    total_in = 0
    total_out = 0
    print('%-40s %-8s %10s %10s' % ('tensor', 'dt', 'in_MB', 'out_MB/err'))
    for name, dims, qt, off in tensors:
        ne0 = dims[0]
        ne1 = dims[1] if len(dims) > 1 else 0
        n = ne0 * (ne1 or 1)
        rows, indim = (ne1, ne0) if ne1 else (1, ne0)
        raw.seek(data_start + off)
        data = raw.read((n + 31) // 32 * 34 if qt == 8 else n * 4)
        total_in += len(data)
        aw_dims = [rows, indim] if ne1 else [ne0]
        if qt == 8:
            vals = deq_q8(data, n)
        else:
            vals = np.frombuffer(data, np.float32, n)
        is_norm = any(mk in name for mk in NORM_MARKS)
        if is_norm or (ne1 == 0):
            payload = vals.tobytes()
            dt = 0
            err = 0.0
        else:
            payload, cb, deq, err = quant_v2(vals)
            dt = 3
        total_out += len(payload)
        nb = name.encode('utf-8')
        buf += struct.pack('<I', len(nb)) + nb
        buf += struct.pack('<I', dt) + struct.pack('<I', len(aw_dims))
        for dd in aw_dims:
            buf += struct.pack('<I', dd)
        buf += struct.pack('<I', len(payload)) + payload
        print('%-40s %-8s %10.1f %9.1f%s' % (name, dt, len(data) / 1048576.0,
              len(payload) / 1048576.0, '' if dt == 0 else ' err=%.4f' % err))

    for k, v in meta:
        nb = k.encode('utf-8')
        buf += struct.pack('<I', len(nb)) + nb
        buf += struct.pack('<I', 0) + struct.pack('<I', 1) + struct.pack('<I', 1)
        buf += struct.pack('<I', 4) + struct.pack('<f', float(v))

    with open(dst, 'wb') as f:
        f.write(bytes(buf))
    print('written %s in=%.2fGB out=%.2fGB mode=%s' % (dst, total_in / 1e9,
          total_out / 1e9, 'uniform(Q4K)' if UNIFORM else 'lloyd-v2'))

if __name__ == '__main__':
    main()
