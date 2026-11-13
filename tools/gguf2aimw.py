# -*- coding: utf-8 -*-
# aimp: GGUF -> AIM-W 三值转换器 (PTQ, stdlib+numpy)
# 用途: 验证"开源模型后处理三值化"的输出质量与三值通道真机速度
# 诚实预期: 纯三值 PTQ 大概率乱码 (业界一致: 三值需 QAT); 实验即判决
# 用法: python gguf2aimw.py <in.gguf> <out.aimw> [mode]
#   mode=all     全张量三值 (最狠档)
#   mode=mixed   token_embd/output 保 fp32, 其余三值 (混合档)
import struct, sys, os
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def read_gguf(path):
    raw = open(path, 'rb')
    def rd(fmt):
        sz = struct.calcsize(fmt)
        return struct.unpack(fmt, raw.read(sz))
    magic, ver = rd('<II')
    assert magic == 0x46554747 and ver == 3, 'not gguf v3'
    n_tensors, n_kv = rd('<QQ')
    def rd_str():
        (ln,) = rd('<Q')
        return raw.read(ln).decode('utf-8', 'replace')
    def rd_val(vt):
        if vt == 8: return rd_str()
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,9:0,10:8,11:8,12:8}[vt]
        if vt == 9:  # array
            (evt, cnt) = struct.unpack('<IQ', raw.read(12))
            if evt == 8:
                for _ in range(cnt): rd_str()
            else:
                esz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[evt]
                raw.seek(esz * cnt, 1)
            return None
        if vt in (6, 12):
            return struct.unpack('<f' if vt == 6 else '<d', raw.read(sz))[0]
        b = raw.read(sz)
        return int.from_bytes(b, 'little') if sz > 1 else b[0]
    kvs = {}
    for _ in range(n_kv):
        k = rd_str()
        (vt,) = struct.unpack('<I', raw.read(4))
        kvs[k] = (vt, rd_val(vt))
    tensors = []
    for _ in range(n_tensors):
        name = rd_str()
        (nd,) = struct.unpack('<I', raw.read(4))
        dims = struct.unpack('<%dQ' % nd, raw.read(8 * nd))
        (qt,) = struct.unpack('<I', raw.read(4))
        (off,) = struct.unpack('<Q', raw.read(8))
        tensors.append((name, dims, qt, off))
    align = kvs.get('general.alignment', (4, 32))[1]
    data_start = (raw.tell() + align - 1) // align * align
    return kvs, tensors, data_start, raw

def f16(v):
    return struct.unpack('<e', struct.pack('<H', v))[0]

def deq_q8(buf, n_elems):
    # 分片向量化: Q8_0 块 34B = 2B f16 scale(LE) + 32B int8
    nb = n_elems // 32
    out = np.empty(n_elems, np.float32)
    CH = 65536
    for s in range(0, nb, CH):
        e = min(s + CH, nb)
        arr = np.frombuffer(buf, np.uint8, (e - s) * 34, s * 34).reshape(e - s, 34)
        h16 = arr[:, 0].astype(np.uint16) | (arr[:, 1].astype(np.uint16) << np.uint16(8))
        scale = h16.view(np.float16).astype(np.float32)   # 位重解释(astype 是数值转换, 曾致 scale 放大 1e7 倍)
        q = arr[:, 2:].copy().view(np.int8).astype(np.float32)
        out[s * 32:e * 32] = (scale[:, None] * q).astype(np.float32).ravel()
    return out

def tern_packed(w):
    scale = np.float32(np.abs(w).max())
    codes = np.clip(np.round(w.astype(np.float64) / max(float(scale), 1e-12)),
                    -1, 1).astype(np.int8)
    n = len(codes)
    m = np.where(codes > 0, 1, np.where(codes < 0, 2, 0)).astype(np.uint8)
    pad = (-n) % 4
    if pad:
        m = np.concatenate([m, np.zeros(pad, np.uint8)])
    mp = m.reshape(-1, 4)
    packed = (mp[:, 0] | (mp[:, 1] << 2) | (mp[:, 2] << 4) | (mp[:, 3] << 6))
    return packed.astype(np.uint8).tobytes() + np.float32(scale).tobytes()

def main():
    src, dst = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'all'
    print('reading gguf header...')
    kvs, tensors, data_start, raw = read_gguf(src)
    g = lambda k: kvs.get(k, (None, None))[1]
    meta = [
        ('meta.nlayers',  g('llama.block_count')),
        ('meta.n_heads',  g('llama.attention.head_count')),
        ('meta.n_kv_heads', g('llama.attention.head_count_kv')),
        ('meta.head_dim', g('llama.attention.key_length')),
        ('meta.n_ff',     g('llama.feed_forward_length')),
        ('meta.n_vocab',  g('llama.vocab_size')),
        ('meta.ctx_len',  min(g('llama.context_length') or 4096, 8192)),
        ('meta.eps',      g('llama.attention.layer_norm_rms_epsilon') or 1e-5),
        ('meta.rope_base', g('llama.rope.freq_base') or 1000000.0),
    ]
    for k, v in meta:
        assert v is not None, 'missing kv ' + k
    KEEP_FP32 = {'token_embd.weight', 'output.weight'} if mode == 'mixed' else set()

    buf = bytearray(b'AIMW')
    buf += struct.pack('<II', 1, len(tensors) + len(meta))
    total_in = 0
    for name, dims, qt, off in tensors:
        ne0, ne1 = dims[0], dims[1] if len(dims) > 1 else 0
        n = ne0 * (ne1 or 1)
        rows, indim = (ne1, ne0) if ne1 else (1, ne0)
        # C 侧 matvec 按 [rows][in] 行主读; GGUF ne[0]=in(fast), ne[1]=rows
        raw.seek(data_start + off)
        data = raw.read((n + 31) // 32 * 34 if qt == 8 else n * 4)
        total_in += len(data)
        if qt == 8:
            vals = deq_q8(data, n)
        else:
            vals = np.frombuffer(data, np.float32, n)
        aw_dims = [rows, indim] if ne1 else [ne0]
        if name in KEEP_FP32:
            payload = vals.tobytes()
            dt = 0
        else:
            payload = tern_packed(vals)
            dt = 2
        nb = name.encode('utf-8')
        buf += struct.pack('<I', len(nb)) + nb
        buf += struct.pack('<I', dt) + struct.pack('<I', len(aw_dims))
        for d in aw_dims:
            buf += struct.pack('<I', d)
        buf += struct.pack('<I', len(payload)) + payload
        print('  %-40s qt=%d %s -> %s' % (name, qt, aw_dims,
              'fp32' if dt == 0 else 'tern(%dKB)' % (len(payload) // 1024)))
    for k, v in meta:
        nb = k.encode('utf-8')
        buf += struct.pack('<I', len(nb)) + nb
        buf += struct.pack('<I', 0) + struct.pack('<I', 1) + struct.pack('<I', 1)
        buf += struct.pack('<I', 4) + struct.pack('<f', float(v))
    with open(dst, 'wb') as f:
        f.write(bytes(buf))
    print('written %s in=%.2fGB mode=%s' % (dst, total_in / 1e9, mode))

if __name__ == '__main__':
    main()
