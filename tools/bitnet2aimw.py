# -*- coding: utf-8 -*-
# aimp M2.9: BitNet b1.58-2B-4T (microsoft, MIT) -> AIM-W 转换器
# 三值 BitLinear: packed U8 (out/4, in), 每 byte 4 码 x 2bit, 码=tern+1, 低 2bit 在前,
#   byte 第 i 字段 = 第 i*R+b 行 (R=packed 行数); weight_scale(bf16 标量) = absmean 的 s=1/mean|w|,
#   前向时除回 -> 我方 TERNP per-tensor scale = 1/weight_scale
# embedding/lm_head(tied, bf16) -> AIM-W Q8(dtype 4, 34B/32w); 归一化层 -> F32
# 用法: python bitnet2aimw.py data/bitnet2b/model.safetensors data/bitnet2b/m2b_tern.aimw
import struct, sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ARCH = dict(nlayers=30, n_heads=20, n_kv_heads=5, head_dim=128,
            n_ff=6912, n_vocab=128256, ctx_len=4096, eps=1e-5, rope_base=500000.0)

def read_st(path):
    """safetensors 纯 stdlib 解析: yield (name, dtype, shape, np.array view)"""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        blob = np.fromfile(path, np.uint8, count=os.path.getsize(path))[base:]
        for name, info in hdr.items():
            if name == '__metadata__':
                continue
            dt, shape = info['dtype'], info['shape']
            o0, o1 = info['data_offsets']
            raw = blob[o0:o1]
            yield name, dt, shape, raw

def bf16_to_f32(raw):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)

def unpack_tern(packed):
    """(R, in) U8 -> (4R, in) int8 三值码 {-1,0,1}; byte 第 i 字段 = 第 i*R+b 行"""
    R, IN = packed.shape
    codes = np.empty((4 * R, IN), np.uint8)
    for i in range(4):
        codes[i * R:(i + 1) * R] = (packed >> (2 * i)) & 3
    return codes.astype(np.int8) - 1

def ternp_pack(codes, scale):
    """AIM-W TERNP: 码 0/1/2 = 0/+1/-1, 4 码/byte 低位在前 + 尾随 f32 scale"""
    n = len(codes)
    m = np.where(codes > 0, 1, np.where(codes < 0, 2, 0)).astype(np.uint8)
    pad = (-n) % 4
    if pad:
        m = np.concatenate([m, np.zeros(pad, np.uint8)])
    mp = m.reshape(-1, 4)
    packed = (mp[:, 0] | (mp[:, 1] << 2) | (mp[:, 2] << 4) | (mp[:, 3] << 6))
    return packed.astype(np.uint8).tobytes() + np.float32(scale).tobytes()

def q8_pack(w):
    """AIM-W Q8(dtype 4): 32 权重一块, scale=f16(amax/127), q=int8; 与 C deq_q8_0 逐位一致"""
    n = len(w)
    assert n % 32 == 0
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
    return out.tobytes()

RENAME_MAP_TMPL = [
    ('model.embed_tokens.weight', 'token_embd.weight'),
    ('model.norm.weight', 'output_norm.weight'),
    ('.self_attn.q_proj', '.attn_q'),
    ('.self_attn.k_proj', '.attn_k'),
    ('.self_attn.v_proj', '.attn_v'),
    ('.self_attn.o_proj', '.attn_output'),
    ('.self_attn.attn_sub_norm', '.attn_sub_norm'),
    ('.mlp.gate_proj', '.ffn_gate'),
    ('.mlp.up_proj', '.ffn_up'),
    ('.mlp.down_proj', '.ffn_down'),
    ('.mlp.ffn_sub_norm', '.ffn_sub_norm'),
    ('.input_layernorm', '.attn_norm'),
    ('.post_attention_layernorm', '.ffn_norm'),
]
TERN_KEYS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

def convert(src, dst):
    # 张量清单: 按加载器期望的顺序
    out = bytearray(b'AIMW')
    meta = [('meta.nlayers', ARCH['nlayers']), ('meta.n_heads', ARCH['n_heads']),
            ('meta.n_kv_heads', ARCH['n_kv_heads']), ('meta.head_dim', ARCH['head_dim']),
            ('meta.n_ff', ARCH['n_ff']), ('meta.n_vocab', ARCH['n_vocab']),
            ('meta.ctx_len', ARCH['ctx_len']), ('meta.eps', ARCH['eps']),
            ('meta.rope_base', ARCH['rope_base'])]
    entries = []  # (name, dtype, dims, payload_fn)
    L = ARCH['nlayers']
    entries.append(('token_embd.weight', 'Q8', [ARCH['n_vocab'], 2560]))
    for l in range(L):
        entries.append(('blk.%d.attn_norm.weight' % l, 'F32', [2560]))
        entries.append(('blk.%d.attn_q.weight' % l, 'TERN', [2560, 2560]))
        entries.append(('blk.%d.attn_k.weight' % l, 'TERN', [640, 2560]))
        entries.append(('blk.%d.attn_v.weight' % l, 'TERN', [640, 2560]))
        entries.append(('blk.%d.attn_sub_norm.weight' % l, 'F32', [2560]))
        entries.append(('blk.%d.attn_output.weight' % l, 'TERN', [2560, 2560]))
        entries.append(('blk.%d.ffn_norm.weight' % l, 'F32', [2560]))
        entries.append(('blk.%d.ffn_gate.weight' % l, 'TERN', [6912, 2560]))
        entries.append(('blk.%d.ffn_up.weight' % l, 'TERN', [6912, 2560]))
        entries.append(('blk.%d.ffn_sub_norm.weight' % l, 'F32', [6912]))
        entries.append(('blk.%d.ffn_down.weight' % l, 'TERN', [2560, 6912]))
    entries.append(('output_norm.weight', 'F32', [2560]))
    # output.weight 缺席: tied embeddings, 引擎回落 token_embd

    def to_src(aim_name):
        if aim_name == 'output_norm.weight':
            return 'model.norm.weight'
        parts = aim_name.split('.')
        # blk.N.<rest>.weight -> model.layers.N.<rest>.weight
        l = int(parts[1])
        rest = '.'.join(parts[2:-1])
        remap = {'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm',
                 'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj',
                 'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj',
                 'attn_sub_norm': 'self_attn.attn_sub_norm',
                 'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj',
                 'ffn_down': 'mlp.down_proj', 'ffn_sub_norm': 'mlp.ffn_sub_norm'}
        return 'model.layers.%d.%s.weight' % (l, remap[rest])

    print('loading source tensors into memory...')
    src_all = {}
    for name, dt, shape, raw in read_st(src):
        src_all[name] = (dt, shape, raw)
    print('tensors loaded: %d' % len(src_all))

    out += struct.pack('<II', 1, len(entries) + len(meta))
    total_out = 0
    for aim_name, kind, dims in entries:
        if aim_name == 'token_embd.weight':
            dt, shape, raw = src_all['model.embed_tokens.weight']
            assert dt == 'BF16', dt
            w = bf16_to_f32(raw)
            payload = q8_pack(w)
            dt_out = 4
        elif kind == 'F32':
            dt, shape, raw = src_all[to_src(aim_name)]
            assert dt == 'BF16', (aim_name, dt)
            payload = bf16_to_f32(raw).tobytes()
            dt_out = 0
        else:
            dt, shape, raw = src_all[to_src(aim_name)]
            assert dt == 'U8', (aim_name, dt)
            packed = np.frombuffer(raw, np.uint8).reshape(shape)
            codes = unpack_tern(packed)
            ws_raw = src_all[to_src(aim_name).replace('.weight', '.weight_scale')][2]
            ws = float(bf16_to_f32(ws_raw)[0])
            scale = np.float32(ws)   # HF AutoBitLinear(offline): y=(A8(x)·W_t)*ws -> W_real=codes*ws (实证 top1 264)
            payload = ternp_pack(codes, scale)
            dt_out = 2
        total_out += len(payload)
        nb = aim_name.encode('utf-8')
        out += struct.pack('<I', len(nb)) + nb
        out += struct.pack('<I', dt_out) + struct.pack('<I', len(dims))
        for dd in dims:
            out += struct.pack('<I', dd)
        out += struct.pack('<I', len(payload)) + payload
        print('  %-32s dt=%d %7.1fKB' % (aim_name, dt_out, len(payload) / 1024.0))

    for k, v in meta:
        kb = k.encode('utf-8')
        out += struct.pack('<I', len(kb)) + kb
        out += struct.pack('<I', 0) + struct.pack('<I', 1) + struct.pack('<I', 1)
        out += struct.pack('<I', 4) + struct.pack('<f', float(v))

    with open(dst, 'wb') as f:
        f.write(bytes(out))
    print('written %s (%.2f GB)' % (dst, len(out) / 1e9))

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
