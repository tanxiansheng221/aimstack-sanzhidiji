# -*- coding: utf-8 -*-
# aimp M1: 合成最小 GGUF -> 驱动 C 探针 -> 断言对账 (Python 仅作对账工具, 不进产物)
import struct, subprocess, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'build/test.gguf'
EXE = sys.argv[2] if len(sys.argv) > 2 else 'build/aimp_probe.exe'
ALIGN = 64

def s(v):  # gguf string
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b

def kv_str(k, v):
    return s(k) + struct.pack('<I', 8) + s(v)

def kv_u32(k, v):
    return s(k) + struct.pack('<I', 4) + struct.pack('<I', v)

def kv_f32(k, v):
    return s(k) + struct.pack('<I', 6) + struct.pack('<f', v)

def kv_bool(k, v):
    return s(k) + struct.pack('<I', 7) + struct.pack('<B', 1 if v else 0)

def kv_arr_str(k, vs):
    b = s(k) + struct.pack('<I', 9) + struct.pack('<I', 8) + struct.pack('<Q', len(vs))
    for v in vs:
        b += s(v)
    return b

def kv_arr_f32(k, vs):
    return (s(k) + struct.pack('<I', 9) + struct.pack('<I', 6) +
            struct.pack('<Q', len(vs)) + struct.pack('<%df' % len(vs), *vs))

def tensor_info(name, dims, qt, offset):
    b = s(name) + struct.pack('<I', len(dims))
    for d in dims:
        b += struct.pack('<Q', d)
    return b + struct.pack('<I', qt) + struct.pack('<Q', offset)

# ---- 构造 ----
meta = b''
meta += kv_str('general.architecture', 'testarch')
meta += kv_str('general.name', 'aimp-synth')
meta += kv_u32('general.alignment', ALIGN)
meta += kv_u32('testarch.block_count', 42)
meta += kv_f32('testarch.attention.layer_norm_rms_epsilon', 1e-5)
meta += kv_bool('testarch.bool_flag', True)
meta += kv_arr_str('tokenizer.ggml.tokens', ['<pad>', '<s>', '</s>', 'hello', 'ni_hao'])
meta += kv_arr_f32('test.scores', [0.5, -1.25, 3.75])

tensors = b''
# t0: F32 [2,3] = 24B, offset 0
# t1: Q8_0 [64] = 34*2=68B, offset 64 (对齐)
tensors += tensor_info('t0.weight', [2, 3], 0, 0)
tensors += tensor_info('t1.weight', [64], 8, ALIGN)

header = (struct.pack('<I', 0x46554747) + struct.pack('<I', 3) +
          struct.pack('<Q', 2) + struct.pack('<Q', 8) + meta + tensors)

data_start = (len(header) + ALIGN - 1) // ALIGN * ALIGN
t0_data = struct.pack('<6f', 1, 2, 3, 4, 5, 6)
t1_data = b''
for blk in range(2):  # Q8_0: f32 scale + 32 x int8
    t1_data += struct.pack('<f', 0.5 * (blk + 1)) + bytes([i & 0x7f for i in range(32)])

with open(OUT, 'wb') as f:
    f.write(header)
    f.write(b'\x00' * (data_start - len(header)))
    f.write(t0_data)
    f.seek(data_start + ALIGN)
    f.write(t1_data)

print('synth: %s header=%d data_start=%d' % (OUT, len(header), data_start))

# ---- 驱动 C 探针断言 ----
r = subprocess.run([EXE, OUT], capture_output=True, text=True,
                   encoding='utf-8', errors='replace')
out = r.stdout
ok = True
def chk(name, cond):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        ok = False

chk('exit0', r.returncode == 0)
chk('version3', 'version=3' in out)
chk('align64', 'align=64' in out)
chk('kv8', 'kv=8' in out)
chk('tensors2', 'tensors=2' in out)
chk('arch', '[arch] testarch' in out)
chk('block_count', '= 42' in out)
chk('epsilon', 'layer_norm_rms_epsilon' in out and 'e-06' in out)
chk('tokens5', 'tokens=5' in out)
chk('q8_0', 'Q8_0' in out)
chk('data_start', ('data_start=%d' % data_start) in out)

print('==== ALL_%s ====' % ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
