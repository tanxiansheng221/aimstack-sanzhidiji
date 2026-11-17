# -*- coding: utf-8 -*-
# AIM-W GPU/CPU 推理后端 v0 (自研 PyTorch 前向, 吃自家 .aimw 容器)
# 架构: Llama-style (本容器无 attn_sub_norm -> SwiGLU 路径, 与 engine/forward.c 非BitNet分支逐语义对齐)
# 纪律: 解析器遇到未知 version/dtype/码3 必须拒绝, 禁止猜测 (aimw.c 同款硬性)
# 用法:
#   python tools/gpu_backend.py --aimw data/m2b_tern_all.aimw --prompt-ids 128000,791,7438,315,2324,374 --steps 16
#   python tools/gpu_backend.py --aimw ... --prompt "hello" --steps 64 --tok path/to/tokenizer.json
import argparse, json, mmap, os, struct, sys, time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DT_FP32, DT_TERN, DT_TERNP, DT_Q4KV2, DT_Q8 = 0, 1, 2, 3, 4
TERN_LUT = np.array([0.0, 1.0, -1.0, np.nan], dtype=np.float32)  # 码3=NaN(装载期拒绝)


def p(*a):
    print(*a)
    sys.stdout.flush()


class AimwFile:
    """AIM-W 容器解析 (aimw.c 逐语义对齐; mmap 驻留, 张量按需反量化)"""

    def __init__(self, path):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        buf = self.mm
        if buf[:4] != b'AIMW':
            raise ValueError('bad magic (not AIMW)')
        self.version, self.ntensors = struct.unpack_from('<II', buf, 4)
        if self.version not in (0, 1):
            raise ValueError('unknown version %d' % self.version)
        pos = 12
        self.entries = {}   # name -> (dtype, dims, off_data, nb, scale)
        for _ in range(self.ntensors):
            nl = struct.unpack_from('<I', buf, pos)[0]; pos += 4
            name = buf[pos:pos + nl].decode('utf-8'); pos += nl
            dt, nd = struct.unpack_from('<II', buf, pos); pos += 8
            dims = struct.unpack_from('<%dI' % nd, buf, pos); pos += 4 * nd
            nb = struct.unpack_from('<I', buf, pos)[0]; pos += 4
            n = 1
            for d in dims:
                n *= d
            if dt == DT_FP32:
                want, tail = n * 4, 0
            elif dt == DT_TERN:
                want, tail = n, 4
            elif dt == DT_TERNP:
                want, tail = (n + 3) // 4, 4
            elif dt == DT_Q4KV2:
                if n % 256 != 0:
                    raise ValueError('%s: Q4KV2 dims not /256' % name)
                want, tail = 32 + (n // 256) * 144, 0
            elif dt == DT_Q8:
                if n % 32 != 0:
                    raise ValueError('%s: Q8 dims not /32' % name)
                want, tail = (n // 32) * 34, 0
            else:
                raise ValueError('%s: unknown dtype %d (spec: reject)' % (name, dt))
            if nb != want + tail:
                raise ValueError('%s: payload size mismatch (%d != %d+%d)' % (name, nb, want, tail))
            scale = 0.0
            if dt in (DT_TERN, DT_TERNP):
                scale = struct.unpack_from('<f', buf, pos + want)[0]
            self.entries[name] = (dt, dims, pos, nb, scale, n)
            pos += nb

    def meta_f32(self, name):
        dt, dims, off, nb, scale, n = self.entries[name]
        assert dt == DT_FP32, 'meta tensor not fp32'
        return struct.unpack_from('<f', self.mm, off)[0]

    def dequant(self, name):
        """反量化单张量 -> fp32 numpy (分块处理控内存; 码3拒绝; Q4KV2 拒绝)"""
        dt, dims, off, nb, scale, n = self.entries[name]
        raw = self.mm[off:off + nb]
        CH = 1 << 22   # 4M 元素/块, 中间数组 ~16-32MB
        if dt == DT_FP32:
            return np.frombuffer(raw, dtype=np.float32, count=n).reshape(dims).copy()
        if dt == DT_TERN:
            # 码驻留即 -1/0/1 (int8), 非码表索引
            codes = np.frombuffer(raw, dtype=np.int8, count=n)
            out = np.empty(n, dtype=np.float32)
            for s in range(0, n, CH):
                seg = codes[s:s + CH]
                if np.any((seg != -1) & (seg != 0) & (seg != 1)):
                    raise ValueError('%s: code 3 in TERN (spec: reject)' % name)
                out[s:s + len(seg)] = seg.astype(np.float32)
            return out.reshape(dims) * scale
        if dt == DT_TERNP:
            lut = TERN_LUT
            packed = np.frombuffer(raw, dtype=np.uint8)
            sh = np.array([0, 2, 4, 6], dtype=np.uint8)
            out = np.empty(n, dtype=np.float32)
            for s in range(0, n, CH):   # CH 为 4 的倍数
                e = min(s + CH, n)
                seg = packed[s // 4:(e + 3) // 4]
                codes = ((seg[:, None] >> sh[None, :]) & 3).reshape(-1)[:e - s]
                if np.any(codes == 3):
                    raise ValueError('%s: code 3 in TERNP (spec: reject)' % name)
                out[s:e] = lut[codes.astype(np.int32)]
            return out.reshape(dims) * scale
        if dt == DT_Q8:
            nb32 = n // 32
            blocks = np.frombuffer(raw, dtype=np.uint8, count=nb32 * 34).reshape(nb32, 34)
            sc_raw = np.ascontiguousarray(blocks[:, :2]).reshape(-1)   # 每块 2B f16 scale
            sc_f16 = np.frombuffer(sc_raw, dtype='<f2').astype(np.float32)
            qs = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
            return (qs * sc_f16[:, None]).reshape(dims)
        raise ValueError('%s: dtype %d unsupported in this backend (spec: reject)' % (name, dt))

    def close(self):
        self.mm.close()
        self.f.close()


class AimwModel:
    """Llama-style 前向 (GQA + RoPE half-split + SwiGLU + RMSNorm), KV cache, greedy"""

    def __init__(self, aw, device, dtype, fp32_compute=False):
        m = {k: aw.meta_f32(k) for k in aw.entries if k.startswith('meta.')}
        self.nlayers = int(m['meta.nlayers'])
        self.n_heads = int(m['meta.n_heads'])
        self.n_kv_heads = int(m['meta.n_kv_heads'])
        self.head_dim = int(m['meta.head_dim'])
        self.n_ff = int(m['meta.n_ff'])
        self.n_vocab = int(m['meta.n_vocab'])
        self.ctx_len = int(m['meta.ctx_len'])
        self.eps = m['meta.eps']
        self.rope_base = m['meta.rope_base']
        self.n_embd = aw.entries['token_embd.weight'][1][-1]
        self.device, self.dtype = device, dtype
        self.cdt = torch.float32 if fp32_compute else dtype   # 计算路径 dtype
        self.group = self.n_heads // self.n_kv_heads
        # BitNet 探测: attn_sub_norm 在场即 BitNet (SubLN x2 + relu2), 同 forward.c
        self.bitnet = 'blk.0.attn_sub_norm.weight' in aw.entries
        # lm_head: output.weight 缺席 -> tied (token_embd 同权重), 同 forward.c 回落
        self.lm_name = 'output.weight' if 'output.weight' in aw.entries else 'token_embd.weight'

        p('LOAD dequant -> %s %s (this is the slow part, cached in-process)'
          % (device, dtype))
        t0 = time.time()
        self.W = {}
        big = ['output.weight', 'token_embd.weight', 'output_norm.weight']
        attn = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
                'attn_norm.weight'] + (['attn_sub_norm.weight'] if self.bitnet else [])
        ffn = ['ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight', 'ffn_norm.weight'] \
              + (['ffn_sub_norm.weight'] if self.bitnet else [])
        for i in range(self.nlayers):
            big += ['blk.%d.%s' % (i, s) for s in attn + ffn]
        big = [b for b in big if b in aw.entries]
        for name in big:
            arr = aw.dequant(name)
            self.W[name] = torch.from_numpy(arr).to(device=device, dtype=dtype)
            del arr   # fp32 中间量即转即释, 只驻留目标 dtype
        self.tok_embd = self.W['token_embd.weight']
        p('LOAD done %.1fs  bitnet=%d lm=%s' % (time.time() - t0, self.bitnet, self.lm_name))

    def _lin(self, x, name):
        W = self.W[name]
        if x.dtype != W.dtype:
            W = W.to(x.dtype)   # bf16 驻留 + fp32 计算路径: 上转无损, 等同 fp32 权重
        return x @ W.T

    def _rmsnorm(self, x, wname):
        w = self.W[wname].to(x.dtype)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w.float()).to(x.dtype)

    def _rope(self, x, pos0):
        # half-split 配对 (i 与 i+half), theta_j = base^(-2j/head_dim)  [forward.c 同款]
        # 注意: q/k 进来时已是 (B, NH, T, HD) (transpose 过), T 在 dim 2
        B, H, T, HD = x.shape
        half = HD // 2
        j = torch.arange(half, device=x.device, dtype=torch.float32)
        inv = self.rope_base ** (-(2.0 * j) / HD)
        t = (pos0 + torch.arange(T, device=x.device, dtype=torch.float32))
        ang = t[:, None] * inv[None, :]
        cos = torch.cos(ang)[None, None, :, :]   # (1,1,T,half): 广播到 (B,NH,T,HD)
        sin = torch.sin(ang)[None, None, :, :]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    @torch.no_grad()
    def forward(self, ids, cache=None):
        """ids: list[int]; cache: (k,v) 每层 [B,H,Len,HD] 或 None; 返回 (logits[-1], new_cache)"""
        B, T = 1, len(ids)
        pos0 = 0 if cache is None else cache[0][0].shape[2]
        x = self.tok_embd[torch.tensor(ids, device=self.device)]  # [T, D]
        x = x[None].to(self.cdt)
        HD, NH, NKV, G = self.head_dim, self.n_heads, self.n_kv_heads, self.group

        nk, nv = [], []
        dbg = os.environ.get('AIMDBG')
        dbg2 = os.environ.get('AIMDBG2')
        if dbg:
            p('L%d x nan=%d absmax=%.3g' % (0, int(torch.isnan(x).sum()), float(x.float().abs().max())))
        if dbg2:
            P = T - 1
            p('EMB last x0=%.6g x1=%.6g absmax=%.6g' % (x[0, P, 0], x[0, P, 1], float(x[0, P].abs().max())))
        for i in range(self.nlayers):
            h = self._rmsnorm(x[0], 'blk.%d.attn_norm.weight' % i)[None]
            q = self._lin(h, 'blk.%d.attn_q.weight' % i).view(B, T, NH, HD).transpose(1, 2)
            k = self._lin(h, 'blk.%d.attn_k.weight' % i).view(B, T, NKV, HD).transpose(1, 2)
            v = self._lin(h, 'blk.%d.attn_v.weight' % i).view(B, T, NKV, HD).transpose(1, 2)
            q = self._rope(q, pos0).to(self.cdt)
            k = self._rope(k, pos0).to(self.cdt)
            if dbg2 and i == 0:
                P = T - 1
                p('QKV last q0=%.6g q1=%.6g k0=%.6g k1=%.6g v0=%.6g v1=%.6g'
                  % (q[0, 0, P, 0], q[0, 0, P, 1], k[0, 0, P, 0], k[0, 0, P, 1],
                     v[0, 0, P, 0], v[0, 0, P, 1]))
            if cache is None:
                kc, vc = k, v
            else:
                kc = torch.cat([cache[0][i], k], dim=2)
                vc = torch.cat([cache[1][i], v], dim=2)
            nk.append(kc)
            nv.append(vc)
            kx = kc.repeat_interleave(G, dim=1)
            vx = vc.repeat_interleave(G, dim=1)
            att = (q @ kx.transpose(-2, -1)) / (HD ** 0.5)
            L = kx.shape[2]
            mask = torch.triu(
                torch.full((T, L), float('-inf'), device=x.device),
                diagonal=pos0 + 1)[None, None]
            att = (att.float() + mask).softmax(-1).to(vx.dtype)   # softmax 在 fp32, 回落模型 dtype
            o = (att @ vx).transpose(1, 2).reshape(B, T, NH * HD)
            if dbg2 and i == 0:
                P = T - 1
                p('AO last o0=%.6g o1=%.6g absmax=%.6g  w_pos=%s'
                  % (o[0, P, 0], o[0, P, 1], float(o[0, P].abs().max()),
                     [round(float(att[0, 0, P, t]), 6) for t in range(T)]))
            if self.bitnet:
                # SubLN: attention 输出先过 sub_norm 再进 o_proj (forward.c 同款)
                o = self._rmsnorm(o[0], 'blk.%d.attn_sub_norm.weight' % i)[None]
            x = x + self._lin(o, 'blk.%d.attn_output.weight' % i)
            h = self._rmsnorm(x[0], 'blk.%d.ffn_norm.weight' % i)[None]
            if self.bitnet:
                # relu2(gate) * up -> ffn_sub_norm -> down (SubLN 在激活后)
                g = self._lin(h, 'blk.%d.ffn_gate.weight' % i)
                u = self._lin(h, 'blk.%d.ffn_up.weight' % i)
                act = torch.clamp(g, min=0).pow(2) * u
                act = self._rmsnorm(act[0], 'blk.%d.ffn_sub_norm.weight' % i)[None]
                x = x + self._lin(act, 'blk.%d.ffn_down.weight' % i)
            else:
                g = F.silu(self._lin(h, 'blk.%d.ffn_gate.weight' % i))
                x = x + self._lin(g * self._lin(h, 'blk.%d.ffn_up.weight' % i),
                                  'blk.%d.ffn_down.weight' % i)
            if dbg:
                p('L%d x nan=%d absmax=%.3g' % (i + 1, int(torch.isnan(x).sum()),
                                                float(x.float().abs().max())))
        h = self._rmsnorm(x[0], 'output_norm.weight')
        logits = self._lin(h, self.lm_name)
        return logits[-1].float(), (nk, nv)

    @torch.no_grad()
    def generate(self, ids, steps, eos=None):
        cache = None
        cur = list(ids)
        out = []
        t0 = None
        for n in range(steps):
            logits, cache = self.forward(cur if cache is None else cur[-1:], cache)
            nid = int(torch.argmax(logits))
            if t0 is None:
                t0 = time.time()
            if eos is not None and nid == eos:
                break
            out.append(nid)
            cur.append(nid)
        dt = time.time() - t0 if t0 else 0.0
        return out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aimw', required=True)
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--prompt-ids', default=None, help='逗号分隔, 与 --prompt 二选一')
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--device', default='auto', help='auto/cuda/mps/cpu')
    ap.add_argument('--dtype', default='auto', help='auto(=cuda:bf16/cpu:fp32)/bf16/fp16/fp32')
    ap.add_argument('--tok', default=os.path.join(ROOT, 'data', 'bitnet2b', 'tokenizer.json'))
    ap.add_argument('--show-text', action='store_true')
    ap.add_argument('--probe', action='store_true', help='只跑 prefill, 打印 top-10 logits')
    ap.add_argument('--compute', default='auto', choices=['auto', 'fp32', 'bf16'],
                    help='计算精度: auto(cpu=fp32/gpu=bf16)')
    args = ap.parse_args()

    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtyp = args.dtype
    if dtyp == 'auto':
        dtyp = 'bf16' if dev != 'cpu' else 'fp32'
    dtmap = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}

    if args.prompt_ids:
        ids = [int(s) for s in args.prompt_ids.split(',')]
    else:
        sys.path.insert(0, HERE)
        from llama3_tok import Tok
        tok = Tok(args.tok)
        ids = tok.encode(args.prompt)

    p('DEVICE %s  DTYPE %s' % (dev, dtyp))
    aw = AimwFile(args.aimw)
    # CPU 默认 fp32 计算路径 (bf16 驻留 + 上转无损) — 与 C 引擎 fp32 贪心逐位对齐
    # GPU 默认 bf16 计算路径 (快; 精度略降, --compute fp32 可切)
    fp32c = (dev == 'cpu')
    if args.compute == 'fp32':
        fp32c = True
    elif args.compute == 'bf16':
        fp32c = False
    model = AimwModel(aw, dev, dtmap[dtyp], fp32_compute=fp32c)
    p('META layers=%d heads=%d kv=%d hd=%d vocab=%d ctx=%d eps=%g rope=%g'
      % (model.nlayers, model.n_heads, model.n_kv_heads, model.head_dim,
         model.n_vocab, model.ctx_len, model.eps, model.rope_base))
    p('PROMPT_IDS', ids)
    if args.probe:
        lg, _ = model.forward(ids, None)
        top = torch.topk(lg, 10)
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            p('  %d\t%.4f' % (i, v))
        p('nan=%d inf=%d' % (int(torch.isnan(lg).sum()), int(torch.isinf(lg).sum())))
        p('PROBE_DONE')
        return

    out, dt = model.generate(ids, args.steps)
    p('GREEDY_%d: %s' % (len(out), out))
    p('GEN %.2fs  %.2f tok/s' % (dt, len(out) / dt if dt else 0.0))
    if args.show_text and os.path.exists(args.tok):
        from llama3_tok import Tok
        tok = Tok(args.tok)
        p('GREEDY_TEXT:', repr(tok.decode(out)))
    p('DONE')


if __name__ == '__main__':
    main()
