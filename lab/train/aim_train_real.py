# -*- coding: utf-8 -*-
"""aim_train_real.py — AimLM 真训管线 v1（81.13 加速专项落地产物）
- 数据: v1_mix.jsonl → 字节预分词缓存（npz: 拼接 uint8 + 索引），启动秒级、喂速无忧
- 模型: AimLM（V=277 字节族）+ GEMM 分块线性注意力（aim_lm_fast.linattn_gemm 同族，等价性已对账）
- loss: 回答区（a+EOS）交叉熵；val_exact=答案区 argmax 全对
- 训练: AdamW(0.9,0.95)+wd0.1 + warmup+cosine + AMP fp16(GradScaler) + grad clip
- 断点: {tag}.pt 每 save_every 步（model+opt+step+config），--resume 续训
配置走 AIM_* 环境变量或 argv，DCU/本地一码两跑。
"""
import io, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BOS, PAD, EOS, MASK = 257, 256, 258, 259  # tokenizer.md specials
SEP = 10  # '\n' 分隔问答


def env(k, d):
    return os.environ.get(k) or d


# ---------------- data ----------------
def build_cache(mix_path, cache_path):
    """混料 JSONL → 预分词缓存。sample i: q bytes + b'\n' + a bytes；记录 (start, qlen, alen)。"""
    starts, qlens, alens = [], [], []
    buf = bytearray()
    with io.open(mix_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["q"].encode("utf-8")[:1600]
            a = r["a"].encode("utf-8")[:1600]
            if not q or not a:
                continue
            starts.append(len(buf))
            qlens.append(len(q))
            alens.append(len(a))
            buf += q + b"\n" + a
    starts = np.array(starts, dtype=np.int64)
    qlens = np.array(qlens, dtype=np.int32)
    alens = np.array(alens, dtype=np.int32)
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    np.savez(cache_path, data=arr, starts=starts, qlens=qlens, alens=alens)
    print("cache: %d samples, %.1fMB -> %s" % (len(starts), len(arr) / 1e6, cache_path), flush=True)


def load_cache(cache_path):
    z = np.load(cache_path)
    return z["data"], z["starts"], z["qlens"], z["alens"]


# ---------------- packed data（81.15 训练提速：定长窗口打包，灭 PAD 空转）----------------
def build_packed_cache(mix_path, cache_path, T=512, seed=42, val_frac=0.01):
    """混料 JSONL → 定长 T 窗口缓存。
    - 样本 = [BOS]+q+[SEP]+a+[EOS]（与 unpacked 完全同构），超 T 样本剔除（与旧 tr_idx 口径一致）
    - 1% 样本 holdout 当 val（样本级独立数组，与窗口零重叠）
    - 训练样本按长度排序后贪心装箱（窗口内样本边界对齐，Q→A 完整性保留；尾部 PAD ≤ 均长/2）
    - 窗口含 win_tok( uint16 )/win_am( a+EOS 区 bool )/win_pos( 样本内位置重置 int16 )"""
    items = []
    with io.open(mix_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["q"].encode("utf-8")[:1600]
            a = r["a"].encode("utf-8")[:1600]
            if not q or not a:
                continue
            seq = np.concatenate(([BOS], np.frombuffer(q, dtype=np.uint8), [SEP],
                                  np.frombuffer(a, dtype=np.uint8), [EOS])).astype(np.int16)
            if len(seq) <= T:
                items.append(seq)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(items))
    n_val = max(200, int(len(items) * val_frac))
    val_items = [items[i] for i in perm[:n_val]]
    train_items = [items[i] for i in perm[n_val:]]

    # val 样本级数组（与 load_cache 同构，val_exact 直接复用）
    vs, vq, va, vbuf = [], [], [], bytearray()
    for seq in val_items:
        arr = seq.astype(np.uint8)
        ql = int(np.argmax(arr[1:] == SEP))      # = len(q)（与 load_cache 口径一致）
        al = len(arr) - ql - 3                    # len = BOS+q+SEP+a+EOS
        vs.append(len(vbuf)); vq.append(ql); va.append(al)
        vbuf += arr[1:1 + ql].tobytes() + b"\n" + arr[ql + 2:ql + 2 + al].tobytes()
    vdata = np.frombuffer(bytes(vbuf), dtype=np.uint8)

    # 长度排序贪心装箱
    order = sorted(range(len(train_items)), key=lambda i: len(train_items[i]))
    win_tok, win_am, win_pos = [], [], []
    cur_t, cur_a, cur_p = [], [], []
    for i in order:
        seq = train_items[i]
        if len(cur_t) + len(seq) > T:
            pad = T - len(cur_t)
            # 尾部回填：装不下时用下一题的问题头部当上下文（amask=False 不进 loss，该题完整进下一窗）
            echo = seq[1:1 + pad]  # 跳过 BOS
            n_echo = len(echo)
            win_tok.append(np.array(cur_t + list(echo), dtype=np.uint16))
            win_am.append(np.array(cur_a + [False] * n_echo, dtype=bool))
            win_pos.append(np.array(cur_p + list(range(n_echo)), dtype=np.int16))
            cur_t, cur_a, cur_p = [], [], []
        cur_t += list(seq)
        cur_a += [False] * ((ql_i := int(np.argmax(seq[1:] == SEP)) + 1) + 1) + [True] * (len(seq) - ql_i - 1)
        cur_p += list(range(len(seq)))
    if cur_t:
        pad = T - len(cur_t)
        win_tok.append(np.array(cur_t + [PAD] * pad, dtype=np.uint16))
        win_am.append(np.array(cur_a + [False] * pad, dtype=bool))
        win_pos.append(np.array(cur_p + [0] * pad, dtype=np.int16))
    win_tok = np.array(win_tok); win_am = np.array(win_am); win_pos = np.array(win_pos)
    perm_w = rng.permutation(len(win_tok))  # 单一 perm 同步洗牌三数组
    win_tok, win_am, win_pos = win_tok[perm_w], win_am[perm_w], win_pos[perm_w]
    util = float((win_tok != PAD).mean())
    np.savez(cache_path, win_tok=win_tok, win_am=win_am, win_pos=win_pos,
             vdata=vdata, vstarts=np.array(vs, dtype=np.int64),
             vqlens=np.array(vq, dtype=np.int32), valens=np.array(va, dtype=np.int32))
    print("packed cache: %d windows T=%d util=%.1f%% | val %d samples -> %s"
          % (len(win_tok), T, util * 100, len(vs), cache_path), flush=True)


def load_packed_cache(cache_path):
    z = np.load(cache_path)
    return (z["win_tok"], z["win_am"], z["win_pos"],
            z["vdata"], z["vstarts"], z["vqlens"], z["valens"])



def make_batch(data, starts, qlens, alens, idx, device, with_val_mask=True):
    """一个 batch：BOS+q+\n+a+EOS，右 PAD；返回 tok(B,T) 与 answer mask(B,T)（a+EOS 区）。"""
    B = len(idx)
    lens = np.array([qlens[i] + alens[i] + 3 for i in idx])  # BOS q \n a EOS
    T = int(lens.max())
    tok = np.full((B, T), PAD, dtype=np.int64)
    amask = np.zeros((B, T), dtype=bool)
    for b, i in enumerate(idx):
        s, ql, al = starts[i], qlens[i], alens[i]
        seq = np.concatenate(([BOS], data[s:s + ql], [SEP], data[s + ql + 1:s + ql + 1 + al], [EOS]))
        tok[b, :len(seq)] = seq
        amask[b, ql + 2:ql + 2 + al + 1] = True  # a + EOS
    return (torch.from_numpy(tok).to(device),
            torch.from_numpy(amask).to(device))


# ---------------- model ----------------
class LinAttnBlockGEMM(nn.Module):
    """GEMM 分块线性注意力块（与 aim_lm_fast.linattn_gemm 同族，含块级 checkpoint）。"""

    def __init__(self, D, H1, chunk=64):
        super().__init__()
        self.D, self.chunk = D, chunk
        self.Wq = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.Wk = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.Wv = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.Wo = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.W1 = nn.Parameter(torch.randn(D, H1) / math.sqrt(D))
        self.b1 = nn.Parameter(torch.zeros(H1))
        self.W2 = nn.Parameter(torch.randn(H1, D) / math.sqrt(H1))
        self.b2 = nn.Parameter(torch.zeros(D))
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.register_buffer("_M", torch.tril(torch.ones(chunk, chunk)), persistent=False)

    def forward(self, h):
        from torch.utils.checkpoint import checkpoint
        B, T, D = h.shape
        q = h @ self.Wq
        k = (h @ self.Wk) / math.sqrt(D)
        v = h @ self.Wv
        o = torch.empty_like(q)
        S = torch.zeros(B, D, D, device=h.device, dtype=h.dtype)
        M = self._M
        ch = self.chunk
        for c in range((T + ch - 1) // ch):
            s, e = c * ch, min((c + 1) * ch, T)
            M_c = M[:e - s, :e - s]
            if h.requires_grad:
                # 块内整体 checkpoint（含 S 更新）：反向重算，S 链不保留（省 B×D×D×块数 显存）
                o_new, S = checkpoint(self._chunk_all, q[:, s:e], k[:, s:e], v[:, s:e], S, M_c, use_reentrant=False)
            else:
                o_new, S = self._chunk_all(q[:, s:e], k[:, s:e], v[:, s:e], S, M_c)
            o[:, s:e] = o_new
        h = self.ln1(h + o @ self.Wo)
        h = self.ln2(h + torch.relu(h @ self.W1 + self.b1) @ self.W2 + self.b2)
        return h

    @staticmethod
    def _chunk_all(q_c, k_c, v_c, S_in, M_c):
        A = (q_c @ k_c.transpose(1, 2)) * M_c
        o = (A @ v_c) + q_c @ S_in
        S_new = S_in + k_c.transpose(1, 2) @ v_c
        return o, S_new


class AimLMReal(nn.Module):
    def __init__(self, V=277, D=96, L=2, H1=256, chunk=64, T=768, seed=0,
                 abacus=False, ab_pos=32):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V, self.D, self.T = V, D, T
        self.E = nn.Parameter(torch.randn(V, D, generator=g) * 0.1)
        self.Pp = nn.Parameter(torch.randn(T, D, generator=g) * 0.02)
        # Abacus 位嵌入（McLeish et al., arXiv:2405.17399）：数字字节按其在数字串内的
        # 位权序（0=个位，向左递增）查可学习表 AB，加进 token embedding。
        self.AB = nn.Parameter(torch.randn(ab_pos, D, generator=g) * 0.02) if abacus else None
        self.blocks = nn.ModuleList([LinAttnBlockGEMM(D, H1, chunk) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.Wout = nn.Parameter(torch.randn(D, V, generator=g) / math.sqrt(D))
        self.cfg = dict(V=V, D=D, L=L, H1=H1, chunk=chunk, T=T,
                        abacus=bool(abacus), ab_pos=int(ab_pos))

    def forward(self, tok, pos=None):
        h = self.E[tok]
        h = h + (self.Pp[None, : tok.shape[1], :] if pos is None else self.Pp[pos])
        if self.AB is not None:
            d = (tok >= 48) & (tok <= 57)  # '0'-'9'
            idx = torch.arange(tok.shape[1], device=tok.device)
            t = torch.where(~d, idx, torch.full_like(idx, tok.shape[1]))
            nxt = torch.flip(torch.cummin(torch.flip(t, [-1]), -1)[0], [-1])  # 右侧最近非数字下标
            dpos = (nxt - idx - 1).clamp(min=0, max=self.AB.shape[0] - 1)  # 0=个位（离数字串右缘距离）
            h = h + self.AB[dpos].to(h.dtype) * d.unsqueeze(-1).to(h.dtype)
        h = h * (tok.unsqueeze(-1) != PAD)  # PAD 位清零（embedding 不泄位置信息）
        for blk in self.blocks:
            h = blk(h)
        return self.lnf(h) @ self.Wout


def answer_loss_t(logits, tok, amask):
    """回答区交叉熵：logits[i] 预测 tok[i+1]；amask[t]=tok[t] 属于回答区（a+EOS）。"""
    lg = logits[:, :-1, :]
    tg = tok[:, 1:]
    m = amask[:, 1:]
    return nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none")[m.reshape(-1)].mean()


@torch.no_grad()
def val_exact(model, data, starts, qlens, alens, vidx, device, ncap=1000):
    if len(vidx) > ncap:
        vidx = list(np.random.RandomState(7).choice(vidx, ncap, replace=False))
    model.eval()
    hit = 0
    tot = 0
    for s in range(0, len(vidx), 32):
        idx = vidx[s:s + 32]
        tok, amask = make_batch(data, starts, qlens, alens, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(tok)
        pred = logits.argmax(-1)
        # 答案区每 token：pred[:, t-1] vs tok[:, t]
        m = amask[:, 1:]
        ok = (pred[:, :-1] == tok[:, 1:]) | ~m
        hit += (ok.all(dim=1)).sum().item()
        tot += len(idx)
    model.train()
    return hit / max(tot, 1)


class AdamWBF16:
    """自研 bf16 状态 AdamW（81.14 零预算加速第一刀）：
    m/v 存 bf16（optimizer 显存省一半，1.5B 省 5.9GB → 批次 B2→B4），
    master 权重保持 fp32，动量更新全程 fp32 计算、落盘前 cast bf16。
    零第三方依赖；接口兼容 torch.optim（param_groups/step/zero_grad/state_dict/load_state_dict）。"""
    def __init__(self, params, lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8):
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.eps = eps
        self.t = 0
        self.param_groups = [{"lr": lr, "params": list(params)}]
        self.m = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.param_groups[0]["params"]]
        self.v = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.param_groups[0]["params"]]

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        lr = self.param_groups[0]["lr"]
        for i, p in enumerate(self.param_groups[0]["params"]):
            g = p.grad
            if g is None:
                continue
            m = self.m[i].float()
            v = self.v[i].float()
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            self.m[i].copy_(m)
            self.v[i].copy_(v)
            if self.wd:
                p.add_(p, alpha=-lr * self.wd)
            upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            p.add_(upd, alpha=-lr)

    @torch.no_grad()
    def zero_grad(self, set_to_none=True):
        for p in self.param_groups[0]["params"]:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        for i, (m, v) in enumerate(zip(sd["m"], sd["v"])):
            self.m[i].copy_(m)
            self.v[i].copy_(v)


def main():
    mix = env("AIM_MIX", os.path.join(r"H:\AimCode\lab\cpu-native", "v1_mix.jsonl"))
    tag = env("AIM_TAG", "v1_smoke")
    D = int(env("AIM_D", "96")); L = int(env("AIM_L", "2")); H1 = int(env("AIM_H1", "256"))
    T = int(env("AIM_T", "768")); B = int(env("AIM_B", "32")); STEPS = int(env("AIM_STEPS", "3000"))
    LR = float(env("AIM_LR", "1e-3")); WARM = int(env("AIM_WARM", "200"))
    CHUNK = int(env("AIM_CHUNK", "64")); EVAL_EVERY = int(env("AIM_EVAL", "500"))
    SAVE_EVERY = int(env("AIM_SAVE", "2000")); SEED = int(env("AIM_SEED", "42"))
    ACCUM = int(env("AIM_ACCUM", "1"))
    AMP = env("AIM_AMP", "fp16")  # bf16 | fp16 | off（DCU 用 bf16：范围=fp32，免疫 S 长程累积溢出）
    RESUME = env("AIM_RESUME", "")
    # --- DDP 支持（81.14：第二容器 4 卡终训；AIM_DDP=1 + torchrun 启动）---
    DDP_ON = env("AIM_DDP", "0") == "1"
    rank, world = 0, 1
    if DDP_ON:
        import torch.distributed as dist
        dist.init_process_group(backend=env("AIM_BACKEND", "nccl"))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")  # 本机 gloo 冒烟用
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = env("AIM_OUT", os.path.join(r"H:\AimCode\lab\cpu-native", "train_out"))
    os.makedirs(out_dir, exist_ok=True)
    cache = env("AIM_CACHE", "") or os.path.join(out_dir, tag + "_cache.npz")

    if not os.path.exists(cache):
        if env("AIM_PACK", "0") == "1":
            build_packed_cache(mix, cache, T, SEED)
        else:
            build_cache(mix, cache)
    _z = np.load(cache)
    PACKED = "win_tok" in _z.files
    if PACKED:
        win_tok, win_am, win_pos, vdata, vstarts, vqlens, valens = load_packed_cache(cache)
        _z.close()
        vidx = list(range(len(vstarts)))  # 显式 holdout val（与窗口零重叠）
        vA = (vdata, vstarts, vqlens, valens)
        n_win = len(win_tok)
        win_ids = np.arange(n_win)[rank::world]  # DDP 各 rank 交错切窗
        tr_pool = win_ids
        # GPU 常驻窗口（零 H2D 拷贝）
        win_tok_g = torch.from_numpy(win_tok.astype(np.int64)).to(device)
        win_am_g = torch.from_numpy(win_am).to(device)
        win_pos_g = torch.from_numpy(win_pos.astype(np.int64)).to(device)
        print("packed: %d windows T=%d util=%.1f%% val=%d device=%s rank=%d/%d"
              % (n_win, win_tok.shape[1], float((win_tok != PAD).mean()) * 100,
                 len(vidx), device, rank, world), flush=True)
    else:
        data, starts, qlens, alens = _z["data"], _z["starts"], _z["qlens"], _z["alens"]
        has_val = "vdata" in _z.files
        if has_val:
            vA = (_z["vdata"], _z["vstarts"], _z["vqlens"], _z["valens"])
        _z.close()
        tr_idx = np.array([i for i in range(len(starts)) if qlens[i] + alens[i] + 3 <= T], dtype=np.int64)
        if has_val:
            vidx = list(range(len(vA[1])))
            tr_pool = tr_idx
            print("val: explicit holdout %d rows" % len(vidx), flush=True)
        else:
            rng = np.random.RandomState(SEED)
            perm = rng.permutation(len(tr_idx))
            n_val = max(200, int(len(tr_idx) * 0.01))
            vidx = list(tr_idx[perm[:n_val]])
            tr_pool = tr_idx[perm[n_val:]]
            vA = (data, starts, qlens, alens)
        tr_pool = tr_pool[rank::world]  # DDP：各 rank 交错切分，单卡 world=1 不变
    rng_s = np.random.RandomState(SEED + 7 + rank)  # 训练采样：per-rank 种子
    print("samples: train=%d val=%d device=%s rank=%d/%d" % (
        len(tr_pool), len(vidx), device, rank, world), flush=True)

    ABACUS = env("AIM_ABACUS", "0") == "1"
    AB_POS = int(env("AIM_AB_POS", "32"))
    model_raw = AimLMReal(D=D, L=L, H1=H1, chunk=CHUNK, T=T, seed=SEED,
                          abacus=ABACUS, ab_pos=AB_POS).to(device)
    if DDP_ON:
        model = torch.nn.parallel.DistributedDataParallel(model_raw, gradient_as_bucket_view=True)
    else:
        model = model_raw
    nparam = sum(p.numel() for p in model_raw.parameters())
    print("params: %.2fM (D=%d L=%d H1=%d chunk=%d)" % (nparam / 1e6, D, L, H1, CHUNK), flush=True)

    # --- S1 手写反传（81.15：AIM_AUTOGRAD=0；man.p 与 model_raw 参数共享同一张量对象，
    #     手写梯度直落 model_raw 参数 .grad，optimizer/clip/checkpoint 零改动）---
    MANUAL = env("AIM_AUTOGRAD", "1") == "0"
    man = None
    if MANUAL:
        import aim_manual
        man = aim_manual.ManualAimLM(V=model_raw.V, D=D, L=L, H1=H1, chunk=CHUNK, T=T, seed=SEED)
        shared = {"E": model_raw.E, "Pp": model_raw.Pp,
                  "lnf_g": model_raw.lnf.weight, "lnf_b": model_raw.lnf.bias,
                  "Wout": model_raw.Wout}
        for bi in range(L):
            blk = model_raw.blocks[bi]
            for k in ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2"]:
                shared[f"{k}{bi}"] = getattr(blk, k)
            shared[f"ln1_g{bi}"] = blk.ln1.weight
            shared[f"ln1_b{bi}"] = blk.ln1.bias
            shared[f"ln2_g{bi}"] = blk.ln2.weight
            shared[f"ln2_b{bi}"] = blk.ln2.bias
        man.p = shared
        man.blocks = [{k: shared[f"{k}{bi}"] for k in bp} for bi, bp in enumerate(man.blocks)]
        print("backward: MANUAL (S1 手写反传, AIM_AUTOGRAD=0)", flush=True)
    if ABACUS and MANUAL:
        raise SystemExit("[FATAL] AIM_ABACUS 与手写反传(AIM_AUTOGRAD=0)不兼容——AB 无手写梯度")
    if ABACUS:
        print("abacus: ON (digit-position embeddings, ab_pos=%d, +%.2fM params)"
              % (AB_POS, AB_POS * D / 1e6), flush=True)

    WD = float(env("AIM_WD", "0.1"))   # 81.16: weight decay 可调（grokking 加速器，Power et al. 2022）
    if env("AIM_OPT", "adamw") == "bf16adam":
        opt = AdamWBF16(model.parameters(), lr=LR, weight_decay=WD)
        print("opt: AdamWBF16 (m/v bf16, master fp32) wd=%s" % WD, flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)
        print("opt: AdamW wd=%s" % WD, flush=True)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[AMP]
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    amp_enabled = amp_dtype is not None and device.type == "cuda"
    if MANUAL:
        # 手写反传路径按参数精度（fp32）直算，autocast/GradScaler 不介入
        use_scaler = False
        amp_enabled = False
        print("amp: off (manual backward fp32)", flush=True)

    def autocast_ctx():
        return torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled)

    start_step = 0
    if RESUME and os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location=device)
        if ABACUS:
            missing, unexpected = model_raw.load_state_dict(ck["model"], strict=False)
            print("resume(abacus): missing=%s unexpected=%s (AB 随机初始化)"
                  % (missing, unexpected), flush=True)
        else:
            model_raw.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as ex:
            # 新增参数（如 Abacus AB）使 optimizer 参数组变长 → torch 严格校验拒载。
            # 处置：全新 optimizer 状态（m/v=0），Adam 几百步内重建动量，微调无伤。
            print("opt group mismatch (%s): FRESH optimizer state (m/v=0)" % str(ex)[:80], flush=True)
        start_step = ck["step"]
        print("resumed from %s @ step %d" % (RESUME, start_step), flush=True)

    def lr_at(step):
        es = step - start_step  # 续训语义：warmup/cosine 相对本段起点
        if es < WARM:
            return LR * (es + 1) / WARM
        p = (es - WARM) / max(STEPS - WARM, 1)
        return LR * (0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    t0 = time.time()
    model.train()
    tokens_done = 0
    opt.zero_grad(set_to_none=True)
    for step in range(start_step + 1, start_step + STEPS + 1):
        es = step - start_step  # 本段相对步数（EVAL/SAVE 节奏用）
        lr = lr_at(step)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        for _micro in range(ACCUM):
            idx = rng_s.choice(tr_pool, B, replace=len(tr_pool) < B)
            if PACKED:
                it = torch.from_numpy(idx.astype(np.int64)).to(device)
                tok = win_tok_g[it].long()
                amask = win_am_g[it]
                pos = win_pos_g[it]
                if MANUAL:
                    loss, logits = man.loss_and_grad(tok, pos, amask, grad_scale=1.0 / ACCUM)
                    loss = loss / ACCUM
                else:
                    with autocast_ctx():
                        logits = model(tok, pos)
                        loss = answer_loss_t(logits, tok, amask) / ACCUM
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                tokens_done += int((win_tok_g[it] != PAD).sum())
            else:
                tok, amask = make_batch(data, starts, qlens, alens, idx, device)
                if MANUAL:
                    pos = torch.arange(tok.shape[1], device=device).unsqueeze(0).expand(tok.shape[0], -1)
                    loss, logits = man.loss_and_grad(tok, pos, amask, grad_scale=1.0 / ACCUM)
                    loss = loss / ACCUM
                else:
                    with autocast_ctx():
                        logits = model(tok)
                        loss = answer_loss_t(logits, tok, amask) / ACCUM
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                tokens_done += int((qlens[idx] + alens[idx]).sum())
        if DDP_ON and MANUAL:
            # 手写反传无 DDP 钩子：梯度手动 all-reduce（均值）
            import torch.distributed as dist
            for p_ in model.parameters():
                if p_.grad is not None:
                    dist.all_reduce(p_.grad)
                    p_.grad /= world
        if use_scaler:
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_scaler:
            # inf 梯度经 clip 会变 nan（0×inf）——非有限范数时跳过 clip，交 GradScaler skip 该步
            if not torch.isfinite(gnorm):
                for p_ in model.parameters():
                    if p_.grad is not None and not torch.isfinite(p_.grad).all():
                        p_.grad = None
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        loss = loss * ACCUM  # 日志口径还原
        # NaN 熔断（DDP 下 all-reduce 同步判定，防各 rank 分裂 hang）
        nan_flag = 0.0 if math.isfinite(loss.item()) else 1.0
        if DDP_ON:
            import torch.distributed as dist
            tflag = torch.tensor(nan_flag, device=device)
            dist.all_reduce(tflag)
            nan_flag = tflag.item()
        if nan_flag > 0:
            nan_streak = getattr(main, "_nan", 0) + 1
            main._nan = nan_streak
            if nan_streak >= 3:
                print("ABORT: loss NaN x3（溢出嫌疑），熔断省卡时", flush=True)
                break
        else:
            main._nan = 0
        if rank == 0 and step % 50 == 0:
            print("step %d loss %.4f lr %.2e tok/s %.0f (%.0fs)"
                  % (step, loss.item(), lr, tokens_done / max(time.time() - t0, 1), time.time() - t0), flush=True)
        if rank == 0 and (es % EVAL_EVERY == 0 or es == STEPS):
            ve = val_exact(model_raw, vA[0], vA[1], vA[2], vA[3], vidx, device)
            print("step %d val_exact %.4f" % (step, ve), flush=True)
        if rank == 0 and (es % SAVE_EVERY == 0 or es == STEPS):
            ckpt = os.path.join(out_dir, tag + ".pt")
            tmp = ckpt + ".tmp"  # 原子写：先 tmp 后 rename，保存中断不损坏旧断点
            torch.save({"model": model_raw.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": model_raw.cfg, "T": T}, tmp)
            os.replace(tmp, ckpt)
            print("saved %s @ step %d" % (ckpt, step), flush=True)
    print("=== TRAIN DONE %s ===" % tag, flush=True)


if __name__ == "__main__":
    main()
