# -*- coding: utf-8 -*-
"""aim_manual.py — S1 自研反传引擎（81.15 训练提速第二刀）
手推 AimLM 全部算子梯度，废掉 torch.autograd 计算图：
  前向: embedding+pos → PAD 掩码 → L×[分块线性注意力(S 链) → LN → FFN] → LN → Wout → 答案区 CE
  反向: CE → Wout/lnf → 倒序各块（LN²/FFN/注意力 chunk 递推 dS 链）→ E/Pp scatter
torch 仅作为 BLAS/张量容器（hipBLAS/cuBLAS 本体），零 autograd 依赖。
参数与 AdamWBF16/clip_grad_norm_ 接口兼容（.grad 字段）。
对账: python aim_manual.py 跑 selftest，与 torch.autograd 逐参数 allclose。
"""
import math
import torch

BOS, PAD, EOS, MASK = 257, 256, 258, 259


def _ln_fwd(x, g, b, eps=1e-5):
    """LayerNorm 前向，返回 (y, xhat, rstd)（反传复用）。半精度上转 fp32，fp64 保真。"""
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    mu = xf.mean(-1, keepdim=True)
    xc = xf - mu
    var = xc.pow(2).mean(-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    xhat = xc * rstd
    y = (xhat * g.float() + b.float()).to(x.dtype)
    return y, xhat.to(x.dtype), rstd.to(x.dtype)


def _ln_bwd(dy, xhat, rstd, g):
    """LayerNorm 反向：返回 dx 与 (dg, db)。"""
    dyf = dy.float() if dy.dtype in (torch.float16, torch.bfloat16) else dy
    T = xhat.shape[-1]
    xhf = xhat.float() if xhat.dtype in (torch.float16, torch.bfloat16) else xhat
    gf = g.float() if g.dtype in (torch.float16, torch.bfloat16) else g
    dg = (dyf * xhf).sum(dim=tuple(range(dy.dim() - 1)))
    db = dyf.sum(dim=tuple(range(dy.dim() - 1)))
    dxc = dyf * gf
    dx = (rstd / T) * (T * dxc - dxc.sum(-1, keepdim=True)
                       - xhf * (dxc * xhf).sum(-1, keepdim=True))
    return dx.to(dy.dtype), dg.to(dy.dtype), db.to(dy.dtype)


class ManualAimLM:
    """与 AimLMReal 同构的手写反传模型。p 为 torch 张量（无 autograd），.grad 手工填充。"""

    def __init__(self, V=277, D=96, L=2, H1=256, chunk=64, T=512, seed=0):
        g = torch.Generator().manual_seed(seed)
        def rn(*shape, s=1.0):
            return torch.randn(*shape, generator=g) * s
        self.V, self.D, self.T, self.L, self.chunk = V, D, T, L, chunk
        self.p = {
            "E": rn(V, D, s=0.1), "Pp": rn(T, D, s=0.02),
            "lnf_g": torch.ones(D), "lnf_b": torch.zeros(D),
            "Wout": rn(D, V) / math.sqrt(D),
        }
        self.blocks = []
        for bi in range(L):
            bp = {
                "Wq": rn(D, D) / math.sqrt(D), "Wk": rn(D, D) / math.sqrt(D),
                "Wv": rn(D, D) / math.sqrt(D), "Wo": rn(D, D) / math.sqrt(D),
                "W1": rn(D, H1) / math.sqrt(D), "b1": torch.zeros(H1),
                "W2": rn(H1, D) / math.sqrt(H1), "b2": torch.zeros(D),
                "ln1_g": torch.ones(D), "ln1_b": torch.zeros(D),
                "ln2_g": torch.ones(D), "ln2_b": torch.zeros(D),
            }
            self.blocks.append(bp)
            self.p.update({f"{k}{bi}": v for k, v in bp.items()})

    def parameters(self):
        return list(self.p.values())

    def zero_grad(self, set_to_none=True):
        for t in self.p.values():
            if set_to_none:
                t.grad = None
            elif t.grad is not None:
                t.grad.zero_()

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.p.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            self.p[k].copy_(v)

    def cfg(self):
        return dict(V=self.V, D=self.D, L=self.L, H1=self.blocks[0]["W1"].shape[1],
                    chunk=self.chunk, T=self.T)

    # ---------- 前向 ----------
    def forward(self, tok, pos, amask):
        B, T = tok.shape
        D = self.D
        dt = self.p["E"].dtype
        mask = (tok != PAD).to(dt).unsqueeze(-1)
        h = (self.p["E"][tok] + self.p["Pp"][pos]).to(dt) * mask
        cache = {"tok": tok, "pos": pos, "mask": mask, "blocks": []}
        ch = self.chunk
        for bi, bp in enumerate(self.blocks):
            bc = {"h_in": h}
            q = h @ bp["Wq"]
            k = (h @ bp["Wk"]) / math.sqrt(D)
            v = h @ bp["Wv"]
            o = torch.empty_like(q)
            Sst = torch.zeros(B, D, D, dtype=h.dtype, device=h.device)
            S_list = []
            n_chunk = (T + ch - 1) // ch
            for c in range(n_chunk):
                S_list.append(Sst.clone())
                s, e = c * ch, min((c + 1) * ch, T)
                qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]
                M_c = torch.tril(torch.ones(e - s, e - s, dtype=h.dtype, device=h.device))
                A = (qc @ kc.transpose(1, 2)) * M_c
                o[:, s:e] = A @ vc + qc @ Sst
                Sst = Sst + kc.transpose(1, 2) @ vc
            bc.update({"q": q, "k": k, "v": v, "S_list": S_list})
            z = h + o @ bp["Wo"]
            h1, xh1, rs1 = _ln_fwd(z, bp["ln1_g"], bp["ln1_b"])
            z2 = h1 @ bp["W1"] + bp["b1"]
            a1 = torch.relu(z2)
            h2 = a1 @ bp["W2"] + bp["b2"]
            z3 = h1 + h2  # FFN 残差底是 ln1 之后的 h1（与 AimLMReal 一致）
            h3, xh3, rs3 = _ln_fwd(z3, bp["ln2_g"], bp["ln2_b"])
            bc.update({"o": o, "z_ln1": z, "h1": h1, "xh1": xh1, "rs1": rs1,
                       "a1": a1, "z_ln2": z3, "xh3": xh3, "rs3": rs3})
            cache["blocks"].append(bc)
            h = h3
        hf, xhf, rsf = _ln_fwd(h, self.p["lnf_g"], self.p["lnf_b"])
        cache.update({"xhf": xhf, "rsf": rsf})
        logits = hf @ self.p["Wout"]
        return logits, cache

    def loss_and_grad(self, tok, pos, amask, grad_scale=1.0):
        """前向 + 答案区 CE + 反向。返回 (loss, logits)。梯度落 self.p[*].grad。
        grad_scale：梯度缩放（ACCUM 梯度累积时传 1/ACCUM，与 autograd 路径 loss/ACCUM 对齐）。"""
        logits, cache = self.forward(tok, pos, amask)
        lg = logits[:, :-1, :]
        tg = tok[:, 1:]
        m = amask[:, 1:]
        N = m.sum()
        lgf = lg.float() if lg.dtype in (torch.float16, torch.bfloat16) else lg
        lse = torch.logsumexp(lgf, dim=-1, keepdim=True)
        pick = torch.gather(lgf, -1, tg.unsqueeze(-1))
        ce = ((lse.squeeze(-1) - pick.squeeze(-1))[m]).sum() / N
        # dlogits：logits[i, j] 预测 tok[i, j+1]，m[i, j] 选答案区目标
        dlogits = torch.zeros_like(logits)
        pmat = (lgf - lse).exp()
        idx_b, idx_j = m.nonzero(as_tuple=True)
        d = pmat[idx_b, idx_j]
        d[torch.arange(len(idx_b), device=d.device), tg[idx_b, idx_j]] -= 1.0
        dlogits[idx_b, idx_j] = d / N * grad_scale
        self.backward(cache, dlogits)
        return ce, logits

    # ---------- 反向 ----------
    @torch.no_grad()
    def backward(self, cache, dlogits):
        p = self.p
        B, T, _ = dlogits.shape
        D = self.D
        V = self.V

        def acc(name, g):
            pg = p[name].grad
            p[name].grad = g if pg is None else pg + g

        # lnf + Wout
        hf_hat, rsf = cache["xhf"], cache["rsf"]
        _fl = lambda t: t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
        hf = (_fl(hf_hat) * _fl(p["lnf_g"]) + _fl(p["lnf_b"])).to(dlogits.dtype)
        acc("Wout", hf.reshape(-1, D).T @ dlogits.reshape(-1, V))
        dhf = dlogits @ p["Wout"].T
        dh, dgf, dbf = _ln_bwd(dhf, hf_hat, rsf, p["lnf_g"])
        acc("lnf_g", dgf); acc("lnf_b", dbf)

        for bi in range(self.L - 1, -1, -1):
            bp = self.blocks[bi]
            bc = cache["blocks"][bi]
            # ln2 反传
            dz3, dg2, db2 = _ln_bwd(dh, bc["xh3"], bc["rs3"], bp["ln2_g"])
            acc(f"ln2_g{bi}", dg2); acc(f"ln2_b{bi}", db2)
            # FFN: z3 = h1 + (relu(h1@W1+b1)@W2+b2)——残差底是 h1
            dh_ffn = dz3
            acc(f"W2{bi}", bc["a1"].reshape(-1, bc["a1"].shape[-1]).T @ dh_ffn.reshape(-1, D))
            acc(f"b2{bi}", dh_ffn.sum(dim=(0, 1)))
            da1 = dh_ffn @ bp["W2"].T
            dz2 = da1 * (bc["a1"] > 0).to(da1.dtype)
            acc(f"W1{bi}", bc["h1"].reshape(-1, D).T @ dz2.reshape(-1, dz2.shape[-1]))
            acc(f"b1{bi}", dz2.sum(dim=(0, 1)))
            dh1_ffn = dz2 @ bp["W1"].T
            dh1_total = dz3 + dh1_ffn  # 残差 + FFN 路合流
            dz, dg1, dbg1 = _ln_bwd(dh1_total, bc["xh1"], bc["rs1"], bp["ln1_g"])
            acc(f"ln1_g{bi}", dg1); acc(f"ln1_b{bi}", dbg1)
            do = dz @ bp["Wo"].T
            acc(f"Wo{bi}", bc["o"].reshape(-1, D).T @ dz.reshape(-1, D))
            # 分块线性注意力反传（S 链倒序；S_list[c] = 第 c 块的输入态 S_{c-1}）
            q, k, v = bc["q"], bc["k"], bc["v"]
            S_list = bc["S_list"]
            ch = self.chunk
            dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv = torch.zeros_like(v)
            Dcur = torch.zeros(B, D, D, dtype=q.dtype, device=q.device)
            n_chunk = (T + ch - 1) // ch
            for c in range(n_chunk - 1, -1, -1):
                s, e = c * ch, min((c + 1) * ch, T)
                qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]
                S_prev = S_list[c]
                M_c = torch.tril(torch.ones(e - s, e - s, dtype=q.dtype, device=q.device))
                do_c = do[:, s:e]
                A_c = (qc @ kc.transpose(1, 2)) * M_c
                dA = do_c @ vc.transpose(1, 2)
                dv[:, s:e] = A_c.transpose(1, 2) @ do_c + kc @ Dcur
                dq[:, s:e] = (dA * M_c) @ kc + do_c @ S_prev.transpose(1, 2)
                dk[:, s:e] = (dA * M_c).transpose(1, 2) @ qc + vc @ Dcur.transpose(1, 2)
                Dcur = Dcur + qc.transpose(1, 2) @ do_c
            dh_q = dq @ bp["Wq"].T
            dh_k = (dk @ bp["Wk"].T) / math.sqrt(D)
            dh_v = dv @ bp["Wv"].T
            acc(f"Wq{bi}", bc["h_in"].reshape(-1, D).T @ dq.reshape(-1, D))
            acc(f"Wk{bi}", bc["h_in"].reshape(-1, D).T @ (dk / math.sqrt(D)).reshape(-1, D))
            acc(f"Wv{bi}", bc["h_in"].reshape(-1, D).T @ dv.reshape(-1, D))
            # 块间梯度流无 mask（mask 只在 embedding 处乘一次，与 AimLMReal 计算图一致）
            dh = dz + dh_q + dh_k + dh_v
        # embedding 反传（首块输入即 embedding 输出；mask 在这里乘——∂h0/∂(E+Pp)=mask）
        tok, pos = cache["tok"], cache["pos"]
        dhm = dh * cache["mask"]
        dE = torch.zeros_like(p["E"])
        dE.index_add_(0, tok.reshape(-1), dhm.reshape(-1, D))
        acc("E", dE)
        dPp = torch.zeros_like(p["Pp"])
        dPp.index_add_(0, pos.reshape(-1), dhm.reshape(-1, D))
        acc("Pp", dPp)


# ---------------- 对账 selftest ----------------
def selftest(V=277, D=64, L=2, H1=128, chunk=32, T=96, B=3, seed=42):
    torch.manual_seed(0)
    man = ManualAimLM(V=V, D=D, L=L, H1=H1, chunk=chunk, T=T, seed=seed)
    import aim_train_real as R
    ref = R.AimLMReal(V=V, D=D, L=L, H1=H1, chunk=chunk, T=T, seed=seed)
    # nn.Module 嵌套 key → 平铺 key 映射
    sd = ref.state_dict()
    flat = {"E": "E", "Pp": "Pp", "lnf.weight": "lnf_g", "lnf.bias": "lnf_b", "Wout": "Wout"}
    for bi in range(L):
        for k in ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2"]:
            flat[f"blocks.{bi}.{k}"] = f"{k}{bi}"
        flat[f"blocks.{bi}.ln1.weight"] = f"ln1_g{bi}"
        flat[f"blocks.{bi}.ln1.bias"] = f"ln1_b{bi}"
        flat[f"blocks.{bi}.ln2.weight"] = f"ln2_g{bi}"
        flat[f"blocks.{bi}.ln2.bias"] = f"ln2_b{bi}"
    man.load_state_dict({flat[src]: v for src, v in sd.items()})
    # 随机 batch
    g = torch.Generator().manual_seed(7)
    tok = torch.randint(1, V - 4, (B, T), generator=g)
    pos = torch.arange(T).unsqueeze(0).repeat(B, 1)
    amask = torch.zeros(B, T, dtype=torch.bool)
    amask[:, T // 2:] = True
    # autograd 参考（fp32）
    logits_ref = ref(tok)
    import torch.nn as nn
    lg = logits_ref[:, :-1, :]; tg = tok[:, 1:]; m = amask[:, 1:]
    loss_ref = nn.functional.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none")[m.reshape(-1)].mean()
    loss_ref.backward()
    # 手写反传
    loss_man, logits_man = man.loss_and_grad(tok, pos, amask)
    print("loss_ref %.6f  loss_man %.6f  diff %.2e" % (loss_ref.item(), loss_man.item(),
                                                       abs(loss_ref.item() - loss_man.item())))
    print("logits maxdiff %.2e" % (logits_ref - logits_man).abs().max().item())
    ok = True
    ref_grads = {}
    for src, dst in flat.items():
        mod, _, leaf = src.rpartition(".")
        obj = ref
        for part in mod.split("."):
            if part:
                obj = getattr(obj, part)
        ref_grads[dst] = getattr(obj, leaf).grad
    for k, gv in ref_grads.items():
        if gv is None:
            continue
        gm = man.p[k].grad
        if gm is None:
            print("MISSING grad:", k); ok = False; continue
        diff = (gv - gm).abs().max().item()
        rel = diff / max(gv.abs().max().item(), 1e-8)
        stat = "OK " if (diff < 2e-3 or rel < 1e-4) else "FAIL"
        if diff >= 2e-3 and rel >= 1e-4:
            ok = False
        print("%s %-8s maxdiff %.2e rel %.2e" % (stat, k, diff, rel))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    selftest()
