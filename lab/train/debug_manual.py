# -*- coding: utf-8 -*-
# debug_manual.py — 逐层对照 ManualAimLM vs AimLMReal 前向
import torch, math
import aim_manual as M
import aim_train_real as R

V, D, L, H1, chunk, T, B, seed = 277, 64, 2, 128, 32, 96, 3, 42
man = M.ManualAimLM(V=V, D=D, L=L, H1=H1, chunk=chunk, T=T, seed=seed)
ref = R.AimLMReal(V=V, D=D, L=L, H1=H1, chunk=chunk, T=T, seed=seed)

flat = {"E": "E", "Pp": "Pp", "lnf.weight": "lnf_g", "lnf.bias": "lnf_b", "Wout": "Wout"}
for bi in range(L):
    for k in ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2"]:
        flat[f"blocks.{bi}.{k}"] = f"{k}{bi}"
    flat[f"blocks.{bi}.ln1.weight"] = f"ln1_g{bi}"
    flat[f"blocks.{bi}.ln1.bias"] = f"ln1_b{bi}"
    flat[f"blocks.{bi}.ln2.weight"] = f"ln2_g{bi}"
    flat[f"blocks.{bi}.ln2.bias"] = f"ln2_b{bi}"
sd = ref.state_dict()
man.load_state_dict({flat[s]: v for s, v in sd.items()})

g = torch.Generator().manual_seed(7)
tok = torch.randint(1, V - 4, (B, T), generator=g)
pos = torch.arange(T).unsqueeze(0).repeat(B, 1)

# 手写前向逐层
with torch.no_grad():
    logits_man, cache = man.forward(tok, pos, torch.zeros(B, T, dtype=torch.bool))
    # ref 前向拆解
    p = man.p
    mask = (tok != R.PAD).float().unsqueeze(-1)
    h = (ref.E[tok] + ref.Pp[None]) * mask
    print("h0 diff %.2e" % (h - cache["blocks"][0]["h_in"]).abs().max().item())
    bp = ref.blocks[0]
    q = h @ bp.Wq; k = (h @ bp.Wk) / math.sqrt(D); v = h @ bp.Wv
    print("q diff %.2e" % (q - cache["blocks"][0]["q"]).abs().max().item())
    # 手算 chunk 注意力（无 checkpoint）
    o = torch.empty_like(q)
    Sst = torch.zeros(B, D, D)
    for c in range((T + chunk - 1) // chunk):
        s, e = c * chunk, min((c + 1) * chunk, T)
        qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]
        Mc = torch.tril(torch.ones(e - s, e - s))
        A = (qc @ kc.transpose(1, 2)) * Mc
        o[:, s:e] = A @ vc + qc @ Sst
        Sst = Sst + kc.transpose(1, 2) @ vc
    print("o diff %.2e" % (o - cache["blocks"][0]["o"]).abs().max().item())
    z = h + o @ bp.Wo
    h1 = torch.nn.functional.layer_norm(z, (D,), bp.ln1.weight, bp.ln1.bias)
    print("ln1 out diff %.2e" % (
        h1 - torch.nn.functional.layer_norm(
            cache["blocks"][0]["z_ln1"], (D,),
            man.p["ln1_g0"], man.p["ln1_b0"])).abs().max().item())
    logits_ref = ref(tok)
    print("logits diff %.2e" % (logits_ref - logits_man).abs().max().item())
    # ref 单块前向对照（手动过 ref 的 block0）
    hb = h
    for blk in ref.blocks:
        hb = blk(hb)
    print("ref blocks manual-replay diff (vs cache h_last): %.2e" % (hb - torch.nn.functional.layer_norm(
        cache["blocks"][-1]["z_ln2"], (D,), man.p["lnf_g"], man.p["lnf_b"])).abs().max().item())
