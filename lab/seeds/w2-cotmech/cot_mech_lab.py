# -*- coding: utf-8 -*-
# EXPERIMENT P2 (chapter 71 addendum): mechanism attribution for the CoT result.
# Q: COT's 100% val y1 — via "virtual chain" (pos3 hidden carries supervised h1,
#    pos4 consumes it) or via a direct 3-ary circuit over a,b,c?
# Test: retrain COT (identical config/seed to cot_evolution_lab), then evaluate
#    acc_y1 with attention SOURCE pos3 masked for query pos4 only (pos3's own
#    computation untouched, h1 head unmasked as sanity).
# Verdict: collapse -> virtual chain; intact -> direct circuit.
import numpy as np, json, time, os, sys

R = lambda s: print(s, flush=True)
V, NV, D, L, H1, T = 98, 97, 96, 2, 256, 5
B, STEPS, LR, WD = 128, 6000, 2e-3, 1e-4
PAD = 97
LOG_EVERY = 1000
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2
OUT = "cot_mech%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")

dRng = np.random.default_rng(7)
perm = dRng.permutation(NV**3)
tr_idx = perm[:20000]; va_idx = perm[20000:24096]
def unflatten(idx):
    return idx // (NV*NV), (idx // NV) % NV, idx % NV
Atr, Btr_, Ctr = unflatten(tr_idx); Ava, Bva, Cva = unflatten(va_idx)
H1tr = (Atr + Btr_) % NV; Ytr = (H1tr + Ctr) % NV
H1va = (Ava + Bva) % NV;  Yva = (H1va + Cva) % NV

def make_params(seed):
    rng = np.random.default_rng(seed)
    sc = 0.8 / np.sqrt(D)
    p = {"E": rng.normal(0, sc, (V, D)), "P": rng.normal(0, sc, (T, D))}
    for l in range(L):
        for n in ("Wq", "Wk", "Wv", "Wo"):
            p["%s%d" % (n, l)] = rng.normal(0, sc, (D, D))
        p["W1%d" % l] = rng.normal(0, sc, (D, H1)); p["b1%d" % l] = np.zeros(H1)
        p["W2%d" % l] = rng.normal(0, sc, (H1, D)); p["b2%d" % l] = np.zeros(D)
    p["Wh"] = rng.normal(0, sc, (D, NV)); p["Wy"] = rng.normal(0, sc, (D, NV))
    return p

def fwd(p, tok, block=None):
    """block = (query_pos, masked_source_pos) or None. Mask applies at every layer."""
    h = p["E"][tok] + p["P"][None, :, :]
    for l in range(L):
        hc = h
        q = hc @ p["Wq%d" % l]; k = hc @ p["Wk%d" % l]; v = hc @ p["Wv%d" % l]
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)
        if block is not None:
            qp, sp = block
            s[:, qp, sp] = -np.inf
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s); A = e / e.sum(-1, keepdims=True)
        o = A @ v
        ha = hc + o @ p["Wo%d" % l]
        z = ha @ p["W1%d" % l] + p["b1%d" % l]
        h = ha + np.maximum(z, 0.0) @ p["W2%d" % l] + p["b2%d" % l]
    return h

def ce_stats(h, W, pos, tgt):
    lg = h[:, pos, :] @ W
    m = lg.max(1, keepdims=True)
    lsm = lg - m - np.log(np.exp(lg - m).sum(1, keepdims=True))
    nll = float(-lsm[np.arange(len(tgt)), tgt].mean() / np.log(2))
    acc = float((lg.argmax(1) == tgt).mean())
    return acc, nll

def loss_and_grads(p, x, h1, y):
    Bn = x.shape[0]
    tok1 = np.full((Bn, T), PAD, dtype=np.int64); tok1[:, :3] = x
    h, caches = fwd_caches(p, tok1)
    dh = np.zeros_like(h); tot = 0.0; g = {k: np.zeros_like(v) for k, v in p.items()}
    for pos, name, tgt, key in ((3, "Wh", h1, "h1"), (4, "Wy", y, "y1")):
        lg = h[:, pos, :] @ p[name]
        lg = lg - lg.max(1, keepdims=True)
        Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
        nll = float(-np.log(sm[np.arange(Bn), tgt] + 1e-12).mean())
        tot += nll
        dl = sm.copy(); dl[np.arange(Bn), tgt] -= 1.0; dl /= Bn
        g[name] = h[:, pos, :].T @ dl
        dh[:, pos, :] += dl @ p[name].T
    g1 = bwd(p, tok1, caches, dh)
    for kk in g1: g[kk] = g[kk] + g1[kk]
    return tot, g

def fwd_caches(p, tok):
    h = p["E"][tok] + p["P"][None, :, :]
    caches = []
    for l in range(L):
        hc = h
        q = hc @ p["Wq%d" % l]; k = hc @ p["Wk%d" % l]; v = hc @ p["Wv%d" % l]
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s); A = e / e.sum(-1, keepdims=True)
        o = A @ v
        ha = hc + o @ p["Wo%d" % l]
        z = ha @ p["W1%d" % l] + p["b1%d" % l]
        r = np.maximum(z, 0.0)
        h = ha + r @ p["W2%d" % l] + p["b2%d" % l]
        caches.append((hc, q, k, v, A, o, ha, z, r))
    return h, caches

def bwd(p, tok, caches, dh):
    g = {"E": np.zeros_like(p["E"]), "P": np.zeros_like(p["P"])}
    for l in reversed(range(L)):
        hc, q, k, v, A, o, ha, z, r = caches[l]
        dr = dh @ p["W2%d" % l].T
        dz = dr * (z > 0)
        dha = dh + dz @ p["W1%d" % l].T
        g["W2%d" % l] = r.reshape(-1, H1).T @ dh.reshape(-1, D)
        g["b2%d" % l] = dh.reshape(-1, D).sum(0)
        g["W1%d" % l] = ha.reshape(-1, D).T @ dz.reshape(-1, H1)
        g["b1%d" % l] = dz.reshape(-1, H1).sum(0)
        do = dha @ p["Wo%d" % l].T
        g["Wo%d" % l] = o.reshape(-1, D).T @ dha.reshape(-1, D)
        dv = A.transpose(0, 2, 1) @ do
        g["Wv%d" % l] = hc.reshape(-1, D).T @ dv.reshape(-1, D)
        dA = do @ v.transpose(0, 2, 1)
        ds = A * (dA - (dA * A).sum(-1, keepdims=True))
        dq = ds @ k / np.sqrt(D)
        dk = ds.transpose(0, 2, 1) @ q / np.sqrt(D)
        g["Wq%d" % l] = hc.reshape(-1, D).T @ dq.reshape(-1, D)
        g["Wk%d" % l] = hc.reshape(-1, D).T @ dk.reshape(-1, D)
        dh = dha + dq @ p["Wq%d" % l].T + dk @ p["Wk%d" % l].T + dv @ p["Wv%d" % l].T
    np.add.at(g["E"], tok.reshape(-1), dh.reshape(-1, D))
    g["P"] = dh.sum(0)
    return g

def train_cot():
    p = make_params(42 + _OFF)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    bRng = np.random.default_rng(1001 + _OFF)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / STEPS)))
        bi = bRng.integers(0, len(tr_idx), size=B)
        x = np.stack([Atr[bi], Btr_[bi], Ctr[bi]], 1)
        _, g = loss_and_grads(p, x, H1tr[bi], Ytr[bi])
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % LOG_EVERY == 0:
            ev = evaluate(p, block=None, n=2048)
            R("  step %5d %s (%.0fs)" % (step, ev, time.perf_counter() - t0))
    return p

def evaluate(p, block, n=4096, chunk=1024):
    ns = min(n, len(va_idx)); accs = {"h1": 0, "y1": 0}
    for s in range(0, ns, chunk):
        e = min(s + chunk, ns)
        x = np.stack([Ava[s:e], Bva[s:e], Cva[s:e]], 1)
        tok = np.full((e - s, T), PAD, dtype=np.int64); tok[:, :3] = x
        h = fwd(p, tok, block=block)
        a1, _ = ce_stats(h, p["Wh"], 3, H1va[s:e])
        a2, _ = ce_stats(h, p["Wy"], 4, Yva[s:e])
        accs["h1"] += a1 * (e - s); accs["y1"] += a2 * (e - s)
    return {"acc_h1": round(accs["h1"] / ns, 4), "acc_y1": round(accs["y1"] / ns, 4)}

if __name__ == "__main__":
    R("=== EXPERIMENT P2: mechanism attribution (mask query4 <- source3) ===")
    R("training COT (same seed/config as exp P)...")
    p = train_cot()
    res = {}
    res["control_full"] = evaluate(p, block=None)
    R("control (no mask):        %s" % res["control_full"])
    res["mask_y4_src3"] = evaluate(p, block=(4, 3))
    R("mask pos4<-pos3 (KEY):    %s" % res["mask_y4_src3"])
    res["mask_y4_src0_a"] = evaluate(p, block=(4, 0))
    R("mask pos4<-pos0 (a ctrl): %s" % res["mask_y4_src0_a"])
    json.dump(res, open(OUT, "w"), indent=1)
    R("saved %s" % OUT)
