# -*- coding: utf-8 -*-
# build_spec5_pack.py — 五科特训包：5 个 fmt 高曝光池 + v2fmt 包 50% 回放窗口，合成单缓存
import numpy as np, os

ROOT = r"H:\AimCode\lab\cpu-native"
POOLS = ["math", "mod_arith", "plan", "progexec", "seqind"]

wt, wa, wp = [], [], []
for p in POOLS:
    z = np.load(os.path.join(ROOT, "pool_fmt_%s.npz" % p))
    wt.append(z["win_tok"]); wa.append(z["win_am"]); wp.append(z["win_pos"])
    print(p, "windows:", len(z["win_tok"]))
wt = np.concatenate(wt); wa = np.concatenate(wa); wp = np.concatenate(wp)
n_pool = len(wt)
print("pool windows total:", n_pool)

zv = np.load(os.path.join(ROOT, "v2fmt_pack_cache.npz"))
rng = np.random.RandomState(20261103)
n_replay = n_pool  # 50% 回放
pick = rng.choice(len(zv["win_tok"]), size=n_replay, replace=False)
wt2 = zv["win_tok"][pick]; wa2 = zv["win_am"][pick]; wp2 = zv["win_pos"][pick]

W = np.concatenate([wt, wt2]); A = np.concatenate([wa, wa2]); P = np.concatenate([wp, wp2])
perm = rng.permutation(len(W))
W, A, P = W[perm], A[perm], P[perm]
print("final windows:", len(W))

out = os.path.join(ROOT, "fmt5spec_pack.npz")
np.savez_compressed(out,
                    win_tok=W, win_am=A, win_pos=P,
                    vdata=zv["vdata"], vstarts=zv["vstarts"],
                    vqlens=zv["vqlens"], valens=zv["valens"])
print("saved %s %.1fMB" % (out, os.path.getsize(out) / 1e6))
print("SPEC5 PACK DONE")
