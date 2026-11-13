# -*- coding: utf-8 -*-
# build_pack_v2full2.py — 十三科全量 T=512 打包 v2
# 与 aim_train_real.build_packed_cache 算法逐字对齐（echo 回填/win_pos 样本内 range/单一 perm 洗牌），
# 差异仅：①内存瘦身（bytes 存样本，不再攒 158 万个 int16 小数组）②每 20 万行打进度 ③直接出压缩 npz
import sys, os, io, json, time
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import numpy as np
from aim_train_real import BOS, PAD, EOS, SEP

def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

ROOT = r"H:\AimCode\lab\cpu-native"
MIX = ROOT + r"\v2_1p5b_full_mix.jsonl"
OUT = os.path.join(ROOT, "v2full_pack_cache.npz")
T = 512

# ---- PASS1: 流式读样本 ----
log("PASS1 stream: %s" % MIX)
t0 = time.time()
blobs, qlens = [], []   # blob = q + b"\n" + a（与 build_cache 同构；q 内不含 \n 由上游保证）
n_read = 0
with io.open(MIX, encoding="utf-8") as f:
    for line in f:
        n_read += 1
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["q"].encode("utf-8")[:1600]
        a = r["a"].encode("utf-8")[:1600]
        if q and a and len(q) + len(a) + 3 <= T:   # 原版口径：len(seq)=len(q)+len(a)+3 ≤ T，train/val 同滤
            blobs.append(q + b"\n" + a)
            qlens.append(len(q))
        if n_read % 200000 == 0:
            log("read %d lines, kept %d, %.0fs" % (n_read, len(blobs), time.time() - t0))
n = len(blobs)
log("PASS1 done: %d lines -> %d samples, %.0fs" % (n_read, n, time.time() - t0))

# ---- val 划分（与原版同构：perm 前 1% 当 val，样本级独立数组）----
rng = np.random.RandomState(42)
perm = rng.permutation(n)
n_val = max(200, int(n * 0.01))
val_idx = set(perm[:n_val].tolist())
log("val=%d train=%d" % (n_val, n - n_val))

vs, vq, va, vbuf = [], [], [], bytearray()
for i in sorted(val_idx):
    q, a = blobs[i][:qlens[i]], blobs[i][qlens[i] + 1:]
    vs.append(len(vbuf)); vq.append(len(q)); va.append(len(a))
    vbuf += q + b"\n" + a
vdata = np.frombuffer(bytes(vbuf), dtype=np.uint8)
log("val buffer %.1fMB" % (len(vbuf) / 1e6))

# ---- PASS2: train 装箱（长度排序贪心 + echo 回填，与原版逐字对齐）----
log("sorting by length...")
order = sorted((i for i in range(n) if i not in val_idx), key=lambda i: len(blobs[i]))
log("sorted, packing...")
win_tok, win_am, win_pos = [], [], []
cur_t, cur_a, cur_p = [], [], []
def seq_of(i):
    q, a = blobs[i][:qlens[i]], blobs[i][qlens[i] + 1:]
    ql_i = qlens[i] + 1                      # SEP 在 seq 中的下标
    return [BOS] + list(q) + [SEP] + list(a) + [EOS], ql_i
def flush(final):
    pad = T - len(cur_t)
    if final:
        wt = cur_t + [PAD] * pad
        wa = cur_a + [False] * pad
        wp = cur_p + [0] * pad
    else:
        seq, _ = seq_of(order[pi])            # echo = 下一题问题头部（amask False，本体进下一窗）
        echo = seq[1:1 + pad]
        wt = cur_t + echo
        wa = cur_a + [False] * len(echo)
        wp = cur_p + list(range(len(echo)))
    win_tok.append(np.array(wt, dtype=np.uint16).tobytes())
    win_am.append(np.array(wa, dtype=bool).tobytes())
    win_pos.append(np.array(wp, dtype=np.int16).tobytes())
    del cur_t[:], cur_a[:], cur_p[:]
pi = 0
for pi, i in enumerate(order):
    seq, ql_i = seq_of(i)
    if len(seq) > T:
        continue
    if len(cur_t) + len(seq) > T:
        flush(final=False)
    cur_t += seq
    cur_a += [False] * (ql_i + 1) + [True] * (len(seq) - ql_i - 1)
    cur_p += list(range(len(seq)))
    if (pi + 1) % 200000 == 0:
        log("packed %d/%d windows=%d %.0fs" % (pi + 1, len(order), len(win_tok), time.time() - t0))
if cur_t:
    flush(final=True)
del blobs
log("windows: %d" % len(win_tok))

wt = np.frombuffer(b"".join(win_tok), dtype=np.uint16).reshape(len(win_tok), T)
wa = np.frombuffer(b"".join(win_am), dtype=np.uint8).reshape(len(win_tok), T).astype(bool)
wp = np.frombuffer(b"".join(win_pos), dtype=np.int16).reshape(len(win_tok), T)
del win_tok, win_am, win_pos

perm_w = rng.permutation(len(wt))            # 单一 perm 同步洗牌三数组（与原版一致）
wt, wa, wp = wt[perm_w], wa[perm_w], wp[perm_w]
util = float((wt != PAD).mean())
log("packed: %d windows T=%d util=%.1f%% | val %d samples" % (len(wt), T, util * 100, len(vs)))
log("saving compressed npz -> %s" % OUT)
np.savez_compressed(OUT, win_tok=wt, win_am=wa, win_pos=wp, vdata=vdata,
                    vstarts=np.array(vs, dtype=np.int64),
                    vqlens=np.array(vq, dtype=np.int32), valens=np.array(va, dtype=np.int32))
log("V2FULL PACK DONE  %s  %.1fMB  windows=%d val_rows=%d"
    % (OUT, os.path.getsize(OUT) / 1e6, len(wt), len(vs)))
