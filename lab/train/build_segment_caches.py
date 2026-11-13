# -*- coding: utf-8 -*-
# build_segment_caches.py — 13 个段缓存 = 80% 科目特训窗 + 20% 全科轮转回放窗（考试题严格排除）
# 曝光账：段 3000 步 × 32 窗/步 × 80% ÷ 池窗数 ≥ 51 次/窗
import sys, os, io, json, time
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import numpy as np
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
MIX = ROOT + r"\v2_1p5b_full_mix.jsonl"
T = 512
REPLAY_N = 570   # 每段回放样本数（打包约 300 窗 = 20% 配比）

t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

meta = json.load(open(ROOT + r"\exam_meta.json", encoding="utf-8"))
names = meta["subjects"]

# ---- 重放原构建器的抽签序列，精确复现 13 科考试卷题目（用于回放排除）----
log("stream mix + reproduce exam picks...")
rows, srcs = [], []
with io.open(MIX, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["q"].encode("utf-8")[:1600]
        a = r["a"].encode("utf-8")[:1600]
        if q and a and len(q) + len(a) + 3 <= T:
            rows.append(line)
            srcs.append(r["src"])
n = len(rows)
rng = np.random.RandomState(42)
perm = rng.permutation(n)
val_set = set(perm[:max(200, int(n * 0.01))].tolist())
by_src = {}
for i, s in enumerate(srcs):
    if i not in val_set:
        by_src.setdefault(s, []).append(i)
er = np.random.RandomState(2026)
exam_set = set()
for s in names:   # 与 build_subject_curriculum.py 同序同调用序列
    pool_idx = by_src[s]
    picks = er.choice(pool_idx, size=300 + 2800, replace=False).tolist()
    exam_set.update(picks[:300])
log("exam excluded: %d lines" % len(exam_set))

# ---- 逐科建段缓存 ----
train_all = [i for i in range(n) if i not in val_set]
seg_meta = {}
for k, s in enumerate(names):
    short = s.replace("v2_", "").replace("replay_", "")
    pz = np.load(ROOT + r"\pool_%s.npz" % short)
    wt, wa, wp = pz["win_tok"], pz["win_am"], pz["win_pos"]
    vd, vs_, vq, va = pz["vdata"], pz["vstarts"], pz["vqlens"], pz["valens"]
    n_pool = len(wt)
    # 回放：全域抽（排除考试+主 val），每段独立种子
    cand = [i for i in train_all if i not in exam_set] if k == 0 else None
    if k == 0:
        replay_pool = np.array(cand)
    rr = np.random.RandomState(500 + k)
    ridx = rr.choice(replay_pool, size=REPLAY_N, replace=False)
    sub = ROOT + r"\_replay_%d.jsonl" % k
    with io.open(sub, "w", encoding="utf-8") as w:
        for i in ridx:
            w.write(rows[i])
    rpz_path = ROOT + r"\_replay_%d.npz" % k
    A.build_packed_cache(sub, rpz_path, T=T, seed=42)
    rz = np.load(rpz_path)
    rwt, rwa, rwp = rz["win_tok"], rz["win_am"], rz["win_pos"]
    rz.close()
    # 合并 + 洗牌
    W = n_pool + len(rwt)
    wt2 = np.concatenate([wt, rwt]); wa2 = np.concatenate([wa, rwa]); wp2 = np.concatenate([wp, rwp])
    pm = np.random.RandomState(900 + k).permutation(W)
    wt2, wa2, wp2 = wt2[pm], wa2[pm], wp2[pm]
    np.savez(ROOT + r"\seg_%s.npz" % short, win_tok=wt2, win_am=wa2, win_pos=wp2,
             vdata=vd, vstarts=vs_, vqlens=vq, valens=va)
    pz.close()
    seg_meta[s] = {"short": short, "windows": int(W), "pool_windows": int(n_pool), "replay_windows": int(len(rwt))}
    log("seg_%-14s %d win (pool %d + replay %d) 曝光=%.0f" %
        (short, W, n_pool, len(rwt), 3000 * 32 * 0.8 / n_pool))

json.dump(seg_meta, open(ROOT + r"\seg_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("ALL SEGMENTS DONE %.0fs" % (time.time() - t0))
