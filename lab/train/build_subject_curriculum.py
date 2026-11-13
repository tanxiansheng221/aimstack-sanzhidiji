# -*- coding: utf-8 -*-
# build_subject_curriculum.py — 13 科课程包生成器
# 产物：exam_cache.npz（每科 300 题留出卷，主 val/训练区全隔离）+ pool_<科>.npz ×13（每科高曝光特训池）
# 曝光账：段 3000 步 × 80% × 32 窗/步 ÷ ~1200 窗 ≈ 64 次/窗（>50 底线）
import sys, os, io, json, time
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import numpy as np
import aim_train_real as A

ROOT = r"H:\AimCode\lab\cpu-native"
MIX = sys.argv[1] if len(sys.argv) > 1 else ROOT + r"\v2_1p5b_full_mix.jsonl"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""   # 产物后缀，如 "_fmt"；空串=覆盖原版
T = 512
EXAM_N = 300     # 每科留出卷题数
FOCUS_N = 2800   # 每科特训子池样本数（打包后 ~1200 窗）

t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

# ---- PASS1: 按原打包口径过滤 + 记 (src, line) ----
log("stream mix...")
lines, srcs = [], []
with io.open(MIX, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["q"].encode("utf-8")[:1600]
        a = r["a"].encode("utf-8")[:1600]
        if q and a and len(q) + len(a) + 3 <= T:
            lines.append(line)
            srcs.append(r["src"])
n = len(lines)
log("kept %d samples" % n)

# ---- 主 val 排除（复现打包器 perm）----
rng = np.random.RandomState(42)
perm = rng.permutation(n)
val_set = set(perm[:max(200, int(n * 0.01))].tolist())
log("main val excluded: %d" % len(val_set))

# ---- 按科分组 ----
by_src = {}
for i, s in enumerate(srcs):
    if i not in val_set:
        by_src.setdefault(s, []).append(i)
subj_names = sorted(by_src.keys())
log("subjects: %d" % len(subj_names))
for s in subj_names:
    log("  %-18s %7d" % (s, len(by_src[s])))

# ---- 每科抽考试卷 + 特训子池 ----
er = np.random.RandomState(2026)
used = {}          # subj -> 已占用 index 集合（考试+特训互斥）
exam_data, exam_starts, exam_ql, exam_al, exam_subj = [], [], [], [], []
subj_meta = {}
for si, s in enumerate(subj_names):
    pool_idx = [i for i in by_src[s] if i not in used.get(s, set())]
    picks = er.choice(pool_idx, size=EXAM_N + FOCUS_N, replace=False).tolist()
    exam_idx, focus_idx = picks[:EXAM_N], picks[EXAM_N:]
    used[s] = set(picks)
    # 考试卷：val 样本级数组（q+\n+a）
    for i in exam_idx:
        r = json.loads(lines[i])
        q = r["q"].encode("utf-8")[:1600]
        a = r["a"].encode("utf-8")[:1600]
        exam_starts.append(len(b"".join(exam_data)) if False else 0)  # 占位，下面统一算
        exam_ql.append(len(q)); exam_al.append(len(a))
        exam_data.append(q + b"\n" + a)
        exam_subj.append(si)
    # 特训子池：子集 jsonl → 打包
    short = s.replace("v2_", "").replace("replay_", "")
    sub_path = os.path.join(ROOT, "_curri%s_%s.jsonl" % (TAG, short))
    with io.open(sub_path, "w", encoding="utf-8") as w:
        for i in focus_idx:
            w.write(lines[i])
    pool_npz = os.path.join(ROOT, "pool%s_%s.npz" % (TAG, short))
    A.build_packed_cache(sub_path, pool_npz, T=T, seed=42)
    z = np.load(pool_npz)
    log("%-18s exam %d + focus %d -> %d windows (train %d)" %
        (s, EXAM_N, FOCUS_N, len(z["win_tok"]), len(z["win_tok"]) - 0))
    subj_meta[s] = {"si": si, "short": short, "exam_n": EXAM_N, "focus_windows": int(len(z["win_tok"]))}

# exam 数组重算 starts（顺序拼接）
vdata = bytearray(); starts = []
for blob in exam_data:
    starts.append(len(vdata)); vdata += blob
z = dict(
    vdata=np.frombuffer(bytes(vdata), dtype=np.uint8),
    vstarts=np.array(starts, dtype=np.int64),
    vqlens=np.array(exam_ql, dtype=np.int32),
    valens=np.array(exam_al, dtype=np.int32),
    subj=np.array(exam_subj, dtype=np.int16),
)
np.savez(os.path.join(ROOT, "exam_cache%s.npz" % TAG), **z)
meta = {"subjects": subj_names, "meta": subj_meta, "exam_total": len(exam_subj)}
with io.open(os.path.join(ROOT, "exam_meta%s.json" % TAG), "w", encoding="utf-8") as w:
    json.dump(meta, w, ensure_ascii=False, indent=1)
log("EXAM CACHE DONE: %d questions / %d subjects" % (len(exam_subj), len(subj_names)))
log("cleanup temp jsonl...")
for s in subj_names:
    short = s.replace("v2_", "").replace("replay_", "")
    p = os.path.join(ROOT, "_curri%s_%s.jsonl" % (TAG, short))
    if os.path.exists(p):
        os.remove(p)
log("ALL DONE %.0fs" % (time.time() - t0))
