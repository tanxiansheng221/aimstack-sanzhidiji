# -*- coding: utf-8 -*-
# build_fmt5_corpus.py — 趴零五科格式工程量产：v2full 混料五科替换为新格式链
# 产物: v2fmt_1p5b_full_mix.jsonl（其余 8 科 + replay 行字节级原样保留）
# 条数对账: 每科 (src, split) 计数与原混料严格一致
import io, json, time, sys, os
sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import fmt5_generators as F

ROOT = r"H:\AimCode\lab\cpu-native"
MIX_IN = ROOT + r"\v2_1p5b_full_mix.jsonl"
MIX_OUT = ROOT + r"\v2fmt_1p5b_full_mix.jsonl"
FMT5 = {"v2_math": "math", "v2_mod_arith": "mod_arith", "v2_plan": "plan",
        "v2_progexec": "progexec", "v2_seqind": "seqind"}

t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

# ---- PASS1: 统计五科 (src, split) 计数 ----
log("pass1: count (src,split)...")
need = {}
with io.open(MIX_IN, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("src", "")
        if s in FMT5:
            key = (s, r.get("split", "train"))
            need[key] = need.get(key, 0) + 1
for (s, sp), n in sorted(need.items()):
    log("  %-14s %-5s %7d" % (s, sp, n))

# ---- PASS2: 量产五科新格式 ----
log("pass2: generate fmt5 rows...")
pool = {}   # (src, split) -> [rows]
for (s, sp), n in sorted(need.items()):
    subj = FMT5[s]
    import zlib
    seed = {"train": 20261102, "val": 20261103}[sp] + zlib.crc32(s.encode())
    rng_seed = seed % (2 ** 31)
    rows = []
    rng = __import__("random").Random(rng_seed)
    rej = 0
    while len(rows) < n:
        sm = F.gen_sample(rng, subj)
        if sm is None:
            rej += 1
            continue
        rows.append({"q": sm["q"], "a": sm["cot"] + " || " + sm["ans"],
                     "src": s, "split": sp})
        if len(rows) % 20000 == 0:
            log("  %-14s %-5s %7d/%7d (rej %d, %.0fs)" % (s, sp, len(rows), n, rej, time.time() - t0))
    pool[(s, sp)] = rows
    log("  %-14s %-5s DONE %d (rej %d, %.0fs)" % (s, sp, len(rows), rej, time.time() - t0))

# ---- PASS3: 流式替换写出 ----
log("pass3: stream replace...")
idx = {}
n_kept = n_repl = 0
with io.open(MIX_IN, encoding="utf-8") as f, io.open(MIX_OUT, "w", encoding="utf-8", newline="") as w:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("src", "")
        if s in FMT5:
            key = (s, r.get("split", "train"))
            i = idx.get(key, 0)
            row = pool[key][i]
            idx[key] = i + 1
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_repl += 1
        else:
            w.write(line)
            n_kept += 1
log("kept %d rows unchanged, replaced %d rows" % (n_kept, n_repl))

# ---- 对账 ----
for (s, sp), rows in sorted(pool.items()):
    assert idx.get((s, sp)) == len(rows), "count mismatch %s %s: %d vs %d" % (s, sp, idx.get((s, sp)), len(rows))
log("COUNTS MATCH")

# 压缩合并临时池，省盘
del pool
import hashlib
h = hashlib.sha256()
with io.open(MIX_OUT, "rb") as f:
    for blk in iter(lambda: f.read(1 << 20), b""):
        h.update(blk)
log("sha256[:16]=%s  size=%.1fMB  ALL DONE %.0fs" % (
    h.hexdigest()[:16], os.path.getsize(MIX_OUT) / 1e6, time.time() - t0))
