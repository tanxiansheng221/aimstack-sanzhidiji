# -*- coding: utf-8 -*-
# build_lang4_slot.py v2 — 槽位段语料：子采样控量 + 全有全无律（一条链内不许混格式）
import io, json, os, sys, time, re, random, zlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
SEED = 20261105
t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)
def rng_of(name):
    return random.Random((zlib.crc32(name.encode()) + SEED) % (2**31))

EQ = re.compile(r'([0-9][0-9+\-*() ]*[+*\-][0-9+\-*() ]*[0-9]) *= *(-?\d+)')
SAFE = re.compile(r'^[0-9+\-*() ]+$')

def slotify(text, stats):
    """全有全无：所有算式都验证通过才返回槽位版；任一失败返回 None"""
    out = []
    pos = 0
    n_eq = 0
    for m in EQ.finditer(text):
        lhs, rhs = m.group(1), m.group(2)
        expr = lhs.strip()
        out.append(text[pos:m.start()])
        if not SAFE.match(expr):
            stats['unsafe'] += 1
            return None
        try:
            v = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            stats['evalfail'] += 1
            return None
        if int(v) != int(rhs):
            stats['mismatch'] += 1
            return None
        out.append('[C:%s] = %s' % (expr, rhs))
        pos = m.end()
        n_eq += 1
    out.append(text[pos:])
    if n_eq == 0:
        stats['noeq'] += 1
        return None
    stats['ok'] += n_eq
    return ''.join(out)

rows_out = []
stats = {'ok': 0, 'mismatch': 0, 'evalfail': 0, 'unsafe': 0, 'noeq': 0, 'raw_kept': 0}
SLOT_SRC = {"v2_math": 12000, "v2_mod_arith": 12000}   # 槽位化配额（行）
REPLAY_SRC = {"v2_csp": 3000, "v2_graph": 3000, "v2_logic": 3000, "v2_causal": 3000,
              "v2_fsm": 3000, "v2_plan": 3000, "v2_progexec": 3000, "v2_seqind": 3000}
pool = {s: [] for s in list(SLOT_SRC) + list(REPLAY_SRC)}
with io.open(os.path.join(ROOT, "v2fmt_1p5b_full_mix.jsonl"), encoding="utf-8") as f:
    for line in f:
        for s in pool:
            if '"src": "%s"' % s in line:
                try:
                    r = json.loads(line)
                except Exception:
                    break
                if r.get("q") and r.get("a"):
                    pool[s].append((r["q"], r["a"]))
                break
for s, lst in pool.items():
    rng = rng_of(s)
    rng.shuffle(lst)
    log("%s pool=%d" % (s, len(lst)))

for s, quota in SLOT_SRC.items():
    rng = rng_of(s + "slot")
    n_conv = n_raw = 0
    for q, a in pool[s]:
        if n_conv >= quota:
            break
        a2 = slotify(a, stats)
        if a2 is None:
            continue
        rows_out.append({"q": q, "a": a2, "src": s + "_slot", "split": "train"})
        n_conv += 1
    # 补一些原始版（模型保留裸算式识别力，配对学习）
    for q, a in pool[s]:
        if n_raw >= quota // 4:
            break
        if slotify(a, stats) is None:
            rows_out.append({"q": q, "a": a, "src": s + "_raw", "split": "train"})
            n_raw += 1
    log("%s: slot=%d raw=%d" % (s, n_conv, n_raw))

for s, quota in REPLAY_SRC.items():
    rng = rng_of(s + "rep")
    for q, a in rng.sample(pool[s], min(quota, len(pool[s]))):
        rows_out.append({"q": q, "a": a, "src": s, "split": "train"})

# 小对话 8000 防语言回退
picked = 0
with io.open(os.path.join(ROOT, "lang3b_mix.jsonl"), encoding="utf-8") as f:
    for line in f:
        if picked >= 8000:
            break
        if '"lang_smalltalk"' in line:
            rows_out.append(json.loads(line))
            picked += 1
rng = rng_of("mix4")
rng.shuffle(rows_out)
log("total rows: %d, stats: %s" % (len(rows_out), stats))

mix_path = os.path.join(ROOT, "lang4_mix.jsonl")
with io.open(mix_path, "w", encoding="utf-8", newline="") as w:
    for r in rows_out:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
log("mix4: %.1fMB" % (os.path.getsize(mix_path) / 1e6))

sys.path.insert(0, ROOT)
import aim_train_real as A
A.build_packed_cache(mix_path, os.path.join(ROOT, "lang4_pack.npz"), T=512, seed=42)
import numpy as np_
z = np_.load(os.path.join(ROOT, "lang4_pack.npz"))
log("pack4: %d windows (%.0fs total)" % (len(z["win_tok"]), time.time() - t0))

shown = 0
with io.open(mix_path, encoding="utf-8") as f:
    for line in f:
        if '"_slot"' in line and shown < 2:
            r = json.loads(line)
            print("SLOT样本:", r["a"][:280], flush=True)
            shown += 1
        if shown >= 2:
            break
