# -*- coding: utf-8 -*-
# build_lang5_pack.py — lang5 补丁包：中文拒绝模板 + 英文创意（story_cont_en 抽样）
import io, json, os, sys, time, random, zlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
SEED = 20261105
t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

rows = []
# 1) 中文拒绝 7000
with io.open(os.path.join(ROOT, "lang5_refusal_mix.jsonl"), encoding="utf-8") as f:
    for line in f:
        rows.append(line)
log("refusal_zh: %d" % len(rows))

# 2) 英文创意：从 creative_v0.jsonl 抽 _en 任务 12000 行 ×2 曝光
base = []
with io.open(os.path.join(ROOT, "creative_corpus", "creative_v0.jsonl"), encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("task", "").endswith("_en") and r.get("q") and r.get("ans"):
            base.append((r["q"], r["ans"]))
log("creative_en pool: %d" % len(base))
rng = random.Random((zlib.crc32(b"cre_en") + SEED) % 2**31)
pick = [base[rng.randrange(len(base))] for _ in range(12000)]
seen = set()
for q, a in pick:
    key = q[:80]
    if key in seen:
        continue
    seen.add(key)
    for _ in range(2):
        rows.append(json.dumps({"q": "[创意:%s]" % "story" and None or "", "a": ""}) ) # placeholder replaced below
rows = rows[:len(rows) - len([1 for _ in pick])] if False else rows
# 重新干净写（上面 placeholder 作废）
rows = []
with io.open(os.path.join(ROOT, "lang5_refusal_mix.jsonl"), encoding="utf-8") as f:
    for line in f:
        rows.append(line)
seen = set()
n_en = 0
for q, a in pick:
    key = q[:80]
    if key in seen:
        continue
    seen.add(key)
    qq = "[creative_en] %s" % q
    aa = a
    if len(qq.encode("utf-8")) + len(aa.encode("utf-8")) > 460:
        continue
    for _ in range(2):
        rows.append(json.dumps({"q": qq, "a": aa, "src": "lang_creative_en", "split": "train"}, ensure_ascii=False) + "\n")
        n_en += 1
log("creative_en rows: %d (unique %d)" % (n_en, len(seen)))

out = os.path.join(ROOT, "lang5_mix.jsonl")
with io.open(out, "w", encoding="utf-8", newline="") as w:
    for line in rows:
        if not line.endswith("\n"):
            line += "\n"
        w.write(line)
log("mix5: %d rows, %.1fMB (%.0fs)" % (len(rows), os.path.getsize(out) / 1e6, time.time() - t0))

sys.path.insert(0, ROOT)
import aim_train_real as A
A.build_packed_cache(out, os.path.join(ROOT, "lang5_pack.npz"), T=512, seed=42)
import numpy as np_
z = np_.load(os.path.join(ROOT, "lang5_pack.npz"))
log("pack5: %d windows (%.0fs total)" % (len(z["win_tok"]), time.time() - t0))
