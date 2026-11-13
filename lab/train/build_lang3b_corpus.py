# -*- coding: utf-8 -*-
# build_lang3b_corpus.py — lang3 + CPED（中文情绪对白）扩充版
import csv, io, json, os, sys, time, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"H:\AimCode\lab\cpu-native"
CPED = os.path.join(ROOT, "cped_corpus")
t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

BAD = re.compile(u'[\u0400-\u04FF\u0600-\u06FF\uAC00-\uD7AF\u3040-\u30FF]')
CJK = re.compile(u'[\u4E00-\u9FFF]')

rows = []
seen = set()

# ---- CPED：相邻轮次 → [情绪:X] q / a ----
log("CPED convert ...")
n_pairs = 0
for fn in ["train_split.csv", "valid_split.csv"]:
    path = os.path.join(CPED, fn)
    with io.open(path, encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        dlg = []          # 当前对白的 (emotion, utterance) 序列
        cur = None
        for r in rd:
            if r["Dialogue_ID"] != cur:
                # 结算上一段
                for i in range(len(dlg) - 1):
                    q_e, q_u = dlg[i]
                    a_e, a_u = dlg[i + 1]
                    if q_u == a_u:
                        continue
                    q = "[%s] %s" % (q_e, q_u)
                    key = (q, a_u)
                    if key in seen:
                        continue
                    if BAD.search(q) or BAD.search(a_u):
                        continue
                    if len(CJK.findall(a_u)) < 2 or len(a_u) < 3:
                        continue
                    if len(q.encode("utf-8")) + len(a_u.encode("utf-8")) > 460:
                        continue
                    seen.add(key)
                    rows.append({"q": q, "a": a_u, "src": "lang_cped", "split": "train"})
                    n_pairs += 1
                dlg = []
                cur = r["Dialogue_ID"]
            dlg.append((r["Emotion"], r["Utterance"].strip()))
        # 结算最后一段
        for i in range(len(dlg) - 1):
            q_e, q_u = dlg[i]
            a_e, a_u = dlg[i + 1]
            if q_u == a_u:
                continue
            q = "[%s] %s" % (q_e, q_u)
            key = (q, a_u)
            if key in seen or BAD.search(q) or BAD.search(a_u):
                continue
            if len(CJK.findall(a_u)) < 2 or len(a_u) < 3:
                continue
            if len(q.encode("utf-8")) + len(a_u.encode("utf-8")) > 460:
                continue
            seen.add(key)
            rows.append({"q": q, "a": a_u, "src": "lang_cped", "split": "train"})
            n_pairs += 1
log("CPED pairs: %d (%.0fs)" % (n_pairs, time.time() - t0))

# ---- 并入 lang3_mix（复用已过滤的中文语料）----
mix3 = os.path.join(ROOT, "lang3_mix.jsonl")
out = os.path.join(ROOT, "lang3b_mix.jsonl")
n3 = 0
with io.open(mix3, encoding="utf-8") as f, io.open(out, "w", encoding="utf-8", newline="") as w:
    for line in f:
        w.write(line)
        n3 += 1
    for r in rows:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
log("mix3b: %d lang3 rows + %d cped rows, %.1fMB" % (n3, len(rows), os.path.getsize(out) / 1e6))

# ---- 打包 ----
sys.path.insert(0, ROOT)
import aim_train_real as A
A.build_packed_cache(out, os.path.join(ROOT, "lang3b_pack.npz"), T=512, seed=42)
import numpy as np_
z = np_.load(os.path.join(ROOT, "lang3b_pack.npz"))
w = z["win_tok"]
sample = w[:2000]
cjk = ((sample >= 0xE4) & (sample <= 0xE9)).mean()
log("pack3b: %d windows 汉字字节=%.3f (%.0fs total)" % (len(w), cjk, time.time() - t0))
