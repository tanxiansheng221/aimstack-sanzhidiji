# -*- coding: utf-8 -*-
# build_lang3c_corpus.py — lang3b + ESConv（英文情绪支持）→ 双语包（中+英，其余全砍）
import json, io, os, sys, time, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"H:\AimCode\lab\cpu-native"
t0 = time.time()
def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

BAD = re.compile(u'[\u0400-\u04FF\u0600-\u06FF\uAC00-\uD7AF\u3040-\u30FF]')
rows = []
seen = set()

data = json.load(io.open(os.path.join(ROOT, "esconv", "ESConv.json"), encoding="utf-8"))
log("dialogues: %d" % len(data))
for d in data:
    emo = d.get("emotion_type", "anxiety")
    dlg = [(t["speaker"], t["content"].strip()) for t in d["dialog"]]
    dlg = [(s, c) for s, c in dlg if c]
    # situation 作开场上下文的一对
    sit = (d.get("situation", "") or "").strip().replace("\n", " ")
    if sit and len(dlg) >= 2:
        pairs = [("[emotion:%s] %s" % (emo, sit[:220]), dlg[1][1])]
    else:
        pairs = []
    for i in range(len(dlg) - 1):
        s_q, q_u = dlg[i]
        a_u = dlg[i + 1][1]
        pairs.append(("[emotion:%s] %s" % (emo, q_u), a_u))
    for q, a in pairs:
        q = q.replace("\n", " ").strip()
        a = a.replace("\n", " ").strip()
        if not q or not a or q == a:
            continue
        if BAD.search(q) or BAD.search(a):
            continue
        key = (q, a)
        if key in seen:
            continue
        if len(q.encode("utf-8")) + len(a.encode("utf-8")) > 460:
            q = q[:200]
            if len(q.encode("utf-8")) + len(a.encode("utf-8")) > 460:
                continue
        seen.add(key)
        rows.append({"q": q, "a": a, "src": "lang_esconv", "split": "train"})
log("ESConv pairs: %d (%.0fs)" % (len(rows), time.time() - t0))

# ×3 曝光并入 lang3b mix
mix3b = os.path.join(ROOT, "lang3b_mix.jsonl")
out = os.path.join(ROOT, "lang3c_mix.jsonl")
n3b = 0
with io.open(mix3b, encoding="utf-8") as f, io.open(out, "w", encoding="utf-8", newline="") as w:
    for line in f:
        w.write(line)
        n3b += 1
    added = 0
    for _ in range(3):
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
log("mix3c: %d lang3b rows + %d esconv rows(×3), %.1fMB" % (n3b, added, os.path.getsize(out) / 1e6))

sys.path.insert(0, ROOT)
import aim_train_real as A
A.build_packed_cache(out, os.path.join(ROOT, "lang3c_pack.npz"), T=512, seed=42)
import numpy as np_
z = np_.load(os.path.join(ROOT, "lang3c_pack.npz"))
w = z["win_tok"]
sample = w[:2000]
cjk = ((sample >= 0xE4) & (sample <= 0xE9)).mean()
ascii_ratio = (sample < 128).mean()
log("pack3c: %d windows 汉字=%.3f ASCII=%.3f (%.0fs total)" % (len(w), cjk, ascii_ratio, time.time() - t0))
