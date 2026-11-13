# -*- coding: utf-8 -*-
# LCCC jsonl.gz → CPED 同格式 CSV (Dialogue_ID, Utterance), 取头部 N 个有效对话
import gzip, json, csv, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SRC = r'H:\AimCode\corpus_500m\lccc_base_train.jsonl.gz'
DST = r'H:\AimCode\corpus_500m\lccc_dialog.csv.gz'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
written = 0
seen = 0
import gzip as _gz
with gzip.open(SRC, 'rt', encoding='utf-8') as f, \
     _gz.open(DST, 'wt', encoding='utf-8', newline='') as out:
    w = csv.writer(out)
    w.writerow(['Dialogue_ID', 'Utterance'])
    for line in f:
        if written >= N:
            break
        seen += 1
        try:
            d = json.loads(line)
        except Exception:
            continue
        us = [u.strip() for u in d if isinstance(u, str) and u.strip()]
        if len(us) < 2:
            continue
        for u in us:
            w.writerow(['D%d' % written, u])
        written += 1
print('扫描 %d 行, 写出 %d 对话' % (seen, written))
