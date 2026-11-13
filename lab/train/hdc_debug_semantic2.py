# -*- coding: utf-8 -*-
# §22-B/C: 情绪价值两块试金石 (胜出管线: HDC 编码 + 闭式头, 对话级 split)
#   B: seeker 话术 → emotion_type (8 类, 合并小类)
#   C: seeker 最后一句 → supporter 下一手策略 (8 类) —— 情绪价值的动作层
import numpy as np, json, sys, re, hashlib, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

d = json.load(open(r'H:\AimCode\lab\cpu-native\esconv\ESConv.json', encoding='utf-8'))
rng = np.random.default_rng(7)
order = rng.permutation(len(d))
n_tr = int(len(d)*0.8)

D = 2048
phi_tab = np.random.default_rng(99).uniform(0, 1, D)
def encode(t):
    t = re.sub(r'[^a-z ]', ' ', t.lower())
    g = [t[i:i+3] for i in range(max(1, len(t)-2))]
    cnt = {}
    for s in g:
        idx = int(hashlib.md5(s.encode()).hexdigest(), 16) % D
        cnt[idx] = cnt.get(idx, 0) + 1
    v = np.zeros(D, dtype=complex)
    for idx, c in cnt.items():
        v[idx] += c*np.exp(2j*np.pi*phi_tab[idx])
    return np.concatenate([v.real, v.imag])/max(np.linalg.norm(v), 1e-9)

def run_task(name, texts, labels, tr, te, lam=1e-3):
    classes = sorted(set(labels))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[l] for l in labels])
    X = np.stack([encode(t) for t in texts])
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    Yoh = np.zeros((len(ytr), len(classes))); Yoh[np.arange(len(ytr)), ytr] = 1.0
    G = Xtr.T@Xtr/len(ytr) + lam*np.eye(X.shape[1])
    vo = np.linalg.solve(G, Xtr.T@Yoh/len(ytr))
    acc = float(np.mean(np.argmax(Xte@vo, 1) == yte))
    pred = np.argmax(Xte@vo, 1)
    f1s = []
    for c in range(len(classes)):
        tp = np.sum((pred == c) & (yte == c)); fp = np.sum((pred == c) & (yte != c))
        fn = np.sum((pred != c) & (yte == c))
        f1s.append(2*tp/max(2*tp+fp+fn, 1))
    maj = float(np.mean(ytr == np.bincount(ytr).argmax()))
    print('%s: %d 类 | majority %.3f → acc %.3f (×%.1f)  macro-F1 %.3f'
          % (name, len(classes), maj, acc, acc/max(maj, 1e-9), float(np.mean(f1s))))
    print('   分布:', collections.Counter(labels).most_common())
    return acc

print('=== §22-B: 情绪类型检测 ===')
texts = [' '.join(u['content'] for u in x['dialog'] if u['speaker'] == 'seeker') for x in d]
KEEP = {'anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust'}
labels = [x['emotion_type'] if x['emotion_type'] in KEEP else 'other' for x in d]
run_task('B 情绪检测', texts, labels, order[:n_tr], order[n_tr:])

print('\n=== §22-C: 支持策略预测 (seeker 末句 → supporter 下一手) ===')
STRAT = {'Question', 'Restatement or Paraphrasing', 'Reflection of Feelings',
         'Self-disclosure', 'Affirmation and Reassurance', 'Providing Suggestions',
         'Information', 'Others'}
tc, lc = [], []
did = []
for di, x in enumerate(d):
    dlg = x['dialog']
    for i in range(1, len(dlg)):
        if dlg[i]['speaker'] == 'supporter':
            s = dlg[i]['annotation']['strategy']
            prev = ' '.join(u['content'] for u in dlg[:i] if u['speaker'] == 'seeker')
            if prev.strip() and s in STRAT:
                tc.append(prev); lc.append(s); did.append(di)
did = np.array(did)
tr = np.isin(did, order[:n_tr]); te = np.isin(did, order[n_tr:])
acc = run_task('C 策略预测', tc, lc, tr, te)
