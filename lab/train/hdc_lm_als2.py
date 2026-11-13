# -*- coding: utf-8 -*-
# §23-v6: v4 完整架构(n-gram直连+HDC) + ALS 学字向量 E —— 公平对决「学到的编码 vs 随机编码」
#   每轮: (a) 全特征闭式岭回归解 W (Gram 累计, 零梯度)
#         (b) 固定 W 解 E: 逐字 D×D 正规方程 (批量)
#     e_c: H_c = Σ_i n_{c,i} (p_i ⊙ G ⊙ p_i) + λe I,  G = W_hdc W_hdcᵀ (D×D)
#          rhs_c = Σ_i p_i ⊙ (W_hdc @ R[c*K+i]),  p_i = PW[i] (含位置权)
#   阻尼: E ← E + η(E_als − E), η=0.5 抗振荡; λe=10 抗过拟合
import torch, csv, sys, collections, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
dev = 'cuda'

rows = list(csv.reader(open('/workspace/train_split.csv', encoding='utf-8-sig')))
hdr = rows[0]; ui = hdr.index('Utterance'); di = hdr.index('Dialogue_ID')
dialogs = collections.OrderedDict()
for r in rows[1:]:
    dialogs.setdefault(r[di], []).append(r[ui])
texts = [''.join(us) for us in dialogs.values()]

cc = collections.Counter(''.join(texts))
VOC = 3000
vocab = [c for c, _ in cc.most_common(VOC)]
vmap = {c: i for i, c in enumerate(vocab)}
UNK = len(vocab)

nprng = np.random.default_rng(7)
perm = nprng.permutation(len(texts))
n_tr = int(len(texts)*0.85)
tr_idx = set(perm[:n_tr].tolist())

K, D, HB = 8, 1024, 16000
E = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (VOC+1, D)).astype(np.float32)).to(dev)
PW = torch.tensor([0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 2.0, 4.0], device=dev)[:, None] * \
     torch.from_numpy(nprng.normal(0, 1, (K, D)).astype(np.float32)).to(dev)

def make_windows(tlist, maxn=None, seed=3):
    W, T = [], []
    for t in tlist:
        ids = [vmap.get(c, UNK) for c in t]
        for i in range(K, len(ids)):
            if ids[i] != UNK:
                W.append(ids[i-K:i]); T.append(ids[i])
    W = np.array(W, dtype=np.int64); T = np.array(T, dtype=np.int64)
    if maxn and len(T) > maxn:
        sel = np.random.default_rng(seed).choice(len(T), maxn, replace=False)
        W, T = W[sel], T[sel]
    return W, T

Wtr, Ttr = make_windows([texts[i] for i in tr_idx])
Wte, Tte = make_windows([texts[i] for i in perm[n_tr:]], maxn=60000, seed=4)
NTR = len(Ttr)
print('训练 %d 测试 %d' % (NTR, len(Tte)))
Wtr_t = torch.from_numpy(Wtr).to(dev)
Ytr = torch.zeros(NTR, VOC, device=dev)
Ytr[torch.arange(NTR, device=dev), torch.from_numpy(Ttr).to(dev)] = 1.0

NF = 3000 + HB + HB + D
OH, HP, HT, HC = 0, 3000, 3000+HB, 3000+2*HB
def encode(Wnp, s, e):
    Wi = torch.from_numpy(Wnp[s:e]).to(dev)
    n = Wi.shape[0]
    X = torch.zeros(n, NF, device=dev)
    ar = torch.arange(n, device=dev)
    X[ar, OH + Wi[:, -1]] = 1.0
    X[ar, HP + ((Wi[:, -2]*31 + Wi[:, -1]) % HB)] += 1.0
    X[ar, HT + ((Wi[:, -3]*961 + Wi[:, -2]*31 + Wi[:, -1]) % HB)] += 1.0
    ctx = torch.zeros(n, D, device=dev)
    for i in range(K):
        ctx += E[Wi[:, i]]*PW[i]
    X[:, HC:] = ctx
    return X, Wi

def eval_model(W):
    top1 = top3 = tot = 0
    for s in range(0, len(Tte), 50000):
        X, _ = encode(Wte, s, min(s+50000, len(Tte)))
        Tb = torch.from_numpy(Tte[s:s+50000]).to(dev)
        sc = X@W
        top1 += int((sc.argmax(1) == Tb).sum())
        top3 += int((sc.topk(3, 1).indices == Tb[:, None]).any(1).sum())
        tot += X.shape[0]
    return top1/tot, top3/tot

NCI = torch.zeros(VOC+1, K, device=dev)
for i in range(K):
    NCI[:, i] = torch.bincount(Wtr_t[:, i], minlength=VOC+1).to(torch.float32)

lam, lam_e, ETA = 1e-4, 10.0, 0.5
I_D = torch.eye(D, device=dev)
BLK = 50000
W = None
print('=== §23-v6: v4 架构 + ALS 学 E (零随机梯度) ===')
for rnd in range(1, 4):
    t0 = time.time()
    ATA = torch.zeros(NF, NF, device=dev)
    ATY = torch.zeros(NF, VOC, device=dev)
    for s in range(0, NTR, BLK):
        Xb, _ = encode(Wtr, s, min(s+BLK, NTR))
        ATA += Xb.T@Xb
        ATY += Xb.T@Ytr[s:s+BLK]
    W = torch.linalg.solve(ATA/NTR + lam*torch.eye(NF, device=dev), ATY/NTR)
    a1, a3 = eval_model(W)
    print('轮%d W-解 (%.0fs): top1 %.3f top3 %.3f' % (rnd, time.time()-t0, a1, a3))
    open('/workspace/lm_result.txt', 'a').write('RESULT-v6 rnd%d top1 %.4f top3 %.4f\n' % (rnd, a1, a3))
    # ---- e-解 (固定 W) ----
    t1 = time.time()
    W_hdc = W[HC:, :]                                    # (D,VOC)
    G = W_hdc@W_hdc.T                                    # (D,D)
    PGP = torch.stack([PW[i][:, None]*G*PW[i][None, :] for i in range(K)])  # (K,D,D)
    R = torch.zeros((VOC+1)*K, VOC, device=dev)
    for s in range(0, NTR, BLK):
        Xb, Wi = encode(Wtr, s, min(s+BLK, NTR))
        rb = Ytr[s:s+BLK] - Xb@W
        for i in range(K):
            R.index_add_(0, Wi[:, i]*K + i, rb)
    UR = (R@W_hdc.T).reshape(VOC+1, K, D)                # (C,K,D)
    rhs = (PW[None, :, :]*UR).sum(1)                     # (C,D)
    Eb = E.clone()
    B = 250
    for s in range(0, VOC+1, B):
        Hs = torch.einsum('ck,kij->cij', NCI[s:s+B], PGP) + lam_e*I_D[None]
        Eb[s:s+B] = torch.linalg.solve(Hs, rhs[s:s+B, :, None])[..., 0]
    E += ETA*(Eb - E)
    print('  e-解 (%.0fs) 完成 (η=%.1f λe=%.0f)' % (time.time()-t1, ETA, lam_e))

a1, a3 = eval_model(W)
print('\nv6 最终: top1 %.3f top3 %.3f  (对照 v4 随机E 0.230/0.369, trigram 0.232)' % (a1, a3))
open('/workspace/lm_result.txt', 'a').write('RESULT-v6 final top1 %.4f top3 %.4f\n' % (a1, a3))
torch.save({'W': W.cpu(), 'E': E.cpu()}, '/workspace/als2_lm.pt')
