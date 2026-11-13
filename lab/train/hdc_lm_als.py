# -*- coding: utf-8 -*-
# §23-v5: ALS 容器 LM —— 交替最小二乘学字向量+读出头 (全程零随机梯度)
#   模型: score(w) = U_w · h,  h = Σ_i b_i·(p_i ⊙ e_{c_i})   (位置绑定加权)
#   交替: 固定 e 解 U (岭回归) → 固定 U 解 e (逐字 D×D 正规方程, GPU 批量)
#     e_c 正规方程: [Σ_i n_{c,i} b_i² (p_i⊙G⊙p_i) + λI] e_c = Σ_i b_i (p_i ⊙ (U^T R_{c,i}))
#     其中 G=U^T U, R_{c,i}=位置 i 上字 c 对应样本的残差和, n_{c,i}=出现次数
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

K, D = 8, 1024
BW = torch.tensor([0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 2.0, 4.0], device=dev)      # 位置权
PV = torch.from_numpy(nprng.normal(0, 1, (K, D)).astype(np.float32)).to(dev)     # 位置随机±1
PV = PV / PV.norm(dim=1, keepdim=True)

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
Wtr_t = torch.from_numpy(Wtr).to(dev)          # (N,8) 常驻显存 5MB
Ytr  = torch.zeros(NTR, VOC, device=dev)
Ytr[torch.arange(NTR, device=dev), torch.from_numpy(Ttr).to(dev)] = 1.0
Ytr = Ytr.to(torch.float32)

def encode_ctx(Emat, s, e):
    Wi = Wtr_t[s:e]
    ctx = torch.zeros(Wi.shape[0], D, device=dev)
    for i in range(K):
        ctx += (Emat[Wi[:, i]]*BW[i])*PV[i]
    return ctx

def eval_model(U, Emat):
    top1 = top3 = tot = 0
    Wte_t = torch.from_numpy(Wte).to(dev)
    Tb = torch.from_numpy(Tte).to(dev)
    for s in range(0, len(Tte), 50000):
        Wi = Wte_t[s:s+50000]
        ctx = torch.zeros(Wi.shape[0], D, device=dev)
        for i in range(K):
            ctx += (Emat[Wi[:, i]]*BW[i])*PV[i]
        sc = ctx@U.T
        top1 += int((sc.argmax(1) == Tb[s:s+50000]).sum())
        top3 += int((sc.topk(3, 1).indices == Tb[s:s+50000, None]).any(1).sum())
        tot += Wi.shape[0]
    return top1/tot, top3/tot

Emat = torch.from_numpy(nprng.normal(0, 1.0/np.sqrt(D), (VOC+1, D)).astype(np.float32)).to(dev)
lam = 1e-3
N = len(Ttr)

# 预聚合: 每字每位置的样本计数 n[c,i]
NCI = torch.zeros(VOC+1, K, device=dev)
for i in range(K):
    nci = torch.bincount(Wtr_t[:, i], minlength=VOC+1).to(torch.float32)
    NCI[:, i] = nci

print('=== §23-v5: ALS (交替闭式解, 零随机梯度) ===')
for rnd in range(1, 5):
    t0 = time.time()
    # ---- 解 U (固定 e): 分块 Gram ----
    ATA = torch.zeros(D, D, device=dev)
    ATY = torch.zeros(D, VOC, device=dev)
    BLK = 100000
    for s in range(0, NTR, BLK):
        Hb = encode_ctx(Emat, s, min(s+BLK, NTR))
        ATA += Hb.T@Hb
        ATY += Hb.T@Ytr[s:s+BLK]
    U = torch.linalg.solve(ATA/N + lam*torch.eye(D, device=dev), ATY/N)
    U = U.T                                             # (VOC,D) 读出头: score_w = U_w · h
    t_u = time.time()-t0
    a1, a3 = eval_model(U, Emat)
    print('轮%d U-解 (%.0fs): top1 %.3f top3 %.3f' % (rnd, t_u, a1, a3))
    if rnd == 4:
        break
    # ---- 解 e (固定 U): 残差聚合 + 批量正规方程 ----
    t1 = time.time()
    G = U.T@U                                            # (D,D)
    PGP = torch.stack([BW[i]**2 * (PV[i][:, None]*G*PV[i][None, :]) for i in range(K)])  # (K,D,D)
    # 残差 + R 聚合 (分块 scatter): R[c*K+i] = Σ_{n: c_{n,i}=c} r_n
    R = torch.zeros((VOC+1)*K, VOC, device=dev)          # ((c*K+i), VOC)
    for s in range(0, NTR, BLK):
        Hb = encode_ctx(Emat, s, min(s+BLK, NTR))
        rb = Ytr[s:s+BLK] - Hb@U.T
        Wi = Wtr_t[s:s+BLK]
        for i in range(K):
            cidx = Wi[:, i]*K + i
            R.index_add_(0, cidx, rb)
    # 右端: rhs_c = Σ_i b_i (p_i ⊙ (U^T R[c*K+i]))
    UR = (R.reshape((VOC+1)*K, VOC) @ U).reshape(VOC+1, K, D)     # (C,K,D)
    rhs = ((BW[None, :, None]*PV[None, :, :])*UR).sum(1)           # (C,D)
    # Hessian: H_c = Σ_i n_{c,i} b_i² (p_i⊙G⊙p_i) + λI → 按批 einsum (防 12.6GB 整体物化)
    Eb = Emat.clone()
    B = 250
    I_D = torch.eye(D, device=dev)
    for s in range(0, VOC+1, B):
        Hs = torch.einsum('ck,kij->cij', NCI[s:s+B], PGP) + lam*I_D[None]
        Eb[s:s+B] = torch.linalg.solve(Hs, rhs[s:s+B, :, None])[..., 0]
    Emat = Eb
    print('  e-解 (%.0fs) 完成' % (time.time()-t1))

a1, a3 = eval_model(U, Emat)
print('\nALS 最终: top1 %.3f top3 %.3f  (对照 v4 随机特征 0.230/0.369, trigram 0.232)'
      % (a1, a3))
open('/workspace/lm_result.txt', 'a').write('RESULT-ALS top1 %.4f top3 %.4f\n' % (a1, a3))
torch.save({'U': U.cpu(), 'E': Emat.cpu()}, '/workspace/als_lm.pt')
