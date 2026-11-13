# -*- coding: utf-8 -*-
# §23-v15 运行: 层次塔双塔精确块 ALS (扁平64词元袋 + 8句x24词元话语袋, 零 SGD)
# 用法: python3 hdc_v15_run.py [数据] [轮数] [λrel] [测试对上限]
from hdc_v15_lib import *
import time

NR = int(sys.argv[2]) if len(sys.argv) > 2 else 3
lam_e = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
TE_CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 150000
lam = 1e-2
print('轮数=%d λrel=%.2f (层次塔+双塔精确块下降)' % (NR, lam_e))
I_D = torch.eye(D, device=dev)
N = p_tr
BLK = 20000

def enc_hb(idf, idu):
    h = torch.zeros(idf.shape[0], D, device=dev)
    for i in range(K1):
        h += E_ctx[idf[:, i]]*BW1[i]
    for k in range(KU):
        idk = idu[:, k]
        for t in range(TU):
            h += E_ctx[idk[:, t]]*AH[k]
    return h

def enc_gb(idds):
    g = torch.zeros(idds.shape[0], D, device=dev)
    for j in range(K2):
        g += E_rep[idds[:, j]]
    return g

print('词面重叠 rand: R@1 %.3f R@10 %.3f' % lexical(cand_rand.cpu().numpy()))
print('词面重叠 hard: R@1 %.3f R@10 %.3f' % lexical(cand_hard.cpu().numpy()))

M = torch.eye(D, device=dev)
best = (0.0, 0.0, 0)
print('=== §23-v15: 层次塔 ALS (零 SGD) ===')
for rnd in range(1, NR+1):
    t0 = time.time()
    HH = torch.zeros(D, D, device=dev)
    HG = torch.zeros(D, D, device=dev)
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        h = enc_hb(CTX_T[s:e].long(), CTX_U[s:e].long()); g = enc_gb(REP_T[s:e].long())
        HH += h.T@h; HG += h.T@g
    M = torch.linalg.solve(HH/N + lam*I_D, HG/N)
    del HH, HG
    a1r, a10r = eval_score(M, cand_rand)
    a1h, a10h = eval_score(M, cand_hard)
    print('轮%d M-解 (%.0fs): rand %.3f/%.3f | hard %.3f/%.3f'
          % (rnd, time.time()-t0, a1r, a10r, a1h, a10h))
    open('/workspace/lm_result.txt', 'a').write(
        'RESULT-v15 rnd%d rand %.4f/%.4f hard %.4f/%.4f\n' % (rnd, a1r, a10r, a1h, a10h))
    if a1h > best[0]:
        best = (a1h, a10h, rnd)
    # ---- ctx 块 (精确牛顿: 双特征组散射 + 锚定 + 阻尼) ----
    t1 = time.time()
    Gmat = M@M.T
    bs_ctx = torch.zeros(VOC+2, D, device=dev)
    Lsum = 0.0
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        idf = CTX_T[s:e].long(); idu = CTX_U[s:e].long(); idr = REP_T[s:e].long()
        h = enc_hb(idf, idu); g = enc_gb(idr)
        R = g - h@M
        Lsum += float((R**2).sum())
        RM = R@M.T
        for i in range(K1):
            bs_ctx.index_add_(0, idf[:, i], RM*BW1[i])
        for k in range(KU):
            idk = idu[:, k]
            for t in range(TU):
                bs_ctx.index_add_(0, idk[:, t], RM*AH[k])
    # rhs = RM散射 + s1*G*e0 (锚定项: 二次项系数是 s1*G 而非 s1*I)
    rhs = bs_ctx + (E_ctx@Gmat)*s1[:, None]
    lam_rel = 0.01*float(Gmat.trace())/D                  # 自适应: λrel << λ(Gmat)
    Eb = E_ctx.clone()
    for s in range(0, VOC+2, 502):
        Hs = s1[s:s+502, None, None]*(Gmat[None] + lam_rel*I_D[None]) + 1e-6*I_D[None]
        Eb[s:s+502] = torch.linalg.solve(Hs, rhs[s:s+502, :, None])[..., 0]
    zr = s1 == 0
    Eb[zr] = E_ctx[zr]
    Eb[UNK] = E_ctx[UNK]; Eb[PAD] = 0
    E_ctx = E_ctx + 0.25*(Eb - E_ctx)                     # Jacobi 阻尼 0.25 (实测单点)
    print('  ctx-块 (%.0fs) trainL/N=%.4f' % (time.time()-t1, Lsum/N))
    # ---- rep 块 (对角: R 用新 E_ctx) ----
    t2 = time.time()
    brep = s2[:, None]*E_rep
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        h = enc_hb(CTX_T[s:e].long(), CTX_U[s:e].long()); g = enc_gb(REP_T[s:e].long())
        R = g - h@M
        for j in range(K2):
            brep.index_add_(0, REP_T[s:e, j].long(), -R)
    f_unk = E_rep[UNK].clone()
    upd = brep/(s2[:, None]+1.0)                          # rep: 小绝对正则
    msk = s2 > 0
    E_rep[msk] = E_rep[msk] + 0.25*(upd[msk] - E_rep[msk])  # Jacobi 阻尼 0.25
    E_rep[UNK] = f_unk
    E_rep[PAD] = 0
    # 重编码测试塔
    for s in range(0, P, 20000):
        e = min(s+20000, P)
        Hte[s:e] = enc_h(te_ctx_t[s:e], te_ctu_t[s:e])
        Gte[s:e] = enc_g(te_rep_t[s:e].long())
    print('  rep-块 (%.0fs) 完成' % (time.time()-t2))

a1r, a10r = eval_score(M, cand_rand)
a1h, a10h = eval_score(M, cand_hard)
print('\nv15 末轮: rand %.3f/%.3f hard %.3f/%.3f' % (a1r, a10r, a1h, a10h))
print('v15 hard 峰值: R@1 %.3f R@10 %.3f @轮%d' % best)
open('/workspace/lm_result.txt', 'a').write(
    'RESULT-v15 final hard-peak R1 %.4f R10 %.4f @rnd%d\n' % best)
torch.save({'M': M.cpu(), 'E_ctx': E_ctx.cpu(), 'E_rep': E_rep.cpu()}, '/workspace/als9_lm.pt')
