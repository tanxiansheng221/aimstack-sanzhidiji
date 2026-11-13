# -*- coding: utf-8 -*-
# §23-v14 运行: 双塔精确块 ALS (E_ctx/E_rep 独立, 块下降保证训练损失单调)
# 用法: python3 hdc_v14_run.py [数据] [轮数] [λe] [测试对上限]
from hdc_v13_lib import *
import time

NR = int(sys.argv[2]) if len(sys.argv) > 2 else 3
lam_e = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
TE_CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 150000
lam = 1e-2
print('轮数=%d λe=%.0f (双塔精确块下降)' % (NR, lam_e))
I_D = torch.eye(D, device=dev)
N = p_tr
BLK = 20000

def enc_hb(idds):
    h = torch.zeros(idds.shape[0], D, device=dev)
    for i in range(K1):
        h += E_ctx[idds[:, i]]*BW1[i]
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
print('=== §23-v14: 双塔精确块 ALS (零 SGD) ===')
for rnd in range(1, NR+1):
    t0 = time.time()
    HH = torch.zeros(D, D, device=dev)
    HG = torch.zeros(D, D, device=dev)
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        h = enc_hb(CTX_T[s:e].long()); g = enc_gb(REP_T[s:e].long())
        HH += h.T@h; HG += h.T@g
    M = torch.linalg.solve(HH/N + lam*I_D, HG/N)
    del HH, HG
    a1r, a10r = eval_score(M, cand_rand)
    a1h, a10h = eval_score(M, cand_hard)
    print('轮%d M-解 (%.0fs): rand %.3f/%.3f | hard %.3f/%.3f'
          % (rnd, time.time()-t0, a1r, a10r, a1h, a10h))
    open('/workspace/lm_result.txt', 'a').write(
        'RESULT-v14 rnd%d rand %.4f/%.4f hard %.4f/%.4f\n' % (rnd, a1r, a10r, a1h, a10h))
    if a1h > best[0]:
        best = (a1h, a10h, rnd)
    # ---- ctx 块 (精确牛顿: b = s1⊙E + Σ_i w_i·scatter(R@Mᵀ), R = g−hM) ----
    t1 = time.time()
    Gmat = M@M.T
    bctx = s1[:, None]*E_ctx
    Lsum = 0.0
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        h = enc_hb(CTX_T[s:e].long()); g = enc_gb(REP_T[s:e].long())
        R = g - h@M
        Lsum += float((R**2).sum())
        RM = R@M.T
        for i in range(K1):
            bctx.index_add_(0, CTX_T[s:e, i].long(), RM*BW1[i])
    Eb = E_ctx.clone()
    for s in range(0, VOC+2, 502):
        Hs = s1[s:s+502, None, None]*(Gmat[None] + lam_e*I_D[None]) + 1e-6*I_D[None]
        Eb[s:s+502] = torch.linalg.solve(Hs, bctx[s:s+502, :, None])[..., 0]
    zr = s1 == 0
    Eb[zr] = E_ctx[zr]
    Eb[UNK] = E_ctx[UNK]; Eb[PAD] = 0
    E_ctx = Eb
    print('  ctx-块 (%.0fs) trainL/N=%.4f' % (time.time()-t1, Lsum/N))
    # ---- rep 块 (对角: f = (s2⊙F − Σ_j scatter(R))/(s2+λe), R 用新 E_ctx) ----
    t2 = time.time()
    brep = s2[:, None]*E_rep
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        h = enc_hb(CTX_T[s:e].long()); g = enc_gb(REP_T[s:e].long())
        R = g - h@M
        for j in range(K2):
            brep.index_add_(0, REP_T[s:e, j].long(), -R)
    f_unk = E_rep[UNK].clone()
    E_rep = brep/(s2[:, None]+lam_e)
    E_rep[UNK] = f_unk
    E_rep[PAD] = 0
    # 重编码测试塔
    for s in range(0, P, 20000):
        e = min(s+20000, P)
        Hte[s:e] = enc_hb(te_ctx_t[s:e].long()); Gte[s:e] = enc_gb(te_rep_t[s:e].long())
    print('  rep-块 (%.0fs) 完成' % (time.time()-t2))

a1r, a10r = eval_score(M, cand_rand)
a1h, a10h = eval_score(M, cand_hard)
print('\nv14 末轮: rand %.3f/%.3f hard %.3f/%.3f' % (a1r, a10r, a1h, a10h))
print('v14 hard 峰值: R@1 %.3f R@10 %.3f @轮%d' % best)
open('/workspace/lm_result.txt', 'a').write(
    'RESULT-v14 final hard-peak R1 %.4f R10 %.4f @rnd%d\n' % best)
torch.save({'M': M.cpu(), 'E_ctx': E_ctx.cpu(), 'E_rep': E_rep.cpu()}, '/workspace/als8_lm.pt')
