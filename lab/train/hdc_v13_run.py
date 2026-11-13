# -*- coding: utf-8 -*-
# §23-v13 运行: 300 万对话流式 ALS (双遍 Gram+散射, H/G 不常驻)
# 用法: python3 hdc_v13_run.py [数据] [轮数] [λe] [η] [衰减] [测试对上限]
from hdc_v13_lib import *
import time

NR = int(sys.argv[2]) if len(sys.argv) > 2 else 4
lam_e = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
ETA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
DEC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
lam = 1e-2
print('轮数=%d λe=%.0f η=%.1f 衰减=%.2f' % (NR, lam_e, ETA, DEC))
I_D = torch.eye(D, device=dev)
N = p_tr
BLK = 20000

def enc_hb(idds):
    h = torch.zeros(idds.shape[0], D, device=dev)
    for i in range(K1):
        h += E[idds[:, i]]*BW1[i]
    return h

def enc_gb(idds):
    g = torch.zeros(idds.shape[0], D, device=dev)
    for j in range(K2):
        g += E[idds[:, j]]
    return g

print('词面重叠 rand: R@1 %.3f R@10 %.3f' % lexical(cand_rand.cpu().numpy()))
print('词面重叠 hard: R@1 %.3f R@10 %.3f' % lexical(cand_hard.cpu().numpy()))

M = torch.eye(D, device=dev)
eta = ETA
best = (0.0, 0.0, 0)
print('=== §23-v13: 300 万对话流式 ALS (零 SGD) ===')
for rnd in range(1, NR+1):
    t0 = time.time()
    HH = torch.zeros(D, D, device=dev)
    HG = torch.zeros(D, D, device=dev)
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        idc = CTX_T[s:e].long(); idr = REP_T[s:e].long()
        h = enc_hb(idc); g = enc_gb(idr)
        HH += h.T@h
        HG += h.T@g
    M = torch.linalg.solve(HH/N + lam*I_D, HG/N)
    a1r, a10r = eval_score(M, cand_rand)
    a1h, a10h = eval_score(M, cand_hard)
    print('轮%d M-解 (%.0fs): rand %.3f/%.3f | hard %.3f/%.3f'
          % (rnd, time.time()-t0, a1r, a10r, a1h, a10h))
    open('/workspace/lm_result.txt', 'a').write(
        'RESULT-v13 rnd%d rand %.4f/%.4f hard %.4f/%.4f\n' % (rnd, a1r, a10r, a1h, a10h))
    if a1h > best[0]:
        best = (a1h, a10h, rnd)
    t1 = time.time()
    Gmat = M@M.T
    rhs = torch.zeros(VOC+2, D, device=dev)
    Lsum = 0.0
    for s in range(0, N, BLK):
        e = min(s+BLK, N)
        idc = CTX_T[s:e].long(); idr = REP_T[s:e].long()
        h = enc_hb(idc); g = enc_gb(idr)
        Lsum += float(((h@M - g)**2).sum())
        MG = g@M.T
        for i in range(K1):
            rhs.index_add_(0, idc[:, i], MG*BW1[i])
        GH = h@M
        for j in range(K2):
            rhs.index_add_(0, idr[:, j], g - GH)
    Eb = E.clone()
    for s in range(0, VOC+2, 502):
        Hs = s1[s:s+502, None, None]*Gmat[None] + (s2[s:s+502]+lam_e)[:, None, None]*I_D[None]
        Eb[s:s+502] = torch.linalg.solve(Hs, rhs[s:s+502, :, None])[..., 0]
    Eb[UNK] = E[UNK]                                       # 冻结 UNK
    E += eta*(Eb - E)                                      # eta=1.0 = 精确块下降
    E[PAD] = 0
    # 重编码测试塔
    for s in range(0, P, 20000):
        e = min(s+20000, P)
        idc = te_ctx_t[s:e].long(); idr = te_rep_t[s:e].long()
        Hte[s:e] = enc_hb(idc); Gte[s:e] = enc_gb(idr)
    print('  e-解 (%.0fs) η=%.2f trainL/N=%.4f' % (time.time()-t1, eta, Lsum/N))
    eta *= DEC

a1r, a10r = eval_score(M, cand_rand)
a1h, a10h = eval_score(M, cand_hard)
print('\nv13 末轮: rand %.3f/%.3f hard %.3f/%.3f' % (a1r, a10r, a1h, a10h))
print('v13 hard 峰值: R@1 %.3f R@10 %.3f @轮%d' % best)
open('/workspace/lm_result.txt', 'a').write(
    'RESULT-v13 final hard-peak R1 %.4f R10 %.4f @rnd%d\n' % best)
torch.save({'M': M.cpu(), 'E': E.cpu()}, '/workspace/als7_lm.pt')
