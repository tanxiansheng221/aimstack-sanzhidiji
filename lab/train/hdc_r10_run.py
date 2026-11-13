# -*- coding: utf-8 -*-
# §23-v10 运行: n-gram 袋塔 + 对话均值排斥 (μ), ALS 学字向量, 零 SGD
# 用法: python3 hdc_r10_run.py [μ] [λe] [η] [轮数]
from hdc_r10_lib import *
import time

lam = 1e-3
mu = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
lam_e = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
ETA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 6
print('μ=%.1f λe=%.0f η=%.1f 轮数=%d' % (mu, lam_e, ETA, NR))
I_F = torch.eye(FIN, device=dev)

dlgs = sorted(set(k for a, b, k in tr))
dlg_id = {k: i for i, k in enumerate(dlgs)}
DLG = torch.tensor([dlg_id[k] for a, b, k in tr], dtype=torch.int64, device=dev)
ND = len(dlgs)
Gmean = torch.zeros(ND, FOUT, device=dev)
cnt = torch.zeros(ND, device=dev)
Gmean.index_add_(0, DLG, G)
cnt.index_add_(0, DLG, torch.ones(len(tr), device=dev))
Gmean = Gmean/cnt.clamp_min(1)[:, None]

def eval_pool(M, cand_t):
    # 池中心化评测: 候选先减去候选池均值再取 cos (与排斥目标配平, 无泄漏)
    Hn = Hte/Hte.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Gn = Gte/Gte.norm(dim=1, keepdim=True).clamp_min(1e-9)
    r1 = r10 = 0
    for s in range(0, P, 2048):
        e = min(s+2048, P)
        Gc = Gn[cand_t[s:e]]
        Gc = Gc - Gc.mean(1, keepdim=True)
        Gc = Gc/Gc.norm(dim=2, keepdim=True).clamp_min(1e-9)
        gh = (Hn[s:e]@M).clamp(-8, 8)
        gh = gh/gh.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sc = torch.einsum('bd,bcd->bc', gh, Gc)
        r1 += int((sc.argmax(1) == 0).sum())
        r10 += int((sc.topk(10, 1).indices == 0).any(1).sum())
    return r1/P, r10/P

print('词面重叠 rand: R@1 %.3f R@10 %.3f' % lexical(cand_rand.cpu().numpy()))
print('词面重叠 hard: R@1 %.3f R@10 %.3f' % lexical(cand_hard.cpu().numpy()))

M = torch.zeros(FIN, FOUT, device=dev)
print('=== §23-v10: n-gram 塔 + 排斥 (零 SGD) ===')
for rnd in range(1, NR+1):
    t0 = time.time()
    Gm = G - mu*Gmean[DLG] if mu > 0 else G
    HH = torch.zeros(FIN, FIN, device=dev)
    HG = torch.zeros(FIN, FOUT, device=dev)
    for s in range(0, len(tr), 20000):
        e = min(s+20000, len(tr))
        HH += H[s:e].T@H[s:e]
        HG += H[s:e].T@Gm[s:e]
    M = torch.linalg.solve(HH/len(tr) + lam*I_F, HG/len(tr))
    del HH, HG
    a1r, a10r = eval_score(M, cand_rand)
    a1h, a10h = eval_score(M, cand_hard)
    p1h, p10h = eval_pool(M, cand_hard)
    print('轮%d M-解 (%.0fs): rand %.3f/%.3f | hard %.3f/%.3f | pool-hard %.3f/%.3f'
          % (rnd, time.time()-t0, a1r, a10r, a1h, a10h, p1h, p10h))
    open('/workspace/lm_result.txt', 'a').write(
        'RESULT-v10 rnd%d rand %.4f/%.4f hard %.4f/%.4f pool %.4f/%.4f\n'
        % (rnd, a1r, a10r, a1h, a10h, p1h, p10h))
    t1 = time.time()
    Mc = M[:D, :]
    Gc = Mc@Mc.T
    rhs = torch.zeros(VOC+2, D, device=dev)
    for s in range(0, len(tr), 20000):
        e = min(s+20000, len(tr))
        MGc = Gm[s:e]@Mc.T
        for i in range(K1):
            rhs.index_add_(0, CTX_T[s:e, i], MGc*BW1[i])
        GHc = (H[s:e]@M)[:, :D]
        for j in range(K2):
            rhs.index_add_(0, REP_T[s:e, j], Gm[s:e, :D] - GHc)
    Eb = E.clone()
    for s in range(0, VOC+2, 502):
        Hs = s1[s:s+502, None, None]*Gc[None] + (s2[s:s+502]+lam_e)[:, None, None]*torch.eye(D, device=dev)
        Eb[s:s+502] = torch.linalg.solve(Hs, rhs[s:s+502, :, None])[..., 0]
    E += ETA*(Eb - E)
    E[PAD] = 0
    H = enc_h(CTX_T, CTX_S); G = enc_g(REP_S)
    print('  e-解 (%.0fs) 完成' % (time.time()-t1))

a1r, a10r = eval_score(M, cand_rand)
a1h, a10h = eval_score(M, cand_hard)
p1h, p10h = eval_pool(M, cand_hard)
print('\nv10 末轮: rand %.3f/%.3f hard %.3f/%.3f pool-hard %.3f/%.3f'
      % (a1r, a10r, a1h, a10h, p1h, p10h))
open('/workspace/lm_result.txt', 'a').write(
    'RESULT-v10 final rand %.4f/%.4f hard %.4f/%.4f pool %.4f/%.4f\n'
    % (a1r, a10r, a1h, a10h, p1h, p10h))
torch.save({'M': M.cpu(), 'E': E.cpu()}, '/workspace/als6_lm.pt')
