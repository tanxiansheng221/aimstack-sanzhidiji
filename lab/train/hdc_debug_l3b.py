# -*- coding: utf-8 -*-
# 快测: 三层容器在奇校验-3 (较易) 上是否可学 —— 区分结构墙 vs 难度墙
import numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(113)

def freqs_fd(Fd, Dl, r_):
    return r_.uniform(-1, 1, (Dl, len(Fd))) * np.array(Fd)

def acc(yh, yt): return float(np.mean(np.sign(yh) == np.sign(yt)))

nb = rng.integers(0, 2, (600, 3)); s = 2.0*nb-1
X3 = s + 0.1*rng.normal(0,1,(600,3)); y3 = np.prod(s,1)
nb = rng.integers(0, 2, (500, 3)); s3t = 2.0*nb-1
X3t = s3t + 0.1*rng.normal(0,1,(500,3)); y3t = np.prod(s3t,1)

def phi_feat(h, f, Dl):
    ph = np.exp(2j*np.pi*h@f.T)
    return ph, np.concatenate([ph.real, -ph.imag],1)/np.sqrt(Dl)

def run(hidden, Ds, Do, steps, lr, mb=200, tag=""):
    r2 = np.random.default_rng(7)
    layers = []; dprev = 3
    for H, Dl in zip(hidden, Ds):
        f = freqs_fd([2.0]*dprev, Dl, r2)
        W = r2.normal(0, 1, (H, 2*Dl))/np.sqrt(2*Dl)
        layers.append([f, Dl, W]); dprev = H
    fo = freqs_fd([2.0]*dprev, Do, r2)
    vo = r2.normal(0, 0.3, 2*Do)
    def fwd(X):
        caches = []; h = X
        for f, Dl, W in layers:
            ph, A = phi_feat(h, f, Dl)
            h = np.tanh(A@W.T); caches.append((ph, A, h))
        pho, Ao = phi_feat(h, fo, len(fo)); caches.append((pho, Ao, h))
        return Ao@vo, caches
    def bwd(y, out, caches):
        n=len(y); g=2*(out-y)/n
        pho, Ao, h_out = caches[-1]
        g_v = Ao.T@g; Do=len(fo)
        wc = vo[:Do]+1j*vo[Do:]
        dLdh = g[:,None]*(-(2*np.pi/np.sqrt(Do)))*np.imag(pho@(wc[:,None]*fo))
        gW = [None]*len(layers)
        for li in range(len(layers)-1, -1, -1):
            f, Dl, W = layers[li]
            ph, A, h = caches[li]
            dLdu = dLdh*(1-h**2)
            gW[li] = dLdu.T@A
            if li > 0:
                wc_l = W[:,:Dl]+1j*W[:,Dl:]
                T = np.einsum('nd,hd,dk->nhk', ph, wc_l, f)
                dLdh = np.einsum('ni,nik->nk', dLdu, -(2*np.pi/np.sqrt(Dl))*np.imag(T))
        return gW, g_v
    ps = [W for _,_,W in layers]+[vo]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    best, best_at, curve = 0.0, 0, []
    for step in range(1, steps+1):
        idx = r2.choice(600, mb, replace=False)
        out, C = fwd(X3[idx])
        gW, g_v = bwd(y3[idx], out, C)
        gs = gW+[g_v]
        for i in range(len(ps)):
            gg = gs[i]
            m[i] = 0.9*m[i]+0.1*gg; v[i] = 0.999*v[i]+0.001*gg*gg
            ps[i] = ps[i]-lr*(m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8)
        for li in range(len(layers)): layers[li][2] = ps[li]
        vo = ps[-1]
        if step % 100 == 0:
            a = acc(fwd(X3t)[0], y3t); curve.append(a)
            if a > best: best, best_at = a, step
    print("  %-24s best test %.3f @%d  末段 %s" % (tag, best, best_at, np.array2string(np.array(curve[-6:]), precision=2)))

run([8,8],[32,32],48,2500,0.02,tag="3L 奇校验-3")
run([10],[40],56,2500,0.02,tag="2L 奇校验-3 对照")
