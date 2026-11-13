# -*- coding: utf-8 -*-
# 快测: 三层容器在奇校验-4 上的优化配方扫描
import numpy as np, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(113)

def freqs_fd(Fd, Dl, r_):
    return r_.uniform(-1, 1, (Dl, len(Fd))) * np.array(Fd)

def acc(yh, yt): return float(np.mean(np.sign(yh) == np.sign(yt)))

nb = rng.integers(0, 2, (600, 4)); s = 2.0*nb-1
X4 = s + 0.1*rng.normal(0,1,(600,4)); y4 = np.prod(s,1)
nb = rng.integers(0, 2, (500, 4)); s4t = 2.0*nb-1
X4t = s4t + 0.1*rng.normal(0,1,(500,4)); y4t = np.prod(s4t,1)

def phi_feat(h, f, Dl):
    ph = np.exp(2j*np.pi*h@f.T)
    return ph, np.concatenate([ph.real, -ph.imag],1)/np.sqrt(Dl)

def make_net(hidden, Ds, Do, d_in, r_, w_scale=1.0, vo_scale=0.1, skip=False):
    layers = []; dprev = d_in
    for H, Dl in zip(hidden, Ds):
        f = freqs_fd([2.0]*dprev, Dl, r_)
        W = r_.normal(0, w_scale, (H, 2*Dl))/np.sqrt(2*Dl)
        layers.append([f, Dl, W]); dprev = H
    fo = freqs_fd([2.0]*(dprev*(2 if skip else 1)), Do, r_)
    vo = r_.normal(0, vo_scale, 2*Do)
    return layers, fo, vo

def fwd(h0, layers, fo, vo, res=False, skip=False):
    (f1, D1, W1), (f2, D2, W2) = layers
    ph1, A1 = phi_feat(h0, f1, D1); u1 = A1@W1.T; h1 = np.tanh(u1)
    h1r = h1
    ph2, A2 = phi_feat(h1r, f2, D2); u2 = A2@W2.T; h2 = np.tanh(u2)
    h2r = h2 + h1 if res else h2
    hin = np.concatenate([h2r, h1], 1) if skip else h2r
    pho, Ao = phi_feat(hin, fo, len(fo))
    return Ao@vo, (ph1,A1,u1,h1,h1r,ph2,A2,u2,h2,h2r,pho,Ao,hin)

def bwd(y, out, C, layers, fo, vo, res=False, skip=False):
    ph1,A1,u1,h1,h1r,ph2,A2,u2,h2,h2r,pho,Ao,hin = C
    (f1,D1,W1),(f2,D2,W2) = layers
    n=len(y); g=2*(out-y)/n
    Do=len(fo); wco=vo[:Do]+1j*vo[Do:]
    g_v = Ao.T@g
    dLdhin = g[:,None]*(-(2*np.pi/np.sqrt(Do)))*np.imag(pho@(wco[:,None]*fo))
    if skip:
        dLdh2 = dLdhin[:, :h2.shape[1]]; dLdh1d = dLdhin[:, h2.shape[1]:]
    else:
        dLdh2 = dLdhin; dLdh1d = 0.0
    dLdu2 = dLdh2*(1-np.tanh(u2)**2)
    gW2 = dLdu2.T@A2
    wc2 = W2[:,:D2]+1j*W2[:,D2:]
    T2 = np.einsum('nd,hd,dk->nhk', ph2, wc2, f2)
    dLdh1 = np.einsum('ni,nik->nk', dLdu2, -(2*np.pi/np.sqrt(D2))*np.imag(T2))
    if res: dLdh1 = dLdh1 + dLdh2
    dLdh1 = dLdh1 + dLdh1d
    dLdu1 = dLdh1*(1-np.tanh(u1)**2)
    gW1 = dLdu1.T@A1
    return [gW1, gW2], g_v

def adam(p,g,m,v,t,lr):
    m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
    return p-lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8), m, v

def run(hidden, Ds, Do, steps, lr, w_scale, vo_scale, mb=200, tag="", res=False, skip=False):
    r2 = np.random.default_rng(7)
    layers, fo, vo = make_net(hidden, Ds, Do, 4, r2, w_scale, vo_scale, skip)
    ps = [W for _,_,W in layers]+[vo]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]
    best, best_at = 0.0, 0
    curve = []
    for step in range(1, steps+1):
        idx = r2.choice(600, mb, replace=False) if mb < 600 else np.arange(600)
        out, C = fwd(X4[idx], layers, fo, vo, res, skip)
        gW, g_v = bwd(y4[idx], out, C, layers, fo, vo, res, skip)
        gs = gW+[g_v]
        for i in range(len(ps)):
            ps[i], m[i], v[i] = adam(ps[i], gs[i], m[i], v[i], step, lr)
        for li in range(len(layers)): layers[li][2] = ps[li]
        vo = ps[-1]
        if step % 100 == 0:
            ote, _ = fwd(X4t, layers, fo, vo, res, skip)
            a = acc(ote, y4t); curve.append(a)
            if a > best: best, best_at = a, step
    print("  %-30s best test %.3f @%d  末段 %s" % (tag, best, best_at, np.array2string(np.array(curve[-6:]), precision=2)))

run([8,8],[32,32],48,2000,0.02,1.0,0.3,tag="K: skip 输出层")
run([8,8],[32,32],48,2000,0.02,1.0,0.3,tag="L: 残差", res=True)
run([8,8],[32,32],48,2000,0.02,1.0,0.3,tag="M: skip+残差", res=True, skip=True)
run([8,8],[32,32],48,2000,0.02,1.0,0.3,mb=600,tag="N: skip+残差 全批", res=True, skip=True)
run([8,8],[32,32],48,2000,0.02,1.0,0.3,tag="O: 对照(无skip无res)")
