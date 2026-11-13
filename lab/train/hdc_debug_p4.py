# -*- coding: utf-8 -*-
# 快测: 奇校验-4 上单层核 vs 两层容器是否真分离
import numpy as np, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(91)

def freqs2(F, d, D_=4096):
    f = rng.uniform(-F, F, size=(D_, d))
    return f[np.any(np.abs(f) > 0.05, axis=1)]

def acc(yh, yt): return float(np.mean(np.sign(yh) == np.sign(yt)))

nb = rng.integers(0, 2, (600, 4)); s = 2.0*nb-1
Xtr = s + 0.1*rng.normal(0,1,(600,4)); ytr = np.prod(s,1)
nb = rng.integers(0, 2, (500, 4)); s_te = 2.0*nb-1
Xte = s_te + 0.1*rng.normal(0,1,(500,4)); yte = np.prod(s_te,1)

def kernel_acc(Dk, F=2.0):
    fk = freqs2(F, 4, Dk)
    def A_(X):
        ang = 2*np.pi*X@fk.T
        return np.concatenate([np.cos(ang), -np.sin(ang)],1)/np.sqrt(Dk)
    Ath = A_(Xtr); n=len(Xtr)
    K = Ath@Ath.T/n
    al = np.linalg.solve(K+1e-3*np.eye(n), ytr/n)
    th = Ath.T@al
    return acc(A_(Xte)@th, yte)

for Dk in (128, 512, 1664):
    print("kernel D=%d: %.3f" % (Dk, kernel_acc(Dk)))

# 同预算 (256 实参) 两层容器: H=4, D1=24, D2=32 → 4*48+64 = 256
D1, H, D2 = 24, 4, 32
f1 = freqs2(2.0, 4, D1); f2 = freqs2(1.5, H, D2)
def feat1(X):
    ang = 2*np.pi*X@f1.T
    return np.concatenate([np.cos(ang), -np.sin(ang)],1)/np.sqrt(D1)
A1 = feat1(Xtr); A1te = feat1(Xte)

def fwd_loss(W1_, v2_, XA, y_):
    hh = np.tanh(XA@W1_.T)
    P2 = np.exp(2j*np.pi*hh@f2.T)
    out = (np.real(P2)@v2_[:D2] - np.imag(P2)@v2_[D2:])/D2
    return float(np.mean((out-y_)**2)), out, hh, P2

def backprop(W1_, v2_, XA, y_):
    L, out, hh, P2 = fwd_loss(W1_, v2_, XA, y_)
    n=len(y_)
    g = 2*(out-y_)/n
    g_v2 = np.concatenate([np.real(P2).T@g, -np.imag(P2).T@g])/D2
    w2c = v2_[:D2]+1j*v2_[D2:]
    dodh = -(2*np.pi/D2)*np.imag(P2@(w2c[:,None]*f2))
    dLdu = (g[:,None]*dodh)*(1-hh**2)
    return L, dLdu.T@XA, g_v2

def adam(p,g,m,v,t,lr):
    m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
    return p-lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8), m, v

for seed in (1, 2, 3):
    r2 = np.random.default_rng(seed)
    W1 = r2.normal(0,1,(H,2*D1))/np.sqrt(2*D1)
    v2 = r2.normal(0,0.1,2*D2)
    mW=np.zeros_like(W1); vW=np.zeros_like(W1); m2=np.zeros_like(v2); vv2=np.zeros_like(v2)
    for step in range(1,4001):
        idx = r2.choice(600, 200, replace=False)
        _,gW,gv = backprop(W1,v2,A1[idx],ytr[idx])
        W1,mW,vW = adam(W1,gW,mW,vW,step,0.02)
        v2,m2,vv2 = adam(v2,gv,m2,vv2,step,0.02)
    print("2-layer(256 实参, seed %d): test %.3f" % (seed, acc(fwd_loss(W1,v2,A1te,yte)[1], yte)))
