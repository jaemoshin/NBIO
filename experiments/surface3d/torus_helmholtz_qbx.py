#!/usr/bin/env python
"""QBX + oversampling for the 3D HELMHOLTZ single-layer on a torus (Stage-0, extends Laplace).

Kernel G_k(x,y) = e^{ik|x-y|}/(4pi|x-y|). Addition theorem about off-surface center c:
  G_k(x,y) = (ik/4pi) sum_l (2l+1) j_l(k r_x) h_l^{(1)}(k r_y) P_l(cos gamma),   r_x < r_y,
same Legendre P_l(cos gamma) structure as Laplace — only the RADIAL part changes to spherical
Bessel j_l / Hankel h_l^{(1)}=j_l+i y_l. With c=x0+beta n: r_x=beta, cos gamma = -(n.(y-c))/r_y.
Interior Dirichlet vs manufactured radiating GT u(x)=sum q_i e^{ik|x-c_i|}/(4pi|x-c_i|), c_i exterior.
numpy/complex validation; sweeps expansion order p and wavenumber k. Target solver-grade (<=1e-6).
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
from numpy.fft import rfft, irfft
from scipy.special import spherical_jn, spherical_yn

PI = math.pi


def torus(n_th, n_ph, R=1.0, r=0.35):
    th = np.arange(n_th) * 2 * PI / n_th
    ph = np.arange(n_ph) * 2 * PI / n_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij"); TH = TH.ravel(); PH = PH.ravel()
    ring = R + r * np.cos(PH)
    X = np.stack([ring * np.cos(TH), ring * np.sin(TH), r * np.sin(PH)], 1)
    nrm = np.stack([np.cos(PH) * np.cos(TH), np.cos(PH) * np.sin(TH), np.sin(PH)], 1)
    w = r * ring * (2 * PI / n_th) * (2 * PI / n_ph)
    return X, nrm, w


def upsample_1d(N, M):
    U = np.zeros((M, N))
    for j in range(N):
        e = np.zeros(N); e[j] = 1.0
        E = rfft(e); Ep = np.zeros(M // 2 + 1, dtype=complex)
        k = min(len(E), len(Ep)); Ep[:k] = E[:k]
        U[:, j] = irfft(Ep, M) * (M / N)
    return U


def upsample_matrix(nt, nph, mt, mph):
    return np.kron(upsample_1d(nt, mt), upsample_1d(nph, mph))


def legendre_all(x, p):
    """P_0..P_p(x), list of arrays."""
    P = [np.ones_like(x)]
    if p >= 1:
        P.append(x.copy())
    for l in range(1, p):
        P.append(((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1))
    return P


def qbx_helmholtz(Xc, nrmc, Xf, wf, beta, p, k):
    Nc, Nf = Xc.shape[0], Xf.shape[0]
    C = Xc + beta * nrmc
    S = np.zeros((Nc, Nf), dtype=complex)
    jl_beta = np.array([spherical_jn(l, k * beta) for l in range(p + 1)])   # radial at target
    for t in range(Nc):
        d = Xf - C[t]
        r_y = np.linalg.norm(d, axis=-1).clip(1e-30)
        cosg = -(nrmc[t][None, :] * d).sum(-1) / r_y
        P = legendre_all(cosg, p)
        acc = np.zeros(Nf, dtype=complex)
        for l in range(p + 1):
            hl = spherical_jn(l, k * r_y) + 1j * spherical_yn(l, k * r_y)    # h_l^{(1)}(k r_y)
            acc += (2 * l + 1) * jl_beta[l] * hl * P[l]
        S[t, :] = (1j * k / (4 * PI)) * acc * wf
    return S


def u_helm(X, c, q, k):
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2).clip(1e-12)
    return (np.exp(1j * k * d) / (4 * PI * d)) @ q


def _fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = PI * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def run(nt, nph, os_, p, cbeta, k, R=1.0, r=0.35, seed=0):
    rng = np.random.default_rng(seed)
    Xc, nrmc, wc = torus(nt, nph, R, r)
    mt, mph = nt * os_, nph * os_
    Xf, nrmf, wf = torus(mt, mph, R, r)
    U = upsample_matrix(nt, nph, mt, mph)
    beta = cbeta * (2 * PI * R / nt)
    c = _fib(8) * rng.uniform(2.0, 3.0, (8, 1)); q = rng.normal(size=8) + 1j * rng.normal(size=8)
    g = u_helm(Xc, c, q, k)
    Sc = qbx_helmholtz(Xc, nrmc, Xf, wf, beta, p, k) @ U + 1e-12 * np.eye(Xc.shape[0])
    sigma = np.linalg.solve(Sc, g); sig_f = U @ sigma
    aa = np.linspace(0, 2 * PI, 400, endpoint=False)
    off = rng.uniform(0, 0.4 * r, 400); be = rng.uniform(0, 2 * PI, 400)
    ringq = R + off * np.cos(be)
    Xq = np.stack([ringq * np.cos(aa), ringq * np.sin(aa), off * np.sin(be)], 1)
    uq = u_helm(Xq, c, q, k)
    D = np.sqrt(((Xq[:, None, :] - Xf[None, :, :]) ** 2).sum(-1)).clip(1e-15)
    up = (np.exp(1j * k * D) / (4 * PI * D) * (wf * sig_f)[None, :]).sum(1)
    return float(np.linalg.norm(up - uq) / np.linalg.norm(uq))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/torus_helmholtz.json"); a = ap.parse_args()
    res = {}
    for k in (1.0, 2.0, 4.0):
        for (nt, nph) in [(44, 16), (60, 22)]:
            for p in (4, 8, 12, 16):
                e = float(np.mean([run(nt, nph, 3, p, 2.0, k, seed=s) for s in range(2)]))
                res[f"k{k}_{nt}x{nph}_p{p}"] = e
                print(f"k={k} {nt}x{nph} os3 beta2h p={p:2d}  rel-L2 {e:.3e}", flush=True)
        print("", flush=True)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
    print(f"BEST Helmholtz QBX {min(res.values()):.3e}  (solver-grade <=1e-6)", flush=True)


if __name__ == "__main__":
    main()
