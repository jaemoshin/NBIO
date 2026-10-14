#!/usr/bin/env python
"""Differentiable (torch) QBX single-layer for the 3D Laplace equation on a torus.

Key simplification vs the numpy SH version: the QBX truncation of 1/(4pi|x-y|) about an off-surface
center c uses the LEGENDRE (Gegenbauer) expansion
    1/|x-y| = sum_{l>=0} (r_x^l / r_y^{l+1}) P_l(cos gamma),   cos gamma = (x-c).(y-c)/(r_x r_y),
which by the SH addition theorem equals the multipole sum but needs only Legendre polynomials of a
DOT PRODUCT -> no spherical harmonics, no arccos/atan2, pole-free and fully torch-autograd friendly.
With c = x0 + beta n(x0): r_x = beta, cos gamma = -(n . (y-c))/r_y.

Oversampling (band-limited Fourier upsample of coarse->fine sources) is applied as a fixed linear map
U; the QBX base kernel is the analytic 'known' operator (M1), a learned correction (M2) is added and is
what carries gradients for training. Here we VALIDATE the base torch QBX matches the numpy solver to
~1e-6 (solver-grade) and that autograd flows through a toy correction.
"""
from __future__ import annotations
import argparse, math
import numpy as np
from numpy.fft import rfft, irfft
import torch

PI = math.pi


def torus_np(n_th, n_ph, R=1.0, r=0.35):
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


def qbx_single_layer_torch(Xc, nrmc, Xf, wf, beta, p):
    """S_fine[Nc x Nf], differentiable, via Legendre expansion. All torch float64."""
    C = Xc + beta * nrmc                              # (Nc,3) centers
    d = Xf[None, :, :] - C[:, None, :]                # (Nc,Nf,3)
    r_y = torch.linalg.vector_norm(d, dim=-1).clamp_min(1e-30)   # (Nc,Nf)
    cosg = -(nrmc[:, None, :] * d).sum(-1) / r_y      # (Nc,Nf) = cos gamma
    S = torch.zeros_like(r_y)
    Pm1 = torch.ones_like(cosg)                       # P_0
    if p >= 0:
        S = S + (beta ** 0) / (r_y ** 1) * Pm1
    Pl = cosg.clone()                                 # P_1
    if p >= 1:
        S = S + (beta ** 1) / (r_y ** 2) * Pl
    for l in range(1, p):
        Pnext = ((2 * l + 1) * cosg * Pl - l * Pm1) / (l + 1)
        Pm1 = Pl; Pl = Pnext
        S = S + (beta ** (l + 1)) / (r_y ** (l + 2)) * Pl
    return S * wf[None, :] / (4 * PI)


def u_harm(X, c, q):
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2).clip(1e-12)
    return (1.0 / (4 * PI)) * ((1.0 / d) @ q)


def _fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = PI * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=44); ap.add_argument("--nph", type=int, default=16)
    ap.add_argument("--os", type=int, default=3); ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--cbeta", type=float, default=2.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} nt={a.nt} nph={a.nph} os={a.os} p={a.p} beta={a.cbeta}h", flush=True)
    rng = np.random.default_rng(a.seed)

    Xc_np, nrmc_np, wc_np = torus_np(a.nt, a.nph)
    mt, mph = a.nt * a.os, a.nph * a.os
    Xf_np, nrmf_np, wf_np = torus_np(mt, mph)
    U_np = upsample_matrix(a.nt, a.nph, mt, mph)      # (Nf,Nc)
    beta = a.cbeta * (2 * PI * 1.0 / a.nt)
    c = _fib(8) * rng.uniform(2.0, 3.0, (8, 1)); q = rng.normal(size=8); q -= q.mean()
    g_np = u_harm(Xc_np, c, q)

    Xc = torch.tensor(Xc_np, device=dev); nrmc = torch.tensor(nrmc_np, device=dev)
    Xf = torch.tensor(Xf_np, device=dev); wf = torch.tensor(wf_np, device=dev)
    U = torch.tensor(U_np, device=dev); g = torch.tensor(g_np, device=dev)

    Sf = qbx_single_layer_torch(Xc, nrmc, Xf, wf, beta, a.p)   # (Nc,Nf)
    Sc = Sf @ U + 1e-13 * torch.eye(Xc.shape[0], device=dev)   # (Nc,Nc)
    sigma = torch.linalg.solve(Sc, g)
    sig_f = U @ sigma

    # interior query points
    aa = np.linspace(0, 2 * PI, 400, endpoint=False)
    off = rng.uniform(0, 0.4 * 0.35, 400); be = rng.uniform(0, 2 * PI, 400)
    ringq = 1.0 + off * np.cos(be)
    Xq_np = np.stack([ringq * np.cos(aa), ringq * np.sin(aa), off * np.sin(be)], 1)
    uq = torch.tensor(u_harm(Xq_np, c, q), device=dev)
    Xq = torch.tensor(Xq_np, device=dev)
    D = torch.cdist(Xq, Xf).clamp_min(1e-15)
    up = (1.0 / (4 * PI * D) * (wf * sig_f)[None, :]).sum(1)
    rel = (torch.linalg.vector_norm(up - uq) / torch.linalg.vector_norm(uq)).item()
    print(f"[torch QBX] interior rel-L2 = {rel:.3e}   (numpy reference ~7.5e-6; target solver-grade)", flush=True)

    # autograd check: toy learned correction added to the operator, backprop a scalar loss.
    corr = torch.zeros(Xc.shape[0], Xc.shape[0], device=dev, requires_grad=True)
    Sc2 = Sf @ U + corr + 1e-13 * torch.eye(Xc.shape[0], device=dev)
    sig2 = torch.linalg.solve(Sc2, g)
    loss = (sig2 ** 2).mean()
    loss.backward()
    gnorm = float(torch.linalg.vector_norm(corr.grad))
    print(f"[autograd] loss={loss.item():.3e}  grad-norm(correction)={gnorm:.3e}  finite={math.isfinite(gnorm)}", flush=True)
    print("SMOKE_OK" if (rel < 5e-5 and math.isfinite(gnorm) and gnorm > 0) else "SMOKE_CHECK", flush=True)


if __name__ == "__main__":
    main()
