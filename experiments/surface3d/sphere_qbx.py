#!/usr/bin/env python
"""M1 stage-1: QBX single-layer on star-shaped GENUS-0 surfaces (deformed spheres), solver-grade.

The torus QBX relied on double periodicity (trapezoid spectral + FFT oversampling). Genus-0 needs a
different quadrature: Gauss-Legendre in u=cos(theta) x uniform Fourier in phi. For smooth star-shaped
surfaces the integrand (with the 1/sin-theta Jacobian absorbed) is smooth in (u, phi), so GL x
trapezoid is spectral. Oversampling coarse->fine = kron(barycentric-Lagrange in u, FFT in phi) —
both spectral for functions smooth on the sphere (smooth in u along meridians; periodic in phi).

Geometry: X(theta,phi) = r(theta,phi) * s_hat, r = 1 + amp*(a cos(theta) + b sin^2 cos(2phi)
+ c sin^2 sin(2phi)); analytic partials -> exact normals/area weights.
Validation: interior Dirichlet Laplace vs manufactured harmonic GT (exterior charges). Target <=1e-6.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
from numpy.fft import rfft, irfft

PI = math.pi


# ----------------------------- geometry: star-shaped genus-0 -----------------------------
def sphere_geom(n_u, n_phi, amp=0.3, pa=0.10, pb=0.06, pc=0.04):
    u, wu = np.polynomial.legendre.leggauss(n_u)             # u = cos(theta) in (-1,1)
    phi = np.arange(n_phi) * 2 * PI / n_phi
    Ug, PHI = np.meshgrid(u, phi, indexing="ij")
    Ug = Ug.ravel(); PHI = PHI.ravel()
    st = np.sqrt(1 - Ug ** 2); ct = Ug                       # sin/cos theta (GL interior: st>0)
    cph, sph = np.cos(PHI), np.sin(PHI)
    c2, s2 = np.cos(2 * PHI), np.sin(2 * PHI)
    r = 1 + amp * (pa * ct + pb * st ** 2 * c2 + pc * st ** 2 * s2)
    # partials wrt theta, phi (theta = arccos u)
    r_th = amp * (-pa * st + 2 * st * ct * (pb * c2 + pc * s2))
    r_ph = amp * st ** 2 * (-2 * pb * s2 + 2 * pc * c2)
    s_hat = np.stack([st * cph, st * sph, ct], 1)
    s_th = np.stack([ct * cph, ct * sph, -st], 1)
    s_ph = np.stack([-st * sph, st * cph, np.zeros_like(st)], 1)
    X = r[:, None] * s_hat
    X_th = r_th[:, None] * s_hat + r[:, None] * s_th
    X_ph = r_ph[:, None] * s_hat + r[:, None] * s_ph
    cr = np.cross(X_th, X_ph)                                # outward for near-sphere
    Jfull = np.linalg.norm(cr, axis=1)                       # |X_th x X_ph| (contains sin-theta)
    nrm = cr / Jfull[:, None]
    # dA = Jfull dtheta dphi = (Jfull/st) du dphi  -> GL in u handles du spectrally
    wq = (Jfull / st) * np.repeat(wu, n_phi) * (2 * PI / n_phi)
    return X, nrm, wq


# ----------------------------- oversampling: Lagrange(u) x FFT(phi) -----------------------------
def lagrange_matrix(xf, xt):
    """Barycentric Lagrange interpolation matrix (len(xt) x len(xf)) at arbitrary nodes xf."""
    n = len(xf)
    w = np.ones(n)
    for j in range(n):
        w[j] = 1.0 / np.prod(xf[j] - np.delete(xf, j))       # fine in float64 for n <= ~40
    w = w / np.abs(w).max()                                  # normalize (over/underflow guard)
    L = np.zeros((len(xt), n))
    for i, x in enumerate(xt):
        hit = np.where(np.abs(x - xf) < 1e-14)[0]
        if len(hit):
            L[i, hit[0]] = 1.0
        else:
            t = w / (x - xf)
            L[i] = t / t.sum()
    return L


def fourier_upsample_1d(N, M):
    U = np.zeros((M, N))
    for j in range(N):
        e = np.zeros(N); e[j] = 1.0
        E = rfft(e); Ep = np.zeros(M // 2 + 1, dtype=complex)
        k = min(len(E), len(Ep)); Ep[:k] = E[:k]
        U[:, j] = irfft(Ep, M) * (M / N)
    return U


def upsample_sphere(n_u, n_phi, m_u, m_phi):
    uc, _ = np.polynomial.legendre.leggauss(n_u)
    uf, _ = np.polynomial.legendre.leggauss(m_u)
    return np.kron(lagrange_matrix(uc, uf), fourier_upsample_1d(n_phi, m_phi))


# ----------------------------- QBX (vectorized Legendre, as torch version) -----------------------------
def qbx_laplace_np(Xc, nrmc, Xf, wf, beta, p):
    C = Xc + beta * nrmc
    d = Xf[None, :, :] - C[:, None, :]
    r_y = np.linalg.norm(d, axis=-1).clip(1e-30)
    cosg = -(nrmc[:, None, :] * d).sum(-1) / r_y
    S = 1.0 / r_y
    Pm1 = np.ones_like(cosg); Pl = cosg.copy()
    S = S + beta / r_y ** 2 * Pl
    for l in range(1, p):
        Pn = ((2 * l + 1) * cosg * Pl - l * Pm1) / (l + 1)
        Pm1, Pl = Pl, Pn
        S = S + beta ** (l + 1) / r_y ** (l + 2) * Pl
    return S * wf[None, :] / (4 * PI)


def u_harm(X, c, q):
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2).clip(1e-12)
    return (1.0 / (4 * PI)) * ((1.0 / d) @ q)


def _fib(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n); th = PI * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def run(n_u, n_phi, os_, p, cbeta, amp, seed=0):
    rng = np.random.default_rng(seed)
    Xc, nrmc, wc = sphere_geom(n_u, n_phi, amp)
    m_u, m_phi = n_u * os_, n_phi * os_
    Xf, nrmf, wf = sphere_geom(m_u, m_phi, amp)
    U = upsample_sphere(n_u, n_phi, m_u, m_phi)
    beta = cbeta * (PI / n_u)                                # ~ theta spacing
    c = _fib(8) * rng.uniform(1.9, 2.8, (8, 1)); q = rng.normal(size=8); q -= q.mean()
    g = u_harm(Xc, c, q)
    Sc = qbx_laplace_np(Xc, nrmc, Xf, wf, beta, p) @ U + 1e-13 * np.eye(Xc.shape[0])
    sig = np.linalg.solve(Sc, g); sig_f = U @ sig
    Xq = _fib(300) * rng.uniform(0.05, 0.45, (300, 1))       # interior (min r >= 1-0.3-... safe)
    uq = u_harm(Xq, c, q)
    D = np.sqrt(((Xq[:, None, :] - Xf[None, :, :]) ** 2).sum(-1)).clip(1e-15)
    up = (1.0 / (4 * PI * D) * (wf * sig_f)[None, :]).sum(1)
    return float(np.linalg.norm(up - uq) / np.linalg.norm(uq))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/sphere_qbx.json"); a = ap.parse_args()
    res = {}
    for (nu, nph) in [(20, 24), (28, 32), (36, 44)]:
        for p in (6, 8, 10, 12):
            e = float(np.mean([run(nu, nph, 3, p, 2.0, 0.3, seed=s) for s in range(2)]))
            res[f"{nu}x{nph}_p{p}"] = e
            print(f"sphere {nu}x{nph} os3 beta2h p={p:2d}  rel-L2 {e:.3e}", flush=True)
        print("", flush=True)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
    print(f"BEST sphere QBX {min(res.values()):.3e}  (target <=1e-6)", flush=True)


if __name__ == "__main__":
    main()
