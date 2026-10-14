"""Stage 1: piecewise-constant conductivity (transmission / inclusion) BVPs.

  div(a grad u) = 0,  a = a0 in the matrix, a_k inside inclusion D_k (Laplace in each region),
  outer Dirichlet u=g on Gamma0,  interface conditions [u]=0 and [a du/dn]=0 on each Gamma_k.

Indirect SINGLE-LAYER representation (one density phi per curve):
    u(x) = sum_c S_{Gamma_c}[phi_c](x).
  * Outer Dirichlet (on Gamma0):  sum_c S[phi_c] = g.
  * Flux on Gamma_j (single-layer normal-derivative jump  dn S = (-/+ 1/2 I + K*) ):
        [ 1/2 (a0+a_j)/(a0-a_j) I + K*_jj ] phi_j  +  sum_{l!=j} Kn_{j<-l}[phi_l] + Kn_{j<-0}[phi_0] = 0.
The single layer is log-singular -> Kress (Martensen-Kussmaul) logarithmic quadrature on the self block
(spectral); cross-curve blocks are smooth -> trapezoidal. Adjoint double layer K* uses the curvature
diagonal -kappa/4pi. GT is this multi-region Nystrom BEM, SELF-VALIDATED against the analytic circular
inclusion (separation of variables).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .bie import boundary, sample_coeffs, radius_at


# ---------------------------- operators ----------------------------
def _kress_log_weights(n: int) -> np.ndarray:
    """R_d for the periodic integral  int ln(4 sin^2((t-tau)/2)) g dtau ~ sum_j R_{(i-j)%n} g_j  (n even).
    R_d = -(2pi/M) sum_{m=1}^{M-1} cos(m t_d)/m - (pi/M^2) cos(M t_d),  M=n/2,  t_d = 2pi d/n."""
    M = n // 2
    d = np.arange(n)
    td = 2 * np.pi * d / n
    m = np.arange(1, M)
    R = -(2 * np.pi / M) * (np.cos(np.outer(td, m)) / m).sum(axis=1) - (np.pi / M ** 2) * np.cos(M * td)
    return R   # (n,)


def single_layer(bdA: Dict, bdB: Dict, selfblock: bool) -> np.ndarray:
    """(S phi)(x_i^A) = int -1/2pi ln|x_i^A - y| phi(y) ds_y.  selfblock -> Kress; else trapezoidal."""
    XA, XB = bdA["X"], bdB["X"]
    nB = XB.shape[0]; speedB = bdB["speed"]
    if not selfblock:
        d = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=2)
        return (-1.0 / (2 * np.pi)) * np.log(d) * (2 * np.pi / nB) * speedB[None, :]
    n = nB
    R = _kress_log_weights(n)                                   # (n,)
    idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n   # (n,n) -> d
    Rmat = R[idx]                                               # circulant
    diff = XA[:, None, :] - XB[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    t = 2 * np.pi * np.arange(n) / n
    sin_half = np.abs(np.sin((t[:, None] - t[None, :]) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        M2 = np.log(r / (2 * sin_half))                         # smooth remainder
    np.fill_diagonal(M2, np.log(speedB))                        # diagonal limit ln|x'(t)|
    S = (-1.0 / (2 * np.pi)) * (0.5 * Rmat + (2 * np.pi / n) * M2) * speedB[None, :]
    return S


def adjoint_dlp(bdA: Dict, bdB: Dict, selfblock: bool) -> np.ndarray:
    """K*[i,j] = dG/dn_{x_i^A} (x_i^A, y_j^B) * w_j = (1/2pi)(y-x).n_x /|y-x|^2 * w_j.  self diag -kappa/4pi."""
    XA, NA = bdA["X"], bdA["normal"]
    XB = bdB["X"]; nB = XB.shape[0]; wB = (2 * np.pi / nB) * bdB["speed"]
    diff = XB[None, :, :] - XA[:, None, :]                      # y - x
    r2 = (diff ** 2).sum(axis=2)
    num = (diff * NA[:, None, :]).sum(axis=2)                   # (y-x).n_x
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (1.0 / (2 * np.pi)) * num / r2
    if selfblock:
        np.fill_diagonal(k, -bdA["curv"] / (4 * np.pi))
    return k * wB[None, :]


# ---------------------------- geometry ----------------------------
def sample_inclusions(rng, n_incl, outer_coeffs, n_modes_incl=3, rmin_frac=0.12, rmax_frac=0.26):
    """Non-overlapping star-shaped inclusions strictly inside the outer domain."""
    ang = np.linspace(0, 2 * np.pi, 256)
    r_out_min = float(radius_at(outer_coeffs, ang).min())
    incls = []
    tries = 0
    while len(incls) < n_incl and tries < 400:
        tries += 1
        rad = rng.uniform(rmin_frac, rmax_frac) * r_out_min
        cdist = rng.uniform(0, r_out_min - 1.6 * rad)
        cang = rng.uniform(0, 2 * np.pi)
        center = np.array([cdist * np.cos(cang), cdist * np.sin(cang)])
        c = sample_coeffs(rng, n_modes_incl, amp=0.18, r0=rad)
        ok = True
        for (ic, ir, _) in incls:
            if np.linalg.norm(center - ic) < 1.35 * (rad + ir):
                ok = False; break
        if ok and (np.linalg.norm(center) + 1.3 * rad < r_out_min):
            incls.append((center, rad, c))
    return incls


def build_curves(outer_coeffs, incls, n_b_out, n_b_in):
    curves = [boundary(outer_coeffs, n_b_out)]
    for (center, rad, c) in incls:
        bd = boundary(c, n_b_in)
        bd = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bd.items()}
        bd["X"] = bd["X"] + center[None, :]
        curves.append(bd)
    return curves   # [outer, incl_1, ...]


# ---------------------------- multi-region BEM ----------------------------
def transmission_solve(curves: List[Dict], a_list: List[float], g: np.ndarray):
    """curves[0]=outer (Dirichlet g), curves[1:]=inclusions; a_list[0]=matrix a0, a_list[k]=incl k.
    Returns densities (list per curve) and an interior evaluator."""
    a0 = a_list[0]
    sizes = [c["X"].shape[0] for c in curves]
    off = np.cumsum([0] + sizes)
    N = off[-1]
    A = np.zeros((N, N)); rhs = np.zeros(N)
    # row block 0: outer Dirichlet  sum_c S_{0<-c} phi_c = g
    for c in range(len(curves)):
        A[off[0]:off[1], off[c]:off[c + 1]] = single_layer(curves[0], curves[c], selfblock=(c == 0))
    rhs[off[0]:off[1]] = g
    # row block j>=1: flux 2nd-kind   [1/2 (a0+aj)/(a0-aj) I + K*_jj] phi_j + sum_{l!=j} Kn_{j<-l} phi_l = 0
    for j in range(1, len(curves)):
        aj = a_list[j]
        # flux jump -> Neumann-Poincare form (lam I - K*)phi = ...; here operator is (lam I + K*), so lam<0:
        lam = -0.5 * (a0 + aj) / (a0 - aj)
        for c in range(len(curves)):
            blk = adjoint_dlp(curves[j], curves[c], selfblock=(c == j))
            if c == j:
                blk = blk + lam * np.eye(sizes[j])
            A[off[j]:off[j + 1], off[c]:off[c + 1]] = blk
        # rhs is 0
    phi, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    dens = [phi[off[c]:off[c + 1]] for c in range(len(curves))]

    def evaluate(Xq: np.ndarray) -> np.ndarray:
        u = np.zeros(Xq.shape[0])
        for c in range(len(curves)):
            XB = curves[c]["X"]; nB = XB.shape[0]; wB = (2 * np.pi / nB) * curves[c]["speed"]
            d = np.linalg.norm(Xq[:, None, :] - XB[None, :, :], axis=2)
            u += ((-1.0 / (2 * np.pi)) * np.log(d) * wB[None, :]) @ dens[c]
        return u
    return dens, evaluate


# ---------------------------- analytic disk validation ----------------------------
def analytic_disk(R0, R, a0, ain, Xq, field="x"):
    """Concentric: inclusion disk radius R (a=ain) in disk radius R0 (a=a0), Dirichlet g=x (=R0 cos th)
    on Gamma0. Separation of variables: u_in=A r cos th; u_m=(B r + C/r) cos th.  field='x'."""
    # BCs: outer u_m(R0)=R0 ; continuity A R = B R + C/R ; flux ain*A = a0*(B - C/R^2)
    # solve linear system for A,B,C
    M = np.array([[0.0, R0, 1.0 / R0],
                  [R, -R, -1.0 / R],
                  [ain, -a0, a0 / R ** 2]])
    b = np.array([R0, 0.0, 0.0])
    Acoef, Bcoef, Ccoef = np.linalg.solve(M, b)
    r = np.linalg.norm(Xq, axis=1); cth = Xq[:, 0] / np.maximum(r, 1e-12)
    inside = r <= R
    u = np.where(inside, Acoef * r * cth, (Bcoef * r + Ccoef / np.maximum(r, 1e-12)) * cth)
    return u
