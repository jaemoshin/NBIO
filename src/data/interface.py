"""Elliptic interface problems with prescribed jumps (Stage A of the head-to-head extension).

  div(a grad u) = 0 in each region (a = a_minus inside Gamma1, a_plus in the matrix),
  outer Dirichlet u = g on Gamma0,  interface jumps  [u] = u_plus - u_minus = alpha,
  [a u_n] = a_plus u_plus_n - a_minus u_minus_n = beta  on Gamma1.

Unlike the homogeneous single-layer transmission solver (which forces [u]=0), this uses the DIRECT
(Calderon) boundary integral formulation with Cauchy data (u, u_n) as unknowns, coupling the two regions
through the jump conditions. It needs only the single layer S (weakly singular -> Kress) and the double
layer D (bounded, curvature diagonal) -- no hypersingular operator. GT for validation is a manufactured
piecewise-harmonic solution (separate exterior/interior charge sets), exact in both regions.

Calderon identity used (G = -1/2pi ln|x-y|, -Delta G = delta, outward normal n, x on Gamma):
    1/2 u(x) + D[u](x) - S[u_n](x) = 0,   D[u]=int dG/dn_y u ds,  S[q]=int G q ds.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .bie import boundary, sample_coeffs, sample_charges, eval_u, _double_layer_kernel, radius_at
from .transmission import single_layer, build_curves, sample_inclusions


# ----------------------------- operators -----------------------------
def double_layer(bdA: Dict, bdB: Dict, selfblock: bool, normal_sign: float = 1.0) -> np.ndarray:
    """D[i,j] = dG/dn_y(x_i^A, y_j^B) * w_j, with source normal = normal_sign * bdB.normal.
    selfblock -> diagonal curvature limit -curv/(4pi) (times normal_sign)."""
    XA = bdA["X"]; XB = bdB["X"]; NB = normal_sign * bdB["normal"]; wB = bdB["w"]
    K, _ = _double_layer_kernel(XA, XB, NB)
    if selfblock:
        np.fill_diagonal(K, normal_sign * (-bdB["curv"] / (4 * np.pi)))
    return K * wB[None, :]


def eval_grad_n(points, normals, charges):
    """Normal derivative of the manufactured potential u = sum q (-1/2pi) log|x-c| at (points, normals)."""
    c = charges["c"]; q = charges["q"]
    diff = points[:, None, :] - c[None, :, :]                  # (P,M,2)
    r2 = (diff ** 2).sum(-1)
    dot = (diff * normals[:, None, :]).sum(-1)                 # (x-c).n
    return (-1.0 / (2 * np.pi)) * ((dot / r2) @ q)


# ----------------------------- direct interface solver (one inclusion) -----------------------------
def interface_solve(c_out, incl, a_plus, a_minus, g, alpha, beta, nbo, nbi):
    """Direct BIE for one interface. Returns (region evaluator).
    c_out: outer coeffs; incl=(center, rad, coeffs); g on Gamma0; alpha=[u], beta=[a u_n] on Gamma1."""
    curves = build_curves(c_out, [incl], nbo, nbi)
    G0, G1 = curves[0], curves[1]
    n0, n1 = nbo, nbi
    # unknowns: q0 (Gamma0, n0), uP (u_plus on Gamma1, n1), qP (u_plus_n on Gamma1, n1)
    N = n0 + 2 * n1
    A = np.zeros((N, N)); rhs = np.zeros(N)
    o_q0, o_uP, o_qP = 0, n0, n0 + n1

    # ---- matrix region Omega+ : boundary Gamma0 (normal +n0) and Gamma1 (normal -n1) ----
    # Calderon collocated on Gamma0 (rows 0:n0):
    #   1/2 u0 + [D_{0<-0}[u0] + D_{0<-1}^-[uP]] - [S_{0<-0}[q0] + S_{0<-1}[uP_n]] = 0,  uP_n = -qP
    D00 = double_layer(G0, G0, selfblock=True)
    D01 = double_layer(G0, G1, selfblock=False, normal_sign=-1.0)     # Gamma1 normal = -n1 for Omega+
    S00 = single_layer(G0, G0, selfblock=True)
    S01 = single_layer(G0, G1, selfblock=False)
    # unknown q0 (with -S00), uP (with +D01), qP (with +S01 since u_n=-qP -> -S01[-qP]=+S01[qP])
    A[0:n0, o_q0:o_q0 + n0] = -S00
    A[0:n0, o_uP:o_uP + n1] = D01
    A[0:n0, o_qP:o_qP + n1] = S01
    rhs[0:n0] = -(0.5 * g + D00 @ g)                                 # u0 = g known -> move to RHS

    # Calderon collocated on Gamma1 from Omega+ (rows n0:n0+n1):
    #   1/2 uP + [D_{1<-0}[u0] + D_{1<-1}^-[uP]] - [S_{1<-0}[q0] + S_{1<-1}[uP_n=-qP]] = 0
    D10 = double_layer(G1, G0, selfblock=False)
    D11m = double_layer(G1, G1, selfblock=True, normal_sign=-1.0)
    S10 = single_layer(G1, G0, selfblock=False)
    S11 = single_layer(G1, G1, selfblock=True)
    A[n0:n0 + n1, o_q0:o_q0 + n0] = -S10
    A[n0:n0 + n1, o_uP:o_uP + n1] = 0.5 * np.eye(n1) + D11m
    A[n0:n0 + n1, o_qP:o_qP + n1] = S11
    rhs[n0:n0 + n1] = -(D10 @ g)

    # ---- inclusion region Omega- : boundary Gamma1 (normal +n1) ----
    # 1/2 uM + D_{1<-1}^+[uM] - S_{1<-1}[uM_n] = 0,  uM = uP - alpha,  uM_n = qM = (a+ qP - beta)/a-
    D11p = double_layer(G1, G1, selfblock=True, normal_sign=+1.0)
    # substitute uM = uP - alpha, qM = (a+/a-) qP - beta/a-
    A[n0 + n1:N, o_uP:o_uP + n1] = 0.5 * np.eye(n1) + D11p
    A[n0 + n1:N, o_qP:o_qP + n1] = -S11 * (a_plus / a_minus)
    rhs[n0 + n1:N] = (0.5 * alpha + D11p @ alpha) - S11 @ (beta / a_minus)

    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    q0 = sol[o_q0:o_q0 + n0]; uP = sol[o_uP:o_uP + n1]; qP = sol[o_qP:o_qP + n1]
    uM = uP - alpha; qM = (a_plus * qP - beta) / a_minus

    def evaluate(Xq, region):  # region in {"plus","minus"}: representation u(x)=S[u_n]-D[u]
        if region == "minus":   # inside D-, boundary Gamma1 normal +n1
            k, _ = _double_layer_kernel(Xq, G1["X"], G1["normal"])
            dG = -(k * G1["w"][None, :]) @ uM
            sg = -(1.0 / (2 * np.pi)) * (np.log(np.linalg.norm(Xq[:, None, :] - G1["X"][None, :, :], axis=2)
                                                .clip(1e-14)) * G1["w"][None, :]) @ qM
            return sg + dG
        # matrix: boundary Gamma0 (n0) + Gamma1 (-n1)
        u = np.zeros(Xq.shape[0])
        for bd, uu, qq, nsign in [(G0, g, q0, 1.0), (G1, uP, -qP, -1.0)]:
            k, _ = _double_layer_kernel(Xq, bd["X"], nsign * bd["normal"])
            u += -(k * bd["w"][None, :]) @ uu
            u += -(1.0 / (2 * np.pi)) * (np.log(np.linalg.norm(Xq[:, None, :] - bd["X"][None, :, :], axis=2)
                                                .clip(1e-14)) * bd["w"][None, :]) @ qq
        return u
    return evaluate, curves
