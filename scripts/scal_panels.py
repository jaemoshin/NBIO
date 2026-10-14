#!/usr/bin/env python
"""Method 5: adaptive panel refinement proof-of-concept (additive; modifies nothing).

Panel Nystrom for the interior Laplace double-layer 2nd-kind BIE. Because the double-layer kernel is BOUNDED
(curvature diagonal limit), a composite Gauss-Legendre rule over panels needs NO singular quadrature. We
compare UNIFORM panels vs CURVATURE-ADAPTIVE panels (breakpoints equidistributed w.r.t. an arc-length x
|curvature| monitor) on a high-aspect / high-curvature star geometry, and report interior rel-L2 (vs an
EXACT manufactured harmonic solution) as a function of total node count. Adaptive should reach a target
accuracy with fewer nodes -> the basis for sub-O(n) effective cost on geometries with localized features.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from numpy.polynomial.legendre import leggauss

from src.data.bie import sample_charges, eval_u


# ---- analytic curve from Fourier radius r(theta) (evaluate at ARBITRARY theta, unlike bie.boundary) ----
def curve_at(theta, coeffs):
    r0, a, b = coeffs["r0"], coeffs["a"], coeffs["b"]
    K = len(a); k = np.arange(1, K + 1)[:, None]
    C, S = np.cos(k * theta[None, :]), np.sin(k * theta[None, :])
    r = r0 * (1 + a @ C + b @ S)
    rp = r0 * ((-a * k[:, 0]) @ S + (b * k[:, 0]) @ C)
    rpp = r0 * ((-a * k[:, 0] ** 2) @ C + (-b * k[:, 0] ** 2) @ S)
    ct, st = np.cos(theta), np.sin(theta)
    X = np.stack([r * ct, r * st], 1)
    xp = rp * ct - r * st; yp = rp * st + r * ct
    speed = np.hypot(xp, yp)
    normal = np.stack([yp, -xp], 1) / speed[:, None]
    curv = (r ** 2 + 2 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
    return X, normal, speed, curv


def panel_nodes(breaks, q):
    """Composite Gauss-Legendre nodes/weights in theta over panels defined by `breaks` (len P+1, 0..2pi)."""
    gl_x, gl_w = leggauss(q)
    th, wth = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        th.append(mid + half * gl_x); wth.append(half * gl_w)
    return np.concatenate(th), np.concatenate(wth)


def solve_eval(coeffs, breaks, q, ch, Xq):
    th, wth = panel_nodes(breaks, q)
    X, N, speed, curv = curve_at(th, coeffs)
    w = wth * speed                                  # arc-length quadrature weights
    g = eval_u(X, ch)
    diff = X[None, :, :] - X[:, None, :]             # (i target, j source) y - x
    r2 = (diff ** 2).sum(-1)
    num = (diff * N[None, :, :]).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (-1.0 / (2 * np.pi)) * num / r2
    np.fill_diagonal(K, -curv / (4 * np.pi))          # double-layer diagonal limit (bounded)
    A = -0.5 * np.eye(len(th)) + K * w[None, :]
    mu = np.linalg.solve(A, g)
    d = X[None, :, :] - Xq[:, None, :]
    r2q = (d ** 2).sum(-1); numq = (d * N[None, :, :]).sum(-1)
    P = (-1.0 / (2 * np.pi)) * numq / r2q
    u = (P * w[None, :]) @ mu
    return u


def uniform_breaks(P):
    return np.linspace(0, 2 * np.pi, P + 1)


def adaptive_breaks(coeffs, P, oversample=4000):
    """Equidistribute panel breakpoints w.r.t. monitor m(theta) = arc_speed * (1 + alpha|curv|)."""
    th = np.linspace(0, 2 * np.pi, oversample, endpoint=False)
    _, _, speed, curv = curve_at(th, coeffs)
    m = speed * (1.0 + 3.0 * np.abs(curv))
    cum = np.concatenate([[0], np.cumsum(0.5 * (m[1:] + m[:-1]) * (2 * np.pi / oversample))])
    cum = np.concatenate([cum, [cum[-1] + m[-1] * (2 * np.pi / oversample)]])[:oversample + 1]
    targets = np.linspace(0, cum[-1], P + 1)
    thg = np.concatenate([th, [2 * np.pi]])
    return np.interp(targets, cum, thg)


def main():
    rng = np.random.default_rng(0)
    # high-curvature geometry: a star with a sharp lobe
    coeffs = {"r0": 1.0, "a": np.array([0.0, 0.0, 0.0, 0.0, 0.45]), "b": np.zeros(5)}
    ch = sample_charges(rng, "interior", coeffs, 6)
    # interior queries (well inside)
    ang = rng.uniform(0, 2 * np.pi, 600); rad = np.sqrt(rng.uniform(0, 1, 600)) * 0.45
    Xq = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    ue = eval_u(Xq, ch)
    q = 10                                           # GL nodes per panel
    rows = []
    for P in (8, 12, 16, 24, 32, 48, 64):
        n = P * q
        u_uni = solve_eval(coeffs, uniform_breaks(P), q, ch, Xq)
        u_ada = solve_eval(coeffs, adaptive_breaks(coeffs, P), q, ch, Xq)
        e_uni = float(np.linalg.norm(u_uni - ue) / np.linalg.norm(ue))
        e_ada = float(np.linalg.norm(u_ada - ue) / np.linalg.norm(ue))
        rows.append({"panels": P, "n_nodes": n, "uniform_rel_l2": e_uni, "adaptive_rel_l2": e_ada,
                     "adaptive_speedup": (e_uni / e_ada if e_ada > 0 else float("inf"))})
        print(f"  P={P:3d} (n={n:4d}): uniform {e_uni:.2e} | adaptive {e_ada:.2e}  ({e_uni/max(e_ada,1e-16):.1f}x)")
    os.makedirs("results", exist_ok=True)
    json.dump({"geometry": "5-fold star, sharp lobe (a5=0.45)", "gl_per_panel": q, "rows": rows},
              open("results/scal_panels.json", "w"), indent=2)
    print("[scal_panels] DONE -> results/scal_panels.json")


if __name__ == "__main__":
    main()
