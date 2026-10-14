#!/usr/bin/env python
"""Validate the direct interface-jump BIE solver against a manufactured piecewise-harmonic solution
(separate interior/exterior charge sets -> exact u_minus inside, u_plus in the matrix, with nonzero
[u] and [a u_n] jumps). Tested on circular and star interfaces, several contrasts."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.bie import boundary, sample_coeffs, eval_u, radius_at
from src.data.interface import interface_solve, eval_grad_n


def circle(R, k=5):
    return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}


def charges_at(rng, n, rlo, rhi):
    ang = rng.uniform(0, 2 * np.pi, n); rad = rng.uniform(rlo, rhi, n)
    q = rng.normal(size=n); q -= q.mean()
    return {"c": np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), "q": q}


def interior_pts(rng, rmax, n):
    ang = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0, 1, n)) * rmax
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)


def annulus_pts(rng, rlo, rhi, n):
    rhi = max(rhi, rlo * 1.25)
    ang = rng.uniform(0, 2 * np.pi, n); rad = rng.uniform(rlo, rhi, n)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)


def run(c_out, incl, label, a_plus, a_minus, nbo=256, nbi=256, seed=0):
    rng = np.random.default_rng(seed)
    center, rad, c_in = incl
    ch_minus = charges_at(rng, 6, rad * 1.6, rad * 2.4 + 0.3)   # outside D- -> u_minus harmonic inside D-
    ch_plus = charges_at(rng, 6, rad * 0.15, rad * 0.55)        # inside D-  -> u_plus harmonic in matrix
    G1 = boundary(c_in, nbi); G1 = {**G1, "X": G1["X"] + center[None, :]}
    G0 = boundary(c_out, nbo)
    uPb = eval_u(G1["X"], ch_plus); uMb = eval_u(G1["X"], ch_minus)
    alpha = uPb - uMb
    beta = a_plus * eval_grad_n(G1["X"], G1["normal"], ch_plus) - a_minus * eval_grad_n(G1["X"], G1["normal"], ch_minus)
    g = eval_u(G0["X"], ch_plus)
    evaluate, _ = interface_solve(c_out, incl, a_plus, a_minus, g, alpha, beta, nbo, nbi)
    # queries
    Xm = center + interior_pts(rng, rad * 0.6, 400)
    rmin_out = float(radius_at(c_out, np.linspace(0, 2 * np.pi, 128)).min())
    Xp = annulus_pts(rng, rad * 1.4, 0.9 * rmin_out, 400)
    em = np.linalg.norm(evaluate(Xm, "minus") - eval_u(Xm, ch_minus)) / np.linalg.norm(eval_u(Xm, ch_minus))
    ep = np.linalg.norm(evaluate(Xp, "plus") - eval_u(Xp, ch_plus)) / np.linalg.norm(eval_u(Xp, ch_plus))
    print(f"  {label:28s} a+/a-={a_plus}/{a_minus}:  u_minus rel {em:.2e} | u_plus rel {ep:.2e}")


def main():
    print("=== direct interface solver vs manufactured piecewise-harmonic (nonzero jumps) ===")
    for (ap, am) in [(1.0, 5.0), (1.0, 0.2), (1.0, 10.0)]:
        run(circle(2.0), (np.zeros(2), 0.6, circle(0.6)), "circle interface", ap, am)
    rng = np.random.default_rng(3)
    co = sample_coeffs(rng, 4, amp=0.18, r0=1.6)               # larger, smoother outer domain
    run(co, (np.array([0.05, -0.03]), 0.30, sample_coeffs(rng, 3, amp=0.12, r0=0.30)), "star interface", 1.0, 6.0, seed=5)
    print("[validate_interface] DONE")


if __name__ == "__main__":
    main()
