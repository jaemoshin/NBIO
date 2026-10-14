#!/usr/bin/env python
"""Validate the inhomogeneous (volume-source) interface solver against a manufactured solution
u = phi/a_k + u_h (piecewise-harmonic u_h), f = -Laplace(phi). Recovers u to FFT+interp accuracy."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.data.bie import boundary, sample_coeffs, eval_u, radius_at
from src.data.interface import eval_grad_n
from src.data.volume_source import gaussians, phi_and_f, inhomogeneous_interface_solve


def circle(R, k=5): return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}


def charges_at(rng, n, rlo, rhi):
    ang = rng.uniform(0, 2 * np.pi, n); rad = rng.uniform(rlo, rhi, n)
    q = rng.normal(size=n); q -= q.mean()
    return {"c": np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), "q": q}


def run(c_out, incl, label, a_plus, a_minus, nbo=256, nbi=192, seed=0):
    rng = np.random.default_rng(seed)
    center, rad, c_in = incl
    cg, Ag, sg = gaussians(rng, 5)
    f_func = lambda X: phi_and_f(X, cg, Ag, sg)[1]
    phi = lambda X: phi_and_f(X, cg, Ag, sg)[0]
    ch_m = charges_at(rng, 5, rad * 1.6, rad * 2.4 + 0.4)
    ch_p = charges_at(rng, 5, rad * 0.15, rad * 0.55)
    G1 = boundary(c_in, nbi); G1 = {**G1, "X": G1["X"] + center[None, :]}
    G0 = boundary(c_out, nbo)
    # manufactured u = phi/a_k + u_h(charges)
    uP1 = phi(G1["X"]) / a_plus + eval_u(G1["X"], ch_p)
    uM1 = phi(G1["X"]) / a_minus + eval_u(G1["X"], ch_m)
    alpha = uP1 - uM1
    # beta = [a u_n]; a*(phi/a)_n = phi_n cancels -> beta = a+ uhp_n - a- uhm_n
    gradn = lambda ch: eval_grad_n(G1["X"], G1["normal"], ch)
    beta = a_plus * gradn(ch_p) - a_minus * gradn(ch_m)
    g = phi(G0["X"]) / a_plus + eval_u(G0["X"], ch_p)
    ev = inhomogeneous_interface_solve({"r0": c_out["r0"], "a": c_out["a"], "b": c_out["b"]},
                                       (center, rad, {"r0": c_in["r0"], "a": c_in["a"], "b": c_in["b"]}),
                                       a_plus, a_minus, f_func, g, alpha, beta, nbo, nbi)
    # queries
    rmin = float(radius_at(c_out, np.linspace(0, 2 * np.pi, 128)).min())
    angm = rng.uniform(0, 2 * np.pi, 400); rm = np.sqrt(rng.uniform(0, 1, 400)) * rad * 0.55
    Xm = center + np.stack([rm * np.cos(angm), rm * np.sin(angm)], 1)
    angp = rng.uniform(0, 2 * np.pi, 400); rp = rng.uniform(rad * 1.6, 0.85 * rmin, 400)
    Xp = np.stack([rp * np.cos(angp), rp * np.sin(angp)], 1)
    uMtrue = phi(Xm) / a_minus + eval_u(Xm, ch_m); uPtrue = phi(Xp) / a_plus + eval_u(Xp, ch_p)
    em = np.linalg.norm(ev(Xm, "minus") - uMtrue) / np.linalg.norm(uMtrue)
    ep = np.linalg.norm(ev(Xp, "plus") - uPtrue) / np.linalg.norm(uPtrue)
    print(f"  {label:24s} a+/a-={a_plus}/{a_minus}:  u_minus rel {em:.2e} | u_plus rel {ep:.2e}")


def main():
    print("=== inhomogeneous (volume-source) interface solver vs manufactured (source + jumps) ===")
    for am in (5.0, 0.2, 10.0):
        run(circle(2.0), (np.zeros(2), 0.6, circle(0.6)), "circle src+jump", 1.0, am)
    rng = np.random.default_rng(3)
    run(sample_coeffs(rng, 4, amp=0.15, r0=1.6),
        (np.array([0.05, -0.03]), 0.30, sample_coeffs(rng, 3, amp=0.10, r0=0.30)), "star src+jump", 1.0, 6.0, seed=5)
    print("[validate_source] DONE")


if __name__ == "__main__":
    main()
