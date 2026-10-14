#!/usr/bin/env python
"""Validate the transmission BEM: (1) single-layer Laplace Dirichlet solve vs exact harmonic (tests Kress S);
(2) full transmission solver vs analytic circular inclusion (tests K*+flux coupling); (3) self-convergence
on irregular star-shaped geometry (n_b=128 vs 256)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.bie import boundary, sample_coeffs
from src.data.transmission import (single_layer, transmission_solve, build_curves, sample_inclusions,
                                   analytic_disk)


def circle_coeffs(R, k=5):
    return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}


def interior_query(rng, R0, n, rmin=0.0, rmax=0.92, avoid=None, tol=0.06):
    pts = []
    while len(pts) < n:
        ang = rng.uniform(0, 2 * np.pi, n); rad = np.sqrt(rng.uniform(0, 1, n)) * rmax * R0
        cand = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        r = np.linalg.norm(cand, axis=1)
        keep = r < rmax * R0
        if avoid is not None:
            keep &= np.abs(r - avoid) > tol
        pts.extend(cand[keep].tolist())
    return np.array(pts[:n])


def main():
    rng = np.random.default_rng(0)
    # (1) single-layer Laplace Dirichlet (no inclusion): S[phi]=g on Gamma0, eval interior vs exact harmonic u=x
    for nb in (128, 256):
        outer = boundary(circle_coeffs(1.5), nb)
        g = outer["X"][:, 0]                         # u = x  (harmonic) on the boundary
        S = single_layer(outer, outer, selfblock=True)
        phi = np.linalg.solve(S, g)
        Xq = interior_query(rng, 1.5, 400)
        d = np.linalg.norm(Xq[:, None, :] - outer["X"][None, :, :], axis=2)
        wB = (2 * np.pi / nb) * outer["speed"]
        u = ((-1.0 / (2 * np.pi)) * np.log(d) * wB[None, :]) @ phi
        rel = np.linalg.norm(u - Xq[:, 0]) / np.linalg.norm(Xq[:, 0])
        print(f"[val1] single-layer Dirichlet (Kress) n_b={nb}: rel-err vs exact harmonic = {rel:.2e}")

    # (2) transmission vs analytic concentric disk
    R0, R = 1.5, 0.5
    for (a0, ain) in [(1.0, 5.0), (1.0, 0.2), (1.0, 10.0)]:
        outer = boundary(circle_coeffs(R0), 256)
        curves = build_curves(circle_coeffs(R0), [(np.zeros(2), R, circle_coeffs(R))], 256, 256)
        g = outer["X"][:, 0]                         # Dirichlet g = x on Gamma0
        dens, ev = transmission_solve(curves, [a0, ain], g)
        Xq = interior_query(rng, R0, 600, avoid=R, tol=0.07)
        u = ev(Xq); ua = analytic_disk(R0, R, a0, ain, Xq)
        rel = np.linalg.norm(u - ua) / np.linalg.norm(ua)
        print(f"[val2] transmission disk a0={a0} ain={ain} (sigma={ain/a0:g}): rel-err vs analytic = {rel:.2e}")

    # (3) self-convergence on irregular geometry (no analytic GT) -> BEM@128 vs BEM@256
    rng = np.random.default_rng(3)
    outer_c = sample_coeffs(rng, 5)
    incls = sample_inclusions(rng, 2, outer_c)
    Xq = interior_query(rng, 1.0, 500, rmax=0.7)
    us = {}
    for nb in (128, 256):
        curves = build_curves(outer_c, incls, nb, nb)
        g = curves[0]["X"][:, 0]
        _, ev = transmission_solve(curves, [1.0, 6.0] + [6.0] * (len(curves) - 2), g)
        us[nb] = ev(Xq)
    rel = np.linalg.norm(us[256] - us[128]) / np.linalg.norm(us[256])
    print(f"[val3] irregular {len(incls)}-inclusion self-convergence |u256-u128|/|u256| = {rel:.2e}")
    print("[validate_transmission] DONE")


if __name__ == "__main__":
    main()
