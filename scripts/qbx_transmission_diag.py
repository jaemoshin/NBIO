#!/usr/bin/env python
"""QBX for transmission close-evaluation. (A) concentric disk (ANALYTIC GT): evaluate near the interface
with direct vs QBX. (B) two near-touching circles: decompose the gap-region error into evaluation
(BEM@128 + QBX) vs assembly (BEM@512 + QBX) vs the failing baseline (BEM@128 + direct), referenced to
BEM@1024 + QBX."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.bie import boundary, sample_charges, eval_u
from src.data.transmission import build_curves, transmission_solve, analytic_disk
from src.numerics.qbx import multi_single_layer_qbx


def circle(R, k=5):
    return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}


def qbx_eval(curves, dens, Xq):
    return multi_single_layer_qbx(Xq, [c["X"] for c in curves], dens,
                                  curves_N=[c["normal"] for c in curves], order=12, r_fac=4.0, oversample=4)


def direct_eval(curves, dens, Xq):
    u = np.zeros(Xq.shape[0])
    for c, ph in zip(curves, dens):
        wb = c["w"]; d = np.abs((Xq[:, 0] + 1j * Xq[:, 1])[:, None]
                                - (c["X"][:, 0] + 1j * c["X"][:, 1])[None, :]).clip(1e-14)
        u += (-1.0 / (2 * np.pi)) * (np.log(d) * wb[None, :]) @ ph
    return u


def partA():
    print("=== (A) concentric disk transmission, evaluate near interface r=R (ANALYTIC GT) ===")
    R0, R, a0, ain = 2.0, 0.6, 1.0, 5.0
    curves = build_curves(circle(R0), [(np.zeros(2), R, circle(R))], 384, 384)
    g = curves[0]["X"][:, 0]                                 # Dirichlet u=x (=R0 cos th) -> matches analytic_disk
    dens, _ = transmission_solve(curves, [a0, ain], g)
    print(f"{'d(=|r-R|)':>10} {'direct':>12} {'QBX':>12}")
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    for d in (0.1, 0.03, 0.01, 0.003, 0.001):
        pts = []
        for s in (-1, 1):                                    # both sides of the interface
            r = R + s * d
            pts.append(np.stack([r * np.cos(th), r * np.sin(th)], 1))
        Xq = np.concatenate(pts)
        ua = analytic_disk(R0, R, a0, ain, Xq)
        ud = direct_eval(curves, dens, Xq); uq = qbx_eval(curves, dens, Xq)
        rd = np.linalg.norm(ud - ua) / np.linalg.norm(ua); rq = np.linalg.norm(uq - ua) / np.linalg.norm(ua)
        print(f"{d:>10.3f} {rd:>12.2e} {rq:>12.2e}")


def two_circle(gap, R0=2.0, r=0.45, nbo=128, nbi=128, contrast=5.0, seed=0):
    rng = np.random.default_rng(seed)
    cen = r + gap / 2
    incls = [(np.array([-cen, 0.0]), r, circle(r)), (np.array([cen, 0.0]), r, circle(r))]
    curves = build_curves(circle(R0), incls, nbo, nbi)
    ch = sample_charges(rng, "interior", circle(R0), 6)
    g = eval_u(curves[0]["X"], ch)
    dens, _ = transmission_solve(curves, [1.0, contrast, contrast], g)
    return curves, dens


def gap_queries(gap, r=0.45, n=60):
    x = np.linspace(-0.7 * gap / 2, 0.7 * gap / 2, n)
    y = np.linspace(-0.3 * r, 0.3 * r, 5)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.ravel(), Y.ravel()], 1)


def partB():
    print("\n=== (B) two near-touching circles: decompose gap error (ref = BEM@1024 + QBX) ===")
    print(f"{'gap':>7} {'BEM128+direct':>14} {'BEM128+QBX':>12} {'BEM512+QBX':>12}")
    for gap in (0.1, 0.05, 0.02):
        Xq = gap_queries(gap)
        cref, dref = two_circle(gap, nbo=1024, nbi=1024); uref = qbx_eval(cref, dref, Xq)
        c128, d128 = two_circle(gap, nbo=128, nbi=128)
        c512, d512 = two_circle(gap, nbo=512, nbi=512)
        e_dir = np.linalg.norm(direct_eval(c128, d128, Xq) - uref) / np.linalg.norm(uref)
        e_q128 = np.linalg.norm(qbx_eval(c128, d128, Xq) - uref) / np.linalg.norm(uref)
        e_q512 = np.linalg.norm(qbx_eval(c512, d512, Xq) - uref) / np.linalg.norm(uref)
        print(f"{gap:>7.3f} {e_dir:>14.2e} {e_q128:>12.2e} {e_q512:>12.2e}")


def main():
    partA(); partB(); print("\n[qbx_transmission_diag] DONE")


if __name__ == "__main__":
    main()
