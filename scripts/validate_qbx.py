#!/usr/bin/env python
"""Validate QBX close-evaluation. Manufactured harmonic u* (exterior charges) is represented exactly as a
single-layer SL[phi]=u* (solve S phi = u*|Gamma). Evaluate SL[phi] at interior targets at controlled
distance d from Gamma via (a) direct trapezoidal quadrature and (b) QBX; compare to exact u*. Direct must
blow up as d->0; QBX must stay accurate. Tested on circle and star geometries."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.bie import boundary, sample_coeffs, sample_charges, eval_u
from src.data.transmission import single_layer
from src.numerics.qbx import single_layer_qbx, _arc_weights


def direct_sl(Xq, Xb, phi):
    wb = _arc_weights(Xb)
    d = np.abs((Xq[:, 0] + 1j * Xq[:, 1])[:, None] - (Xb[:, 0] + 1j * Xb[:, 1])[None, :]).clip(1e-14)
    return (-1.0 / (2 * np.pi)) * (np.log(d) * wb[None, :]) @ phi


def run(coeffs, label, n_b=256, seed=0):
    rng = np.random.default_rng(seed)
    bd = boundary(coeffs, n_b)
    ch = sample_charges(rng, "interior", coeffs, 6)          # charges outside -> u* harmonic inside
    g = eval_u(bd["X"], ch)
    S = single_layer(bd, bd, selfblock=True)
    phi = np.linalg.solve(S, g)                               # SL density: SL[phi] = u* inside
    print(f"\n=== {label} (n_b={n_b}) ===  [d = distance inward from Gamma]")
    print(f"{'d':>8} {'direct rel-L2':>14} {'QBX rel-L2':>12}")
    for d in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        Xt = bd["X"] - d * bd["normal"]                       # push inward (interior)
        ue = eval_u(Xt, ch)                                   # exact
        ud = direct_sl(Xt, bd["X"], phi)
        uq = single_layer_qbx(Xt, bd["X"], phi, Nb=bd["normal"], order=12, r_fac=4.0, oversample=4)
        rd = np.linalg.norm(ud - ue) / np.linalg.norm(ue)
        rq = np.linalg.norm(uq - ue) / np.linalg.norm(ue)
        print(f"{d:>8.3f} {rd:>14.2e} {rq:>12.2e}")


def main():
    run({"r0": 1.5, "a": np.zeros(5), "b": np.zeros(5)}, "circle R=1.5")
    rng = np.random.default_rng(1)
    run(sample_coeffs(rng, 5), "star (random)", seed=2)
    print("\n[validate_qbx] DONE")


if __name__ == "__main__":
    main()
