#!/usr/bin/env python
"""Validate the torch multi-curve single-layer QBX against the numpy reference + analytic concentric disk."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import eval_u
from src.data.transmission import build_curves, transmission_solve, analytic_disk
from src.numerics.qbx import multi_single_layer_qbx
from src.numerics.qbx_torch import single_layer_qbx_multi
from src.utils.common import configure_torch, get_device


def circle(R, k=5):
    return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}


def direct_eval(curves, dens, Xq):
    u = np.zeros(Xq.shape[0])
    for c, ph in zip(curves, dens):
        d = np.abs((Xq[:, 0] + 1j * Xq[:, 1])[:, None] - (c["X"][:, 0] + 1j * c["X"][:, 1])[None, :]).clip(1e-14)
        u += (-1.0 / (2 * np.pi)) * (np.log(d) * c["w"][None, :]) @ ph
    return u


def main():
    configure_torch(); dev = get_device(True)
    R0, R, a0, ain = 2.0, 0.6, 1.0, 5.0
    curves = build_curves(circle(R0), [(np.zeros(2), R, circle(R))], 384, 384)
    g = curves[0]["X"][:, 0]
    dens, _ = transmission_solve(curves, [a0, ain], g)
    Xs_list = [torch.tensor(c["X"], dtype=torch.float32, device=dev)[None] for c in curves]
    Ns_list = [torch.tensor(c["normal"], dtype=torch.float32, device=dev)[None] for c in curves]
    phi_list = [torch.tensor(d, dtype=torch.float32, device=dev)[None] for d in dens]
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    print(f"{'d':>8} {'direct':>11} {'numpy-QBX':>11} {'torch-QBX':>11}  (rel-L2 vs ANALYTIC)")
    for d in (0.1, 0.03, 0.01, 0.003, 0.001):
        Xq = np.concatenate([np.stack([(R + s * d) * np.cos(th), (R + s * d) * np.sin(th)], 1) for s in (-1, 1)])
        ua = analytic_disk(R0, R, a0, ain, Xq)
        ud = direct_eval(curves, dens, Xq)
        unp = multi_single_layer_qbx(Xq, [c["X"] for c in curves], dens,
                                     curves_N=[c["normal"] for c in curves], order=12, oversample=4)
        Xqt = torch.tensor(Xq, dtype=torch.float32, device=dev)[None]
        ut = single_layer_qbx_multi(Xqt, Xs_list, phi_list, Ns_list, order=12, oversample=4)[0].cpu().numpy()
        f = lambda u: np.linalg.norm(u - ua) / np.linalg.norm(ua)
        print(f"{d:>8.3f} {f(ud):>11.2e} {f(unp):>11.2e} {f(ut):>11.2e}")
    print("\n[validate_qbx_tm_torch] DONE")


if __name__ == "__main__":
    main()
