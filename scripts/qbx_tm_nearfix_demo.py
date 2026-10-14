#!/usr/bin/env python
"""Near-touching fix INSIDE the differentiable transmission operator. NBIO-tm structure on two near-touching
circles, three configs: direct (fails) -> +QBX eval -> +QBX eval & oversampled cross-block assembly.
Referenced to a high-res numpy BEM+QBX solve. Plus a differentiability check through the full QBX+oversample
path. (NBIO-tm structure = split=False = the classical multi-region solver in torch -> mirrors BEM+QBX.)"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import sample_charges, eval_u
from src.data.transmission import build_curves, transmission_solve
from src.numerics.qbx import multi_single_layer_qbx
from src.models import build_model
from src.utils.common import configure_torch, get_device


def circle(R, k=5):
    return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}


def two_circle_curves(gap, R0=2.0, r=0.45, nbo=128, nbi=128):
    cen = r + gap / 2
    incls = [(np.array([-cen, 0.0]), r, circle(r)), (np.array([cen, 0.0]), r, circle(r))]
    return build_curves(circle(R0), incls, nbo, nbi)


def gap_queries(gap, r=0.45, n=60):
    x = np.linspace(-0.7 * gap / 2, 0.7 * gap / 2, n); y = np.linspace(-0.3 * r, 0.3 * r, 5)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.ravel(), Y.ravel()], 1)


def batch_from(curves, a_in, g, Xq, device):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    return {"Xb_out": t(curves[0]["X"])[None], "Nb_out": t(curves[0]["normal"])[None],
            "wb_out": t(curves[0]["w"])[None], "curv_out": t(curves[0]["curv"])[None],
            "Xb_in": t(np.stack([c["X"] for c in curves[1:]]))[None],
            "Nb_in": t(np.stack([c["normal"] for c in curves[1:]]))[None],
            "wb_in": t(np.stack([c["w"] for c in curves[1:]]))[None],
            "curv_in": t(np.stack([c["curv"] for c in curves[1:]]))[None],
            "a_in": t(a_in)[None], "g": t(g)[None], "Xq": t(Xq)[None]}


def main():
    configure_torch(); device = get_device(True)
    model = build_model({"model": {"name": "nbio_tm", "kwargs": {"split": False}}}).to(device).eval()
    print(f"{'gap':>7} {'direct':>11} {'QBX(ov1)':>11} {'QBX+ov4':>11} {'QBX+adapt':>11}  (gap-region rel-L2 vs BEM@1024+QBX)")
    for gap in (0.10, 0.05, 0.02, 0.01, 0.005):
        rng = np.random.default_rng(0)
        ch = sample_charges(rng, "interior", circle(2.0), 6)
        Xq = gap_queries(gap)
        # high-res numpy reference
        cref = two_circle_curves(gap, nbo=1024, nbi=1024); gref = eval_u(cref[0]["X"], ch)
        dref, _ = transmission_solve(cref, [1.0, 5.0, 5.0], gref)
        uref = multi_single_layer_qbx(Xq, [c["X"] for c in cref], dref,
                                      curves_N=[c["normal"] for c in cref], order=12, oversample=4)
        # operator at n_b=128
        curves = two_circle_curves(gap, nbo=128, nbi=128); g = eval_u(curves[0]["X"], ch)
        b = batch_from(curves, [5.0, 5.0], g, Xq, device)
        res = {}
        for tag, qbx, ov in [("direct", False, 1), ("QBX(ov1)", True, 1), ("QBX+ov4", True, 4), ("QBX+adapt", True, "adaptive")]:
            model.qbx = qbx; model.qbx_kw = dict(order=12, r_fac=4.0, near_fac=2.0); model.assembly_oversample = ov
            with torch.no_grad():
                u = model(b)[0].cpu().numpy()
            res[tag] = np.linalg.norm(u - uref) / np.linalg.norm(uref)
        print(f"{gap:>7.3f} {res['direct']:>11.2e} {res['QBX(ov1)']:>11.2e} {res['QBX+ov4']:>11.2e} {res['QBX+adapt']:>11.2e}")

    # differentiability through QBX + oversampled assembly (split=True so there are params)
    m = build_model({"model": {"name": "nbio_tm", "kwargs": {"split": True, "hidden": 32, "n_layers": 2}}}).to(device)
    m.qbx = True; m.qbx_kw = dict(order=10, r_fac=4.0, near_fac=2.0); m.assembly_oversample = 4; m.train()
    curves = two_circle_curves(0.05, nbo=128, nbi=128); ch = sample_charges(np.random.default_rng(1), "interior", circle(2.0), 6)
    g = eval_u(curves[0]["X"], ch); Xq = gap_queries(0.05); uref_t = torch.tensor(
        multi_single_layer_qbx(Xq, [c["X"] for c in curves], transmission_solve(curves, [1.0, 5.0, 5.0], g)[0],
                               curves_N=[c["normal"] for c in curves]), dtype=torch.float32, device=device)
    b = batch_from(curves, [5.0, 5.0], g, Xq, device)
    loss = ((m(b)[0] - uref_t) ** 2).mean(); loss.backward()
    gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None and p.requires_grad)
    print(f"\n[diff] QBX+ov4 backward grad-norm {gn:.3e} finite={np.isfinite(gn)}")
    assert gn > 0 and np.isfinite(gn)
    print("[qbx_tm_nearfix_demo] DONE")


if __name__ == "__main__":
    main()
