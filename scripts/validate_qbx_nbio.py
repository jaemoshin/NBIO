#!/usr/bin/env python
"""Differentiable-QBX NBIO: confirm it (1) RETAINS neural-operator accuracy on standard (away-from-boundary)
queries, (2) RETAINS zero-shot resolution, (3) FIXES near-boundary close evaluation (direct degrades, QBX
stays accurate), (4) is end-to-end DIFFERENTIABLE (trainable through QBX). Loads the trained nbio_s0
checkpoint if present, else an untrained split model."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.common import get_device, configure_torch
from src.models import build_model
from src.data.bie import boundary, sample_coeffs, sample_charges, sample_queries, eval_u


def make_batch(coeffs_list, charges_list, n_b, query, device, d=0.02):
    Xb, Nb, wb, curv, g, Xq, uq = [], [], [], [], [], [], []
    for c, ch in zip(coeffs_list, charges_list):
        bd = boundary(c, n_b)
        if query == "away":
            q = sample_queries(np.random.default_rng(0), c, "interior", 256, margin=0.15)
        else:                                   # near-boundary: push boundary nodes inward by d
            idx = np.linspace(0, n_b - 1, 256).astype(int)
            q = bd["X"][idx] - d * bd["normal"][idx]
        Xb.append(bd["X"]); Nb.append(bd["normal"]); wb.append(bd["w"]); curv.append(bd["curv"])
        g.append(eval_u(bd["X"], ch)); Xq.append(q); uq.append(eval_u(q, ch))
    t = lambda L: torch.tensor(np.stack(L), dtype=torch.float32, device=device)
    return {"Xb": t(Xb), "Nb": t(Nb), "wb": t(wb), "curv": t(curv), "g": t(g), "Xq": t(Xq)}, t(uq)


def rel(u, uq):
    return (torch.linalg.vector_norm(u - uq, dim=1) / torch.linalg.vector_norm(uq, dim=1).clamp_min(1e-12)).mean().item()


def main():
    configure_torch(); device = get_device(True)
    ckpt = "results/nbio_s0/best.pt"
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(c["config"]); model.load_state_dict(c["state_dict"]); tag = "trained nbio_s0"
    else:
        model = build_model({"model": {"name": "nbio", "kwargs": {"hidden": 64, "n_layers": 3, "split": True}}})
        tag = "untrained split (no checkpoint found)"
    model = model.to(device).eval()
    print(f"[qbx-nbio] using {tag}")
    rng = np.random.default_rng(7)
    cs = [sample_coeffs(rng, 5) for _ in range(12)]
    chs = [sample_charges(rng, "interior", c, 6) for c in cs]

    def run(model, query, n_b=128, d=0.02):
        b, uq = make_batch(cs, chs, n_b, query, device, d=d)
        with torch.no_grad():
            model.qbx = False; ud = model(b)
            model.qbx = True; model.qbx_kw = dict(order=12, r_fac=4.0, near_fac=2.0); uq_ = model(b)
        return rel(ud, uq), rel(uq_, uq)

    print("\n(1) RETAIN accuracy on standard away-from-boundary queries (direct vs QBX should match):")
    rd, rq = run(model, "away"); print(f"    away  : direct {rd:.2e} | QBX {rq:.2e}")
    print("\n(3) FIX near-boundary close evaluation (direct should degrade as d->0; QBX stays accurate):")
    for d in (0.05, 0.02, 0.005, 0.001):
        rd, rq = run(model, "near", d=d); print(f"    d={d:<6}: direct {rd:.2e} | QBX {rq:.2e}")
    print("\n(2) RETAIN zero-shot resolution (near-boundary d=0.01, vary n_b; QBX stays accurate):")
    for nb in (64, 128, 256):
        rd, rq = run(model, "near", n_b=nb, d=0.01); print(f"    n_b={nb:<4}: direct {rd:.2e} | QBX {rq:.2e}")

    print("\n(4) DIFFERENTIABLE end-to-end through QBX (backward + grad-norm):")
    model.qbx = True; model.train()
    b, uq = make_batch(cs[:4], chs[:4], 128, "near", device, d=0.02)
    loss = ((model(b) - uq) ** 2).mean(); loss.backward()
    gn = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None and p.requires_grad)
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"    loss {loss.item():.3e}  grad-norm {gn:.3e}  finite={finite}")
    assert gn > 0 and finite, "QBX path not differentiable"
    print("\n[validate_qbx_nbio] DONE")


if __name__ == "__main__":
    main()
