#!/usr/bin/env python
"""CPU unit tests for ScreeningLawNBIO: constraints, exact kappa=0 endpoint, gradients."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model
from src.data.bie import sample_coeffs, boundary, sample_charges, eval_u, bem_solve, sample_queries


def make_batch(kappa, dtype=torch.float64):
    rng = np.random.default_rng(5)
    co = sample_coeffs(rng); bd = boundary(co, 64); ch = sample_charges(rng, "interior", co)
    g = eval_u(bd["X"], ch, kappa)
    Xq = sample_queries(rng, co, "interior", 16, margin=0.12)
    uq = eval_u(Xq, ch, kappa)
    b = {k: torch.tensor(v, dtype=dtype).unsqueeze(0) for k, v in
         [("Xb", bd["X"]), ("Nb", bd["normal"]), ("wb", bd["w"]), ("curv", bd["curv"]),
          ("g", g), ("Xq", Xq)]}
    b["kappa"] = torch.tensor([kappa], dtype=dtype)
    return b, bd, ch, Xq, uq


def main():
    for variant, kwargs in (("mixture", {"n_atoms": 8}), ("radial_mlp", {}),
                            ("radial_endpoint", {}), ("radial_quadratic", {})):
        cfg = {"model": {"name": "nbio_screening", "kwargs": {"variant": variant, **kwargs}}}
        m = build_model(cfg).double()
        n = sum(p.numel() for p in m.parameters())
        s = torch.tensor([0.0, 0.5, 2.0, 10.0, 25.0], dtype=torch.float64)
        law = m.m_law(s)
        assert abs(law[0].item() - 1.0) < 1e-12, "m(0)=1 violated"
        if variant in {"radial_endpoint", "radial_quadratic"}:
            s0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            endpoint_derivative = torch.autograd.grad(m.m_law(s0).sum(), s0)[0]
            assert endpoint_derivative.abs().max() < 1e-12, "m'(0)=0 violated"
        if variant == "mixture":
            assert (law > 0).all() and (law <= 1 + 1e-12).all(), "bounds violated"
            assert (law[1:] <= law[:-1] + 1e-12).all(), "monotonicity violated"
        b, bd, ch, Xq, uq = make_batch(0.0)
        _, ev = bem_solve(bd, eval_u(bd["X"], ch, 0.0), "interior", kappa=0.0)
        with torch.no_grad():
            u = m(b).numpy().ravel()
        dev = np.abs(u - ev(Xq)).max()
        assert dev < 1e-10, f"kappa=0 endpoint not exact: {dev}"
        b2, *_ = make_batch(2.0)
        out = m(b2)
        out.square().mean().backward()
        gnorm = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
        assert np.isfinite(gnorm) and gnorm > 0, "gradients broken"
        print(f"[{variant}] params={n} m(0)=1 exact, endpoint dev {dev:.1e}, grad-norm {gnorm:.2e} OK")
    print("ALL UNIT TESTS PASS")


if __name__ == "__main__":
    main()
