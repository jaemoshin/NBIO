#!/usr/bin/env python
"""Exterior-problem splits (unbounded domain). GT is the EXACT manufactured solution; the classical-BEM
baseline uses the deflated double-layer + least-squares (pure double-layer is rank-deficient for 2D
exterior Dirichlet). Validates the deflated BEM vs exact before writing."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import BIEConfig, make_sample, validate_bem


def gen(n, cfg, seed):
    rng = np.random.default_rng(seed)
    s = [make_sample(rng, cfg) for _ in range(n)]
    return {k: torch.from_numpy(np.stack([x[k] for x in s])) for k in s[0].keys()}


def main():
    out = "data_cache"
    errs = [validate_bem(seed=s, n_b=256, region="exterior") for s in range(8)]
    print(f"deflated exterior BEM vs exact (n_b=256): mean {np.mean(errs):.2e} max {np.max(errs):.2e}")
    cfg = BIEConfig(region="exterior", n_b=128, n_q=256)
    for split, n, seed in [("train", 800, 21), ("val", 200, 22), ("test", 200, 11)]:
        d = gen(n, cfg, seed)
        torch.save(d, os.path.join(out, f"exterior_{split}.pt"))
        b = d["bem_uq"]; fin = torch.isfinite(b).all(dim=1)
        rel = (torch.linalg.vector_norm(b[fin] - d["uq"][fin], dim=1)
               / (torch.linalg.vector_norm(d["uq"][fin], dim=1) + 1e-9)).mean().item() if fin.any() else float("nan")
        print(f"  exterior_{split}: N={n}  BEM rel-L2={rel:.2e}  finite={int(fin.sum())}/{n}")
    print("[gen_exterior] DONE")


if __name__ == "__main__":
    main()
