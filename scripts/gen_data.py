#!/usr/bin/env python
"""Generate the Laplace-BVP operator-learning dataset. Validates the Nystrom-BEM solver against the
EXACT manufactured solution before writing anything (catches sign/curvature bugs)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import (BIEConfig, make_sample, sample_coeffs, sample_charges, sample_queries,
                          boundary, eval_u, bem_solve, validate_bem)


def gen_random(n, cfg, seed):
    rng = np.random.default_rng(seed)
    samples = [make_sample(rng, cfg) for _ in range(n)]
    keys = samples[0].keys()
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in keys}


def gen_fixed_geometry_multires(n, region, n_modes, n_q, n_charges, nb_list, seed):
    """Same geometries/charges/queries across several boundary resolutions -> isolates zero-shot n_b."""
    rng = np.random.default_rng(seed)
    coeffs = [sample_coeffs(rng, n_modes) for _ in range(n)]
    charges = [sample_charges(rng, region, c, n_charges) for c in coeffs]
    queries = [sample_queries(rng, c, region, n_q) for c in coeffs]
    out = {}
    for nb in nb_list:
        rows = []
        for c, ch, Xq in zip(coeffs, charges, queries):
            bd = boundary(c, nb)
            g = eval_u(bd["X"], ch); uq = eval_u(Xq, ch)
            d = {"Xb": bd["X"], "Nb": bd["normal"], "wb": bd["w"], "curv": bd["curv"], "speed": bd["speed"],
                 "g": g, "Xq": Xq, "uq": uq, "a": c["a"], "b": c["b"]}
            try:
                mu, ev = bem_solve(bd, g, region); d["bem_uq"] = ev(Xq); d["mu"] = mu
            except np.linalg.LinAlgError:
                d["bem_uq"] = np.full_like(uq, np.nan); d["mu"] = np.full_like(g, np.nan)
            rows.append({k: np.asarray(v, np.float32) for k, v in d.items()})
        out[nb] = {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0].keys()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_cache")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-modes", type=int, default=5)
    ap.add_argument("--n-b", type=int, default=128)
    ap.add_argument("--n-q", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== BEM self-validation vs EXACT manufactured solution ===")
    for region in ("interior", "exterior"):
        errs = [validate_bem(seed=s, n_b=256, region=region) for s in range(5)]
        print(f"  {region:8s} n_b=256: rel-err vs exact = {np.mean(errs):.2e} (max {np.max(errs):.2e})")
        assert np.mean(errs) < 1e-3, f"BEM not validating for {region} (mean {np.mean(errs):.2e})"
    for nb in (64, 128):
        e = np.mean([validate_bem(seed=s, n_b=nb, region="interior") for s in range(5)])
        print(f"  interior n_b={nb}: rel-err vs exact = {e:.2e}")

    cfg_i = BIEConfig(n_modes=args.n_modes, region="interior", n_b=args.n_b, n_q=args.n_q)
    print("\n=== generating splits ===")
    for split, n, seed in [("train", args.n_train, 1), ("val", args.n_val, 2), ("test", args.n_test, 3)]:
        d = gen_random(n, cfg_i, seed)
        torch.save(d, os.path.join(args.out_dir, f"interior_{split}.pt"))
        # report this split's classical-BEM accuracy (reference ceiling for the learned operator)
        bem_rel = (torch.linalg.vector_norm(d["bem_uq"] - d["uq"], dim=1)
                   / (torch.linalg.vector_norm(d["uq"], dim=1) + 1e-9)).mean().item()
        print(f"  interior_{split}: N={n} n_b={args.n_b} n_q={args.n_q}  BEM rel-L2={bem_rel:.2e}")

    print("\n=== zero-shot resolution test (same geometry @ n_b in {64,128,256}) ===")
    mr = gen_fixed_geometry_multires(120, "interior", args.n_modes, args.n_q, 6, [64, 128, 256], seed=7)
    for nb, d in mr.items():
        torch.save(d, os.path.join(args.out_dir, f"interior_testres{nb}.pt"))
        bem_rel = (torch.linalg.vector_norm(d["bem_uq"] - d["uq"], dim=1)
                   / (torch.linalg.vector_norm(d["uq"], dim=1) + 1e-9)).mean().item()
        print(f"  interior_testres{nb}: N=120  BEM rel-L2={bem_rel:.2e}")

    print("\n=== exterior test (unbounded domain; volumetric grids cannot) ===")
    cfg_e = BIEConfig(n_modes=args.n_modes, region="exterior", n_b=args.n_b, n_q=args.n_q)
    de = gen_random(200, cfg_e, seed=11)
    torch.save(de, os.path.join(args.out_dir, "exterior_test.pt"))
    bem_rel = (torch.linalg.vector_norm(de["bem_uq"] - de["uq"], dim=1)
               / (torch.linalg.vector_norm(de["uq"], dim=1) + 1e-9)).mean().item()
    print(f"  exterior_test: N=200 n_b={args.n_b}  BEM rel-L2={bem_rel:.2e}")
    print("\n[gen_data] DONE")


if __name__ == "__main__":
    main()
