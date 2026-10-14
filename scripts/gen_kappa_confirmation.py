#!/usr/bin/env python
"""Generate a second untouched holdout for the discovered full-prior family result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bie import sample_charges, sample_coeffs, sample_queries  # noqa: E402
from gen_kappa_conditioned import make_row, stack  # noqa: E402


INTERPOLATION_KAPPAS = (0.0625, 0.2, 0.6, 1.1, 1.6, 2.2, 3.0, 3.8)
EXTRAPOLATION_KAPPAS = (4.25, 5.25)
TEST_KAPPAS = (0.0,) + INTERPOLATION_KAPPAS + EXTRAPOLATION_KAPPAS
MARGINS = (0.12, 0.02)
RESOLUTIONS = (64, 128, 256)


def bases(n_samples, seed):
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(n_samples):
        coeffs = sample_coeffs(rng)
        charges = sample_charges(rng, "interior", coeffs, n_charges=6)
        queries = {
            margin: sample_queries(rng, coeffs, "interior", n_q=256, margin=margin)
            for margin in MARGINS
        }
        output.append((coeffs, charges, queries))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data_cache/kappa_confirmation")
    parser.add_argument("--test-n", type=int, default=96)
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frozen_bases = bases(args.test_n, 9303)
    for kappa in TEST_KAPPAS:
        for margin in MARGINS:
            for n_b in RESOLUTIONS:
                rows = [
                    make_row(
                        coeffs, charges, queries[margin], kappa, n_b, True,
                        query_margin=margin,
                    )
                    for coeffs, charges, queries in frozen_bases
                ]
                data = stack(rows)
                kappa_tag = str(kappa).replace(".", "p")
                margin_tag = f"m{int(round(1000 * margin)):03d}"
                torch.save(data, out / f"test_k{kappa_tag}_{margin_tag}_nb{n_b}.pt")
                error = torch.linalg.vector_norm(data["bem_uq"] - data["uq"], dim=-1)
                error /= torch.linalg.vector_norm(data["uq"], dim=-1).clamp_min(1e-12)
                print(
                    f"kappa={kappa:g} margin={margin:g} nb={n_b} "
                    f"correct BIE mean={error.mean().item():.3e}", flush=True
                )
    manifest = {
        "purpose": "confirm_discovered_continuous_family_full_prior_result",
        "checkpoint_policy": "all model checkpoints frozen before this holdout was generated",
        "interpolation_kappas": list(INTERPOLATION_KAPPAS),
        "extrapolation_kappas": list(EXTRAPOLATION_KAPPAS),
        "diagnostic_kappas": [0.0],
        "test_geometry_seed": 9303, "n_instances": args.test_n,
        "margins": list(MARGINS), "resolutions": list(RESOLUTIONS),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
