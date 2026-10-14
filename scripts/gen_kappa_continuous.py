#!/usr/bin/env python
"""Generate continuous-kappa training data and a fresh post-failure holdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bie import (  # noqa: E402
    sample_charges, sample_coeffs, sample_queries,
)
from gen_kappa_conditioned import make_row, stack  # noqa: E402


TEST_KAPPAS = (0.0, 0.125, 0.375, 0.75, 1.25, 1.75, 2.5, 3.5, 4.0, 4.5, 5.5)
PRIMARY_INTERPOLATION = (0.125, 0.375, 0.75, 1.25, 1.75, 2.5, 3.5)
PRIMARY_EXTRAPOLATION = (4.5, 5.5)
RESOLUTIONS = (64, 128, 256)
QUERY_MARGINS = (0.12, 0.02)


def continuous_kappas(rng, n_samples, endpoint_fraction=0.1):
    n_endpoint = int(round(endpoint_fraction * n_samples))
    n_continuous = n_samples - 2 * n_endpoint
    strata = (np.arange(n_continuous) + rng.uniform(size=n_continuous)) / n_continuous
    values = np.concatenate((
        np.zeros(n_endpoint),
        4.0 * strata,
        np.full(n_endpoint, 4.0),
    ))
    rng.shuffle(values)
    return values


def generated_split(n_samples, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for kappa in continuous_kappas(rng, n_samples):
        coeffs = sample_coeffs(rng)
        charges = sample_charges(rng, "interior", coeffs, n_charges=6)
        query = sample_queries(rng, coeffs, "interior", n_q=256, margin=0.12)
        rows.append(make_row(
            coeffs, charges, query, float(kappa), 128, False, query_margin=0.12
        ))
    return stack(rows)


def paired_test_bases(n_samples, seed):
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(n_samples):
        coeffs = sample_coeffs(rng)
        charges = sample_charges(rng, "interior", coeffs, n_charges=6)
        queries = {
            margin: sample_queries(rng, coeffs, "interior", n_q=256, margin=margin)
            for margin in QUERY_MARGINS
        }
        bases.append((coeffs, charges, queries))
    return bases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data_cache/kappa_continuous")
    parser.add_argument("--train-n", type=int, default=2400)
    parser.add_argument("--val-n", type=int, default=400)
    parser.add_argument("--test-n", type=int, default=64)
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.save(generated_split(args.train_n, 9201), out / "yukawa_train.pt")
    torch.save(generated_split(args.val_n, 9202), out / "yukawa_val.pt")
    bases = paired_test_bases(args.test_n, 9203)
    combined = []
    for kappa in TEST_KAPPAS:
        for margin in QUERY_MARGINS:
            for n_b in RESOLUTIONS:
                rows = [
                    make_row(
                        coeffs, charges, queries[margin], kappa, n_b, True,
                        query_margin=margin,
                    )
                    for coeffs, charges, queries in bases
                ]
                data = stack(rows)
                kappa_tag = str(kappa).replace(".", "p")
                margin_tag = f"m{int(round(1000 * margin)):03d}"
                torch.save(data, out / f"test_k{kappa_tag}_{margin_tag}_nb{n_b}.pt")
                if margin == 0.12 and n_b == 128:
                    combined.extend(rows)
                error = torch.linalg.vector_norm(data["bem_uq"] - data["uq"], dim=-1)
                error /= torch.linalg.vector_norm(data["uq"], dim=-1).clamp_min(1e-12)
                print(
                    f"kappa={kappa:g} margin={margin:g} nb={n_b} "
                    f"correct BIE mean={error.mean().item():.3e}",
                    flush=True,
                )
    torch.save(stack(combined), out / "yukawa_test.pt")
    manifest = {
        "protocol": "continuous_kappa_followup_after_discrete_anchor_failure",
        "train_range": [0.0, 4.0],
        "endpoint_fraction_each": 0.1,
        "test_kappas": list(TEST_KAPPAS),
        "primary_interpolation_kappas": list(PRIMARY_INTERPOLATION),
        "primary_extrapolation_kappas": list(PRIMARY_EXTRAPOLATION),
        "resolutions": list(RESOLUTIONS), "query_margins": list(QUERY_MARGINS),
        "train_n": args.train_n, "val_n": args.val_n, "test_n": args.test_n,
        "seeds": {"train": 9201, "val": 9202, "test": 9203},
        "freshness": (
            "The primary test kappa values and geometry seed were fixed after diagnosing the prior "
            "discrete-anchor holdout; none of the primary values appeared in that earlier test."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
