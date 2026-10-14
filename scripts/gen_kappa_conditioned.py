#!/usr/bin/env python
"""Generate a frozen mixed-kappa Yukawa operator-family benchmark."""
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
    boundary, bem_solve, eval_u, sample_charges, sample_coeffs, sample_queries,
)


TRAIN_KAPPAS = (0.0, 1.0, 2.0, 4.0)
TEST_KAPPAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
RESOLUTIONS = (64, 128, 256)
QUERY_MARGINS = (0.12, 0.02)


def stack(rows):
    return {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}


def make_row(coeffs, charges, query, kappa, n_b, with_bem, query_margin=0.12):
    geometry = boundary(coeffs, n_b)
    boundary_value = eval_u(geometry["X"], charges, kappa)
    target = eval_u(query, charges, kappa)
    row = {
        "Xb": geometry["X"], "Nb": geometry["normal"], "wb": geometry["w"],
        "curv": geometry["curv"], "speed": geometry["speed"], "g": boundary_value,
        "Xq": query, "uq": target, "kappa": np.asarray(kappa),
        "query_margin": np.asarray(query_margin),
    }
    if with_bem:
        data = {
            "X": geometry["X"], "normal": geometry["normal"],
            "w": geometry["w"], "curv": geometry["curv"],
        }
        density, correct = bem_solve(data, boundary_value, "interior", kappa=kappa)
        _, laplace = bem_solve(data, boundary_value, "interior", kappa=0.0)
        row.update(mu=density, bem_uq=correct(query), laplace_bem_uq=laplace(query))
    return {key: np.asarray(value, dtype=np.float32) for key, value in row.items()}


def generated_split(n_samples, seed, n_b=128, with_bem=False):
    rng = np.random.default_rng(seed)
    kappas = np.resize(np.asarray(TRAIN_KAPPAS), n_samples)
    rng.shuffle(kappas)
    rows = []
    for kappa in kappas:
        coeffs = sample_coeffs(rng)
        charges = sample_charges(rng, "interior", coeffs, n_charges=6)
        query = sample_queries(rng, coeffs, "interior", n_q=256, margin=0.12)
        rows.append(make_row(
            coeffs, charges, query, float(kappa), n_b, with_bem,
            query_margin=0.12,
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
    parser.add_argument("--out-dir", default="data_cache/kappa_conditioned")
    parser.add_argument("--train-n", type=int, default=1200)
    parser.add_argument("--val-n", type=int, default=200)
    parser.add_argument("--test-n", type=int, default=64)
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.save(generated_split(args.train_n, 8101), out / "yukawa_train.pt")
    torch.save(generated_split(args.val_n, 8102), out / "yukawa_val.pt")
    bases = paired_test_bases(args.test_n, 8103)
    manifest = {
        "train_kappas": list(TRAIN_KAPPAS), "test_kappas": list(TEST_KAPPAS),
        "resolutions": list(RESOLUTIONS), "query_margins": list(QUERY_MARGINS),
        "train_n": args.train_n,
        "val_n": args.val_n, "test_instances_per_cell": args.test_n,
        "split_seeds": {"train": 8101, "val": 8102, "test": 8103},
    }
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
                path = out / f"test_k{kappa_tag}_{margin_tag}_nb{n_b}.pt"
                torch.save(data, path)
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
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
