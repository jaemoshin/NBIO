#!/usr/bin/env python
"""Evaluate frozen continuous-kappa checkpoints on boundary-load OOD Fourier traces.

This is an evaluation-only audit. It reuses the frozen confirmation geometries and checkpoints,
replaces manufactured point-source boundary traces by low-mode Fourier traces, and recomputes the
correct modified-Helmholtz BEM reference for each kappa.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bie import bem_solve  # noqa: E402
from src.models import build_model  # noqa: E402


KAPPAS = (0.0625, 0.2, 0.6, 1.1, 1.6, 2.2, 3.0, 3.8)
METHODS = ("full_norm", "none_norm")
OUT = ROOT / "results" / "kappa_confirmation_fourier"


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def fourier_trace(n_boundary: int, sample_id: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + sample_id)
    cosine = rng.standard_normal(7)
    sine = rng.standard_normal(7)
    theta = 2.0 * np.pi * np.arange(n_boundary) / n_boundary
    values = np.full(n_boundary, cosine[0])
    for mode in range(1, 7):
        values += cosine[mode] * np.cos(mode * theta)
        values += sine[mode] * np.sin(mode * theta)
    return values


def load_models(device: torch.device):
    models = defaultdict(list)
    for method in METHODS:
        for seed in (0, 1, 2):
            checkpoint_path = ROOT / f"results/kappa_continuous/{method}_s{seed}/best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            models[method].append((seed, model))
    return models


def rel_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + 1e-30))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.load(
        ROOT / "data_cache/kappa_confirmation/test_k0p0625_m120_nb128.pt",
        map_location="cpu", weights_only=False,
    )
    reference_base = torch.load(
        ROOT / "data_cache/kappa_confirmation/test_k0p0625_m120_nb256.pt",
        map_location="cpu", weights_only=False,
    )
    if not torch.equal(base["Xq"], reference_base["Xq"]):
        raise RuntimeError("128- and 256-node confirmation files do not share query points")
    n_samples, n_boundary = base["Xb"].shape[:2]
    traces = np.stack([fourier_trace(n_boundary, i) for i in range(n_samples)])
    reference_traces = np.stack([
        fourier_trace(reference_base["Xb"].shape[1], i) for i in range(n_samples)
    ])
    models = load_models(device)
    raw_rows = []

    for kappa in KAPPAS:
        reference = []
        coarse_correct = []
        fixed_laplace = []
        for sample_id in range(n_samples):
            coarse_boundary = {
                "X": base["Xb"][sample_id].numpy(),
                "normal": base["Nb"][sample_id].numpy(),
                "w": base["wb"][sample_id].numpy(),
                "curv": base["curv"][sample_id].numpy(),
            }
            fine_boundary = {
                "X": reference_base["Xb"][sample_id].numpy(),
                "normal": reference_base["Nb"][sample_id].numpy(),
                "w": reference_base["wb"][sample_id].numpy(),
                "curv": reference_base["curv"][sample_id].numpy(),
            }
            _, coarse_eval = bem_solve(
                coarse_boundary, traces[sample_id], "interior", kappa=kappa
            )
            _, correct_eval = bem_solve(
                fine_boundary, reference_traces[sample_id], "interior", kappa=kappa
            )
            _, laplace_eval = bem_solve(
                coarse_boundary, traces[sample_id], "interior", kappa=0.0
            )
            query = base["Xq"][sample_id].numpy()
            reference.append(correct_eval(query))
            coarse_correct.append(coarse_eval(query))
            fixed_laplace.append(laplace_eval(query))
        reference = np.asarray(reference)
        coarse_correct = np.asarray(coarse_correct)
        fixed_laplace = np.asarray(fixed_laplace)

        for sample_id in range(n_samples):
            raw_rows.append({
                "method": "correct_bie_nb128", "seed": -1, "kappa": kappa,
                "sample_id": sample_id,
                "rel_l2": rel_l2(coarse_correct[sample_id], reference[sample_id]),
            })
            raw_rows.append({
                "method": "fixed_laplace_bie", "seed": -1, "kappa": kappa,
                "sample_id": sample_id,
                "rel_l2": rel_l2(fixed_laplace[sample_id], reference[sample_id]),
            })

        common = {
            key: base[key].to(device)
            for key in ("Xb", "Nb", "wb", "curv", "Xq")
        }
        common["g"] = torch.tensor(traces, dtype=torch.float32, device=device)
        common["kappa"] = torch.full((n_samples,), kappa, dtype=torch.float32, device=device)
        with torch.no_grad():
            for method, seeded_models in models.items():
                for seed, model in seeded_models:
                    predictions = []
                    for start in range(0, n_samples, 32):
                        stop = min(start + 32, n_samples)
                        batch = {key: value[start:stop] for key, value in common.items()}
                        predictions.append(model(batch).cpu().numpy())
                    prediction = np.concatenate(predictions)
                    for sample_id in range(n_samples):
                        raw_rows.append({
                            "method": method, "seed": seed, "kappa": kappa,
                            "sample_id": sample_id,
                            "rel_l2": rel_l2(prediction[sample_id], reference[sample_id]),
                        })
        print(f"kappa={kappa:g} complete", flush=True)

    raw_path = OUT / "raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["method"], row["kappa"])].append(row["rel_l2"])
    summary = []
    for (method, kappa), values in sorted(grouped.items()):
        array = np.asarray(values)
        summary.append({
            "method": method, "kappa": kappa, "n_rows": len(array),
            "mean_rel_l2": float(array.mean()), "median_rel_l2": float(np.median(array)),
            "p90_rel_l2": float(np.percentile(array, 90)),
        })
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    aggregate = {}
    for method in (*METHODS, "correct_bie_nb128", "fixed_laplace_bie"):
        values = np.asarray([row["rel_l2"] for row in raw_rows if row["method"] == method])
        aggregate[method] = {
            "mean_rel_l2": float(values.mean()),
            "median_rel_l2": float(np.median(values)),
            "p90_rel_l2": float(np.percentile(values, 90)),
            "n_rows": int(len(values)),
        }
    result = {
        "protocol": {
            "checkpoint_policy": "existing checkpoints frozen before this audit",
            "boundary_load": "six-mode Fourier trace, independent of kappa",
            "reference": "256-node correct-kernel Nystrom BEM; models evaluated at 128 nodes",
            "n_samples": n_samples, "n_boundary": n_boundary,
            "query_margin": 0.12, "kappas": KAPPAS, "model_seeds": [0, 1, 2],
        },
        "aggregate": aggregate,
        "full_vs_none_relative_improvement": float(
            1.0 - aggregate["full_norm"]["mean_rel_l2"] / aggregate["none_norm"]["mean_rel_l2"]
        ),
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
