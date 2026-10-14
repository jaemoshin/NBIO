#!/usr/bin/env python
"""Evaluate fresh-data-seed exact-protocol replications on the frozen confirmation holdout."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402


KAPPAS = (0.0625, 0.2, 0.6, 1.1, 1.6, 2.2, 3.0, 3.8)
SLICES = (("bulk", 0.12, 128), ("near", 0.02, 256))
RESULT = ROOT / "results" / "kappa_seed_replication"
REPORT = ROOT / "reports" / "kappa_seed_replication_results.md"


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def row_hashes(tensor: torch.Tensor):
    return {
        hashlib.sha1(row.contiguous().numpy().tobytes()).hexdigest()
        for row in tensor.cpu()
    }


def load_model(data_seed: int, method: str, device: torch.device):
    path = RESULT / f"data_s{data_seed}_{method}" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    validation = []
    holdout = torch.load(
        ROOT / "data_cache/kappa_confirmation/test_k0p0625_m120_nb128.pt",
        map_location="cpu", weights_only=False,
    )
    holdout_hashes = row_hashes(holdout["Xb"])
    leakage = []
    for data_seed in range(3):
        train = torch.load(
            ROOT / f"data_cache/kappa_seed_replication/data_s{data_seed}/yukawa_train.pt",
            map_location="cpu", weights_only=False,
        )
        shared = row_hashes(train["Xb"]) & holdout_hashes
        leakage.append({
            "data_seed": data_seed, "shared_boundary_tensors": len(shared),
        })
        if shared:
            raise RuntimeError(f"data seed {data_seed} shares holdout geometries")
        models = {
            method: load_model(data_seed, method, device) for method in ("full", "none")
        }
        for method in ("full", "none"):
            history = json.loads(
                (RESULT / f"data_s{data_seed}_{method}" / "history.json").read_text()
            )
            validation.append({
                "data_seed": data_seed, "method": method,
                "best_val_rel_l2": history["best_val_rel_l2"],
            })
        for slice_name, margin, n_boundary in SLICES:
            margin_tag = f"m{int(round(1000 * margin)):03d}"
            for kappa in KAPPAS:
                data = torch.load(
                    ROOT / "data_cache/kappa_confirmation"
                    / f"test_k{tag(kappa)}_{margin_tag}_nb{n_boundary}.pt",
                    map_location="cpu", weights_only=False,
                )
                for method, model in models.items():
                    errors = []
                    with torch.no_grad():
                        for start in range(0, data["uq"].shape[0], 32):
                            stop = min(start + 32, data["uq"].shape[0])
                            batch = {key: value[start:stop].to(device) for key, value in data.items()}
                            prediction = model(batch)
                            error = torch.linalg.vector_norm(
                                prediction - batch["uq"], dim=-1
                            ) / torch.linalg.vector_norm(batch["uq"], dim=-1).clamp_min(1e-9)
                            errors.extend(error.cpu().tolist())
                    for sample_id, error in enumerate(errors):
                        rows.append({
                            "data_seed": data_seed, "method": method, "kappa": kappa,
                            "slice": slice_name, "query_margin": margin,
                            "n_b": n_boundary, "sample_id": sample_id, "rel_l2": error,
                        })
                print(
                    f"data_seed={data_seed} {slice_name} kappa={kappa:g} complete",
                    flush=True,
                )

    with (RESULT / "raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (RESULT / "validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation[0].keys())
        writer.writeheader()
        writer.writerows(validation)

    summary = []
    for slice_name, _, n_boundary in SLICES:
        for method in ("full", "none"):
            values = np.asarray([
                row["rel_l2"] for row in rows
                if row["slice"] == slice_name and row["method"] == method
            ])
            summary.append({
                "slice": slice_name, "n_b": n_boundary, "method": method,
                "n_rows": len(values), "mean_rel_l2": float(values.mean()),
                "median_rel_l2": float(np.median(values)),
                "p90_rel_l2": float(np.percentile(values, 90)),
            })
    with (RESULT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    paired = {}
    for slice_name, _, _ in SLICES:
        cluster = defaultdict(list)
        for row in rows:
            if row["slice"] == slice_name:
                cluster[(row["method"], row["sample_id"])].append(row["rel_l2"])
        full = np.asarray([np.mean(cluster[("full", i)]) for i in range(96)])
        none = np.asarray([np.mean(cluster[("none", i)]) for i in range(96)])
        difference = full - none
        rng = np.random.default_rng(47)
        boot = np.asarray([
            difference[rng.integers(0, 96, 96)].mean() for _ in range(10_000)
        ])
        paired[slice_name] = {
            "full_mean": float(full.mean()), "none_mean": float(none.mean()),
            "relative_improvement": float(1.0 - full.mean() / none.mean()),
            "mean_difference": float(difference.mean()),
            "bootstrap_mean_difference_ci95": [
                float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            ],
            "full_win_fraction": float((difference < 0).mean()),
        }
    audit = {
        "protocol": {
            "training": "exact project trainer: 100 epochs, cosine annealing, best checkpoint",
            "train_data_seeds": [12001, 12002, 12003],
            "validation_data_seeds": [13001, 13002, 13003],
            "model_seeds": [100, 101, 102],
            "holdout": "frozen kappa_confirmation seed 9303",
        },
        "validation": validation, "leakage": leakage, "paired": paired,
        "verdict": "PAPERIZE" if all(
            cell["bootstrap_mean_difference_ci95"][1] < 0 for cell in paired.values()
        ) else "NARROW",
    }
    (RESULT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    lines = [
        "# Exact-protocol continuous-kappa data-seed replication",
        "",
        "Three fresh train/validation dataset seeds were used with paired model initializations. Both "
        "methods use the exact project trainer (100 epochs, cosine annealing, and validation-selected best "
        "checkpoints) and are evaluated on the pre-existing frozen confirmation holdout.",
        "",
        "| slice | full prior | no prior | full improvement | paired 95% CI | win fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for slice_name in ("bulk", "near"):
        cell = paired[slice_name]
        lines.append(
            f"| {slice_name} | `{cell['full_mean']:.3e}` | `{cell['none_mean']:.3e}` | "
            f"`{100 * cell['relative_improvement']:.1f}%` | "
            f"`[{cell['bootstrap_mean_difference_ci95'][0]:.3e}, "
            f"{cell['bootstrap_mean_difference_ci95'][1]:.3e}]` | "
            f"`{100 * cell['full_win_fraction']:.1f}%` |"
        )
    lines.extend(["", "Verdict: `" + audit["verdict"] + "`."])
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
