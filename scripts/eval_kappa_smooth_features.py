#!/usr/bin/env python
"""Compare smooth pair features with the legacy conditioned correction."""
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

from src.models import build_model  # noqa: E402


KAPPAS = (0.0625, 0.2, 0.6, 1.1, 1.6, 2.2, 3.0, 3.8)
SLICES = (("bulk", 0.12, 128), ("near", 0.02, 256))
RESULT = ROOT / "results" / "kappa_smooth_features"


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def load_model(root: Path, data_seed: int, method: str, device: torch.device):
    path = root / f"data_s{data_seed}_{method}" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), path


def rel_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - target, dim=-1) / torch.linalg.vector_norm(
        target, dim=-1
    ).clamp_min(1e-9)


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values: np.ndarray, seed: int, n_boot: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def clustered(rows: list[dict], method: str, slice_name: str) -> np.ndarray:
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["slice"] == slice_name:
            groups[row["sample_id"]].append(row["rel_l2"])
    return np.asarray([np.mean(groups[index]) for index in range(96)])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy_root = ROOT / "results" / "kappa_seed_replication"
    models = {}
    checkpoint_paths = []
    for feature, root in (("legacy", legacy_root), ("smooth", RESULT)):
        for method in ("full", "none"):
            for data_seed in range(3):
                model, path = load_model(root, data_seed, method, device)
                models[(f"{feature}_{method}", data_seed)] = model
                checkpoint_paths.append(str(path.relative_to(ROOT)))

    rows = []
    for slice_name, margin, n_b in SLICES:
        margin_tag = f"m{int(round(1000 * margin)):03d}"
        for kappa in KAPPAS:
            data = torch.load(
                ROOT / "data_cache" / "kappa_confirmation"
                / f"test_k{tag(kappa)}_{margin_tag}_nb{n_b}.pt",
                map_location="cpu", weights_only=False,
            )
            for method, key in (
                ("correct_yukawa_bie", "bem_uq"),
                ("fixed_laplace_bie", "laplace_bem_uq"),
            ):
                for sample_id, error in enumerate(rel_l2(data[key], data["uq"]).tolist()):
                    rows.append({
                        "method": method, "data_seed": -1, "kappa": kappa,
                        "slice": slice_name, "query_margin": margin, "n_b": n_b,
                        "sample_id": sample_id, "rel_l2": error,
                    })
            for (method, data_seed), model in models.items():
                errors = []
                with torch.no_grad():
                    for start in range(0, data["uq"].shape[0], 32):
                        batch = {
                            key: value[start:start + 32].to(device)
                            for key, value in data.items()
                        }
                        errors.extend(rel_l2(model(batch), batch["uq"]).cpu().tolist())
                for sample_id, error in enumerate(errors):
                    rows.append({
                        "method": method, "data_seed": data_seed, "kappa": kappa,
                        "slice": slice_name, "query_margin": margin, "n_b": n_b,
                        "sample_id": sample_id, "rel_l2": error,
                    })
            print(f"[smooth-feature] {slice_name} kappa={kappa:g}", flush=True)

    RESULT.mkdir(parents=True, exist_ok=True)
    write_csv(rows, RESULT / "raw.csv")
    summary = []
    for slice_name, _, n_b in SLICES:
        for method in sorted(set(row["method"] for row in rows)):
            values = np.asarray([
                row["rel_l2"] for row in rows
                if row["slice"] == slice_name and row["method"] == method
            ])
            summary.append({
                "slice": slice_name,
                "n_b": n_b,
                "method": method,
                "n_rows": len(values),
                "mean_rel_l2": float(values.mean()),
                "median_rel_l2": float(np.median(values)),
                "p90_rel_l2": float(np.quantile(values, 0.9)),
            })
    write_csv(summary, RESULT / "summary.csv")

    comparisons = []
    comparison_specs = (
        ("smooth_full", "legacy_full"),
        ("smooth_none", "legacy_none"),
        ("smooth_full", "smooth_none"),
        ("smooth_full", "fixed_laplace_bie"),
    )
    seed = 20260722
    for slice_name, _, _ in SLICES:
        for target, control in comparison_specs:
            target_values = clustered(rows, target, slice_name)
            control_values = clustered(rows, control, slice_name)
            difference = target_values - control_values
            comparisons.append({
                "slice": slice_name,
                "target": target,
                "control": control,
                "target_mean": float(target_values.mean()),
                "control_mean": float(control_values.mean()),
                "relative_improvement": float(
                    (control_values.mean() - target_values.mean()) / max(control_values.mean(), 1e-30)
                ),
                "target_win_fraction": float(np.mean(difference < 0.0)),
                "mean_difference": float(difference.mean()),
                "paired_difference_ci": bootstrap(difference, seed),
            })
            seed += 1
    smooth_vs_wrong = [
        row for row in comparisons
        if row["target"] == "smooth_full" and row["control"] == "fixed_laplace_bie"
    ]
    smooth_vs_legacy = [
        row for row in comparisons
        if row["target"] == "smooth_full" and row["control"] == "legacy_full"
    ]
    audit = {
        "protocol": {
            "training_data_seeds": [0, 1, 2],
            "paired_model_seeds": [100, 101, 102],
            "holdout": "frozen kappa_confirmation seed 9303",
            "feature_change": (
                "legacy normalized direction/distance replaced by raw displacement, squared distance, "
                "and smooth normal contractions; capacity and optimizer unchanged"
            ),
            "checkpoints": checkpoint_paths,
        },
        "comparisons": comparisons,
        "retention_rule": (
            "Promote smooth features only if they beat the wrong fixed prior on both slices and do not "
            "degrade legacy full-prior mean error by more than 25% on either slice."
        ),
        "promote_smooth_features": bool(
            all(row["relative_improvement"] > 0 for row in smooth_vs_wrong)
            and all(row["relative_improvement"] > -0.25 for row in smooth_vs_legacy)
        ),
    }
    (RESULT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
