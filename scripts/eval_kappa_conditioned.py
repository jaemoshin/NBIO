#!/usr/bin/env python
"""Evaluate one conditioned NBIO over seen, interpolated, and extrapolated kappa.

The three learned controls have identical correction capacity and training data;
only the analytic prior differs (full Laplace, none, or localized singular part).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402


KAPPAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
MARGINS = (0.12, 0.02)
RESOLUTIONS = (64, 128, 256)
RUNS = (
    ("full", "full"),
    ("none", "none"),
    ("local_both", "local"),
    ("local_potential", "local_potential"),
    ("adaptive", "adaptive"),
)
REGIME = {
    0.0: "seen", 0.25: "interpolation", 0.5: "interpolation",
    1.0: "seen", 1.5: "interpolation", 2.0: "seen",
    3.0: "interpolation", 4.0: "seen", 5.0: "extrapolation",
    6.0: "extrapolation",
}


def save_csv(rows, path):
    if not rows:
        raise ValueError("refusing to write empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def instance_rel(prediction, target):
    numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
    return numerator / denominator


def bootstrap_mean(values, seed, n_boot=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def shard_path(data_dir, kappa, margin, n_b):
    kappa_tag = str(kappa).replace(".", "p")
    margin_tag = f"m{int(round(1000 * margin)):03d}"
    return data_dir / f"test_k{kappa_tag}_{margin_tag}_nb{n_b}.pt"


@torch.no_grad()
def evaluate(run_root, data_dir, seeds, batch_size, device):
    models = []
    training = []
    for method, directory_prefix in RUNS:
        for seed in seeds:
            directory = run_root / f"{directory_prefix}_s{seed}"
            model, checkpoint = checkpoint_model(directory / "best.pt", device)
            models.append((method, seed, model))
            history = json.loads((directory / "history.json").read_text())
            gate_rate = float("nan")
            if getattr(model.bio, "log_gate_rate", None) is not None:
                gate_rate = float(torch.nn.functional.softplus(
                    model.bio.log_gate_rate.detach()
                ).cpu())
            training.append({
                "method": method,
                "prior_mode": checkpoint["config"]["model"]["kwargs"]["prior_mode"],
                "local_scope": checkpoint["config"]["model"]["kwargs"].get(
                    "local_scope", "both"
                ),
                "seed": seed,
                "n_params": int(history["n_params"]),
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_rel_l2": float(history["best_val_rel_l2"]),
                "learned_gate_rate": gate_rate,
                "checkpoint": str((directory / "best.pt").relative_to(ROOT)),
            })

    rows = []
    for kappa in KAPPAS:
        for margin in MARGINS:
            for n_b in RESOLUTIONS:
                path = shard_path(data_dir, kappa, margin, n_b)
                dataset = BIEDataset(path)
                for method, tensor_key in (
                    ("correct_yukawa_bie", "bem_uq"),
                    ("fixed_laplace_bie", "laplace_bem_uq"),
                ):
                    errors = instance_rel(dataset.d[tensor_key], dataset.d["uq"]).numpy()
                    for sample_id, error in enumerate(errors):
                        rows.append({
                            "method": method, "seed": -1, "kappa": kappa,
                            "kappa_regime": REGIME[kappa], "query_margin": margin,
                            "n_b": n_b, "sample_id": sample_id,
                            "rel_l2": float(error),
                        })
                loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )
                for mode, seed, model in models:
                    offset = 0
                    for batch in loader:
                        batch = to_device(batch, device)
                        errors = instance_rel(model(batch), batch["uq"]).cpu().numpy()
                        for index, error in enumerate(errors):
                            rows.append({
                                "method": mode, "seed": seed, "kappa": kappa,
                                "kappa_regime": REGIME[kappa], "query_margin": margin,
                                "n_b": n_b, "sample_id": offset + index,
                                "rel_l2": float(error),
                            })
                        offset += len(errors)
                print(
                    f"[eval] kappa={kappa:g} regime={REGIME[kappa]} "
                    f"margin={margin:g} nb={n_b}",
                    flush=True,
                )
    return rows, training


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["method"], row["kappa"], row["kappa_regime"],
            row["query_margin"], row["n_b"],
        )].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        method, kappa, regime, margin, n_b = key
        per_sample = defaultdict(list)
        per_seed = defaultdict(list)
        for row in group:
            per_sample[row["sample_id"]].append(row["rel_l2"])
            per_seed[row["seed"]].append(row["rel_l2"])
        values = np.asarray([np.mean(v) for _, v in sorted(per_sample.items())])
        seed_means = np.asarray([np.mean(v) for _, v in sorted(per_seed.items())])
        ci = bootstrap_mean(values, 20260722 + int(round(100 * kappa)) + n_b)
        output.append({
            "method": method, "kappa": kappa, "kappa_regime": regime,
            "query_margin": margin, "n_b": n_b,
            "n_instances": len(values),
            "n_seeds": 0 if method.endswith("bie") else len(per_seed),
            "mean_rel_l2": float(values.mean()),
            "median_rel_l2": float(np.median(values)),
            "p90_rel_l2": float(np.quantile(values, 0.9)),
            "std_instances": float(values.std(ddof=1)),
            "seed_mean_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
        })
    return output


def averaged_by_geometry(rows, method, regime, margin, n_b):
    grouped = defaultdict(list)
    for row in rows:
        if (row["method"] == method and row["kappa_regime"] == regime
                and abs(row["query_margin"] - margin) < 1e-12
                and row["n_b"] == n_b):
            grouped[row["sample_id"]].append(row["rel_l2"])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_regime(rows, target, control, regime, margin, n_b, seed):
    target_values = averaged_by_geometry(rows, target, regime, margin, n_b)
    control_values = averaged_by_geometry(rows, control, regime, margin, n_b)
    keys = sorted(set(target_values) & set(control_values))
    differences = np.asarray([target_values[key] - control_values[key] for key in keys])
    target_mean = float(np.mean([target_values[key] for key in keys]))
    control_mean = float(np.mean([control_values[key] for key in keys]))
    return {
        "target": target, "control": control, "kappa_regime": regime,
        "query_margin": margin, "n_b": n_b,
        "n_geometry_clusters": len(keys),
        "target_mean_rel_l2": target_mean,
        "control_mean_rel_l2": control_mean,
        "relative_improvement": (control_mean - target_mean) / control_mean,
        "target_win_fraction": float(np.mean(differences < 0)),
        "mean_difference": float(differences.mean()),
        "bootstrap_mean_difference_ci": bootstrap_mean(differences, seed),
    }


def analyze(rows, target):
    comparisons = []
    seed = 20260800
    for regime in ("seen", "interpolation", "extrapolation"):
        for margin in MARGINS:
            for n_b in (128, 256):
                for control in (
                    "none", "full", "local_both", "local_potential", "adaptive"
                ):
                    if control == target:
                        continue
                    comparisons.append(paired_regime(
                        rows, target, control, regime, margin, n_b, seed
                    ))
                    seed += 1
    return {
        "protocol": {
            "train_kappas": [0.0, 1.0, 2.0, 4.0],
            "interpolation_kappas": [0.25, 0.5, 1.5, 3.0],
            "extrapolation_kappas": [5.0, 6.0],
            "selection": "No method selection on test results; local_tau=0.32 inherited from fixed-kappa validation-only selection.",
            "paired_unit": "geometry cluster; each cluster averages kappas and three training seeds",
            "target_method": target,
        },
        "paired_regime_comparisons": comparisons,
    }


def make_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = defaultdict(dict)
    for row in summary:
        lookup[(row["query_margin"], row["n_b"], row["method"])][row["kappa"]] = row
    methods = (
        ("correct_yukawa_bie", "Correct BIE", "black", "--"),
        ("fixed_laplace_bie", "Fixed Laplace BIE", "#999999", ":"),
        ("full", "Full Laplace prior", "#d95f02", "-"),
        ("none", "No prior", "#7570b3", "-"),
        ("local_both", "Local prior (both)", "#66a61e", "-"),
        ("local_potential", "Local prior (potential)", "#1b9e77", "-"),
        ("adaptive", "Adaptive trust gate", "#e7298a", "-"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for ax, margin, n_b, title in (
        (axes[0], 0.12, 128, "Bulk queries, train resolution"),
        (axes[1], 0.12, 256, "Bulk queries, refined boundary"),
        (axes[2], 0.02, 256, "Near-boundary queries"),
    ):
        for method, label, color, linestyle in methods:
            cell = lookup[(margin, n_b, method)]
            x = sorted(cell)
            y = [cell[value]["mean_rel_l2"] for value in x]
            ax.plot(x, y, marker="o", markersize=3.5, label=label,
                    color=color, linestyle=linestyle, linewidth=1.6)
        ax.axvspan(4.0, 6.0, color="#eeeeee", zorder=-2)
        ax.axvline(4.0, color="#bbbbbb", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("relative L2 error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/kappa_conditioned")
    parser.add_argument("--data-dir", default="data_cache/kappa_conditioned")
    parser.add_argument("--out-dir", default="results/kappa_conditioned")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target", default="local_potential")
    parser.add_argument("--figure", default="figures/kappa_conditioned_transfer.png")
    args = parser.parse_args()
    configure_torch()
    device = get_device(True)
    run_root = ROOT / args.run_root
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.out_dir
    rows, training = evaluate(run_root, data_dir, args.seeds, args.batch_size, device)
    summary = summarize(rows)
    analysis = analyze(rows, args.target)
    save_csv(rows, out_dir / "raw.csv")
    save_csv(summary, out_dir / "summary.csv")
    save_csv(training, out_dir / "training_summary.csv")
    save_json(analysis, out_dir / "analysis.json")
    make_figure(summary, ROOT / args.figure)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
