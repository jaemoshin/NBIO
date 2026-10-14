#!/usr/bin/env python
"""Select and evaluate the local-singularity prior without test-set tuning.

Modes:
  tuning   summarize all seed-0 cutoff/scope runs and select by validation only;
  transfer evaluate the selected three-seed model and frozen split/no-split controls.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402


RESULT_ROOT = ROOT / "results/singular_prior_utility"
TRANSFER_DATA = ROOT / "data_cache/tmlr_revision"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
MARGINS = (0.12, 0.02)
NBS = (64, 128, 256)
CONTROL_CKPTS = {
    "full_split": (
        ROOT / "results/tmlr_revision/nbio_yuk_split_s0/best.pt",
        ROOT / "results/nbio_yuk_split_s1/best.pt",
        ROOT / "results/nbio_yuk_split_s2/best.pt",
    ),
    "no_split": (
        ROOT / "results/tmlr_revision/nbio_yuk_nosplit_s0/best.pt",
        ROOT / "results/nbio_yuk_nosplit_s1/best.pt",
        ROOT / "results/nbio_yuk_nosplit_s2/best.pt",
    ),
}


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def save_csv(rows, path):
    if not rows:
        raise ValueError("refusing to write an empty table")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, seed=20260722, n_boot=4000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for index in range(n_boot):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def instance_rel(prediction, target):
    numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
    return numerator / denominator


def checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["config"]


def summarize_tuning(expected=None):
    rows = []
    for directory in sorted(RESULT_ROOT.glob("tune_*_s0")):
        history_path = directory / "history.json"
        checkpoint_path = directory / "best.pt"
        if not history_path.exists() or not checkpoint_path.exists():
            continue
        history = json.loads(history_path.read_text())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        kwargs = checkpoint["config"]["model"]["kwargs"]
        evaluation_path = directory / "eval.json"
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
        test_value = evaluation.get("eval", {}).get("yukawa_test", {}).get("rel_l2", float("nan"))
        rows.append({
            "run": directory.name,
            "prior_scope": kwargs["prior_scope"],
            "prior_cutoff": float(kwargs["prior_cutoff"]),
            "prior_power": float(kwargs["prior_power"]),
            "seed": int(checkpoint["config"]["seed"]),
            "best_epoch": int(checkpoint["epoch"]),
            "best_val_rel_l2": float(history["best_val_rel_l2"]),
            "test_rel_l2_not_used_for_selection": float(test_value),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        })
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"expected {expected} completed tuning runs, found {len(rows)}")
    if not rows:
        raise RuntimeError("no completed tuning runs found")
    rows.sort(key=lambda row: row["best_val_rel_l2"])
    selected = rows[0]
    payload = {
        "selection_metric": "minimum best validation relative L2; test metric ignored",
        "n_candidates": len(rows),
        "selected": selected,
        "all_candidates": rows,
    }
    save_csv(rows, RESULT_ROOT / "tuning_summary.csv")
    save_json(payload, RESULT_ROOT / "selection.json")
    print(json.dumps(payload, indent=2))


def transfer_shard(family, margin, n_b):
    tag = f"m{int(round(1000 * margin)):03d}"
    return TRANSFER_DATA / f"yukawa_k4_{family}_{tag}_nb{n_b}.pt"


@torch.no_grad()
def evaluate_models(local_checkpoints, batch_size, with_qbx=False):
    configure_torch()
    device = get_device(True)
    specs = []
    for method, paths in CONTROL_CKPTS.items():
        for seed, path in enumerate(paths):
            specs.append((method, seed, path))
    for seed, path in enumerate(local_checkpoints):
        specs.append(("local_singular", seed, Path(path)))
    for _, _, path in specs:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    models = [(method, seed, *checkpoint_model(path, device)) for method, seed, path in specs]
    if with_qbx:
        qbx_models = []
        for method, seed, path in specs:
            if method not in {"full_split", "local_singular"}:
                continue
            model, config = checkpoint_model(path, device)
            model.qbx = True
            qbx_models.append((f"{method}_qbx", seed, model, config))
        models.extend(qbx_models)

    rows = []
    for family in FAMILIES:
        for margin in MARGINS:
            for n_b in NBS:
                path = transfer_shard(family, margin, n_b)
                dataset = BIEDataset(path)
                for method, tensor_key in (
                    ("correct_yukawa_bie", "bem_uq"),
                    ("fixed_laplace_bie", "laplace_bem_uq"),
                ):
                    errors = instance_rel(dataset.d[tensor_key], dataset.d["uq"]).numpy()
                    for sample_id, error in enumerate(errors):
                        rows.append({
                            "method": method, "seed": -1, "family": family,
                            "query_margin": margin, "n_b": n_b,
                            "sample_id": sample_id, "rel_l2": float(error),
                        })
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )
                for method, seed, model, _ in models:
                    offset = 0
                    for batch in loader:
                        batch = to_device(batch, device)
                        errors = instance_rel(model(batch), batch["uq"]).cpu().numpy()
                        for index, error in enumerate(errors):
                            rows.append({
                                "method": method, "seed": seed, "family": family,
                                "query_margin": margin, "n_b": n_b,
                                "sample_id": offset + index, "rel_l2": float(error),
                            })
                        offset += len(errors)
                    print(f"[transfer] {method} s{seed} {family} m={margin} nb={n_b}", flush=True)
    return rows


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["family"], row["query_margin"], row["n_b"])].append(row)
    summary = []
    for (method, family, margin, n_b), group in sorted(grouped.items()):
        samples = defaultdict(list)
        seeds = defaultdict(list)
        for row in group:
            samples[row["sample_id"]].append(row["rel_l2"])
            seeds[row["seed"]].append(row["rel_l2"])
        sample_values = np.asarray([np.mean(value) for _, value in sorted(samples.items())])
        seed_means = np.asarray([np.mean(value) for _, value in sorted(seeds.items())])
        ci = bootstrap_ci(sample_values)
        summary.append({
            "method": method,
            "family": family,
            "query_margin": margin,
            "n_b": n_b,
            "n_instances": len(sample_values),
            "n_seeds": len(seeds) if method not in {"correct_yukawa_bie", "fixed_laplace_bie"} else 0,
            "mean_rel_l2": float(sample_values.mean()),
            "median_rel_l2": float(np.median(sample_values)),
            "p90_rel_l2": float(np.quantile(sample_values, 0.9)),
            "std_instances": float(sample_values.std(ddof=1)),
            "seed_mean_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "bootstrap_ci_low": ci[0],
            "bootstrap_ci_high": ci[1],
        })
    return summary


def paired_analysis(rows, summary):
    learned = {
        "full_split", "no_split", "local_singular",
        "full_split_qbx", "local_singular_qbx",
    }
    by_key = defaultdict(dict)
    for row in rows:
        if row["method"] in learned:
            key = (
                row["family"], row["query_margin"], row["n_b"],
                row["sample_id"], row["seed"],
            )
            by_key[key][row["method"]] = row["rel_l2"]

    pairwise = {}
    controls = [method for method in ("full_split", "no_split", "full_split_qbx")
                if any(method in values for values in by_key.values())]
    target = "local_singular_qbx" if any(
        "local_singular_qbx" in values for values in by_key.values()
    ) else "local_singular"
    for control in controls:
        differences = []
        wins = []
        for values in by_key.values():
            if target in values and control in values:
                delta = values[target] - values[control]
                differences.append(delta)
                wins.append(delta < 0)
        pairwise[control] = {
            "n": len(differences),
            "mean_local_minus_control": float(np.mean(differences)),
            "median_local_minus_control": float(np.median(differences)),
            "local_win_fraction": float(np.mean(wins)),
            "bootstrap_mean_difference_ci": bootstrap_ci(differences),
        }

    lookup = {
        (row["method"], row["family"], float(row["query_margin"]), int(row["n_b"])): row
        for row in summary
    }
    slices = {}
    for name, margin in (("bulk", 0.12), ("near", 0.02)):
        slices[name] = {}
        methods = ["full_split", "no_split", "local_singular"]
        if target == "local_singular_qbx":
            methods += ["full_split_qbx", "local_singular_qbx"]
        for method in methods:
            values = [lookup[(method, family, margin, 256)]["mean_rel_l2"] for family in FAMILIES]
            slices[name][method] = float(np.mean(values))
    near_family_wins = {
        control: sum(
            lookup[(target, family, 0.02, 256)]["mean_rel_l2"]
            < lookup[(control, family, 0.02, 256)]["mean_rel_l2"]
            for family in FAMILIES
        )
        for control in controls
    }
    local_bulk = slices["bulk"][target]
    nosplit_bulk = slices["bulk"]["no_split"]
    strong_success = (
        local_bulk <= 1.10 * nosplit_bulk
        and near_family_wins["full_split"] == len(FAMILIES)
        and near_family_wins["no_split"] == len(FAMILIES)
    )
    good_success = (
        local_bulk < slices["bulk"]["full_split"]
        and near_family_wins["no_split"] >= 3
    )
    verdict = "strong" if strong_success else "good" if good_success else "failed"
    return {
        "pairwise_all_slices": pairwise,
        "n_b_256_family_mean": slices,
        "near_n_b_256_family_wins": near_family_wins,
        "paired_target": target,
        "success_rule": {
            "strong": "bulk <= 1.10x no-split and wins all four near families against both controls",
            "good": "bulk beats full split and wins at least three near families against no-split",
            "result": verdict,
        },
    }


def make_figure(summary, output_name="singular_prior_utility.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_qbx = any(row["method"] == "local_singular_qbx" for row in summary)
    methods = ["full_split", "no_split", "local_singular"]
    if has_qbx:
        methods += ["full_split_qbx", "local_singular_qbx"]
    colors = {
        "full_split": "#0072B2", "no_split": "#D55E00", "local_singular": "#009E73",
        "full_split_qbx": "#56B4E9", "local_singular_qbx": "#006D4F",
    }
    labels = {
        "full_split": "full Laplace prior", "no_split": "no prior",
        "local_singular": "local singular prior", "full_split_qbx": "full prior + QBX",
        "local_singular_qbx": "local singular prior + QBX",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=True)
    for axis, (title, margin) in zip(axes, (("bulk queries", 0.12), ("near-boundary queries", 0.02))):
        for method in methods:
            values = []
            lows = []
            highs = []
            for n_b in NBS:
                selected = [
                    row for row in summary
                    if row["method"] == method and row["query_margin"] == margin and row["n_b"] == n_b
                ]
                values.append(np.mean([row["mean_rel_l2"] for row in selected]))
                lows.append(np.mean([row["bootstrap_ci_low"] for row in selected]))
                highs.append(np.mean([row["bootstrap_ci_high"] for row in selected]))
            axis.plot(NBS, values, marker="o", color=colors[method], label=labels[method])
            axis.fill_between(NBS, lows, highs, color=colors[method], alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("boundary nodes")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("mean relative L2 across geometry families")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / f"figures/{output_name}", dpi=200)


def transfer(local_checkpoints, batch_size, with_qbx=False):
    if len(local_checkpoints) != 3:
        raise ValueError("exactly three local-singular checkpoints are required")
    rows = evaluate_models(local_checkpoints, batch_size, with_qbx=with_qbx)
    summary = summarize_rows(rows)
    analysis = paired_analysis(rows, summary)
    selection = json.loads((RESULT_ROOT / "selection.json").read_text())
    payload = {
        "protocol": {
            "selection": "cutoff and scope selected on seed-0 validation only",
            "kappa": 4.0,
            "train_resolution": 128,
            "families": list(FAMILIES),
            "query_margins": list(MARGINS),
            "eval_resolutions": list(NBS),
            "local_checkpoints": [str(Path(path)) for path in local_checkpoints],
        },
        "selection": selection["selected"],
        "analysis": analysis,
    }
    prefix = "transfer_qbx" if with_qbx else "transfer"
    save_csv(rows, RESULT_ROOT / f"{prefix}_raw.csv")
    save_csv(summary, RESULT_ROOT / f"{prefix}_summary.csv")
    save_json(payload, RESULT_ROOT / f"{prefix}_analysis.json")
    make_figure(summary, f"singular_prior_utility{'_qbx' if with_qbx else ''}.png")
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    tuning_parser = subparsers.add_parser("tuning")
    tuning_parser.add_argument("--expected", type=int)
    transfer_parser = subparsers.add_parser("transfer")
    transfer_parser.add_argument("--local-checkpoints", nargs="+", required=True)
    transfer_parser.add_argument("--batch-size", type=int, default=16)
    qbx_parser = subparsers.add_parser("transfer-qbx")
    qbx_parser.add_argument("--local-checkpoints", nargs="+", required=True)
    qbx_parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "tuning":
        summarize_tuning(args.expected)
    elif args.mode == "transfer":
        transfer(args.local_checkpoints, args.batch_size)
    else:
        transfer(args.local_checkpoints, args.batch_size, with_qbx=True)


if __name__ == "__main__":
    main()
