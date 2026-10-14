#!/usr/bin/env python
"""Post-confirmation diagnostic for physical-law recovery from solution supervision."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import kn
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402

RESULT = ROOT / "results" / "screening_law" / "confirmatory_v1"
DATA = ROOT / "data_cache" / "screening_confirmatory_v1"
FIGURE = ROOT / "figures" / "screening_law_confirmatory_v1_recovery.png"
PRIMARY_KAPPA = (1.35, 1.9, 2.55, 3.25, 3.9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_law(s: np.ndarray) -> np.ndarray:
    output = s * kn(1, np.maximum(s, 1e-12))
    output[s == 0] = 1.0
    return output


def empirical_scaled_distances() -> np.ndarray:
    values = []
    for seed_index in range(3):
        shard = torch.load(
            DATA / f"data_s{seed_index}" / "yukawa_train.pt",
            map_location="cpu", weights_only=False,
        )
        for sample_index in range(0, 600, 3):
            boundary = shard["Xb"][sample_index].numpy()
            query = shard["Xq"][sample_index].numpy()
            kappa = float(shard["kappa"][sample_index])
            bb = np.linalg.norm(
                boundary[::4, None] - boundary[None, ::4], axis=-1,
            )
            off_diagonal = bb[~np.eye(len(bb), dtype=bool)]
            qb = np.linalg.norm(
                query[::8, None] - boundary[None, ::4], axis=-1,
            ).ravel()
            values.extend((kappa * off_diagonal, kappa * qb))
    return np.concatenate(values)


def load_models():
    manifest = json.loads((RESULT / "checkpoint_manifest.json").read_text())
    models = []
    for record in manifest["records"]:
        if record["method"] not in {"screening_mix8", "radial_mlp"}:
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        models.append((record["method"], record["seed_index"], model.eval()))
    if len(models) != 6:
        raise RuntimeError(f"expected six screening checkpoints, found {len(models)}")
    return models


def primary_solution_errors() -> dict[tuple[str, int], float]:
    grouped = defaultdict(list)
    for family in ("in_dist", "highmode", "rough", "deform"):
        with (RESULT / "raw" / f"{family}.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["method"] not in {"screening_mix8", "radial_mlp"}:
                    continue
                if float(row["kappa"]) not in PRIMARY_KAPPA:
                    continue
                if float(row["query_margin"]) != 0.12 or int(row["n_boundary"]) != 128:
                    continue
                grouped[(row["method"], int(row["seed_index"]))].append(
                    float(row["rel_l2"])
                )
    return {key: float(np.mean(value)) for key, value in grouped.items()}


def main():
    audit = json.loads((RESULT / "independent_audit.json").read_text())
    if audit["corrected_verdict"] != "PAPERIZE":
        raise RuntimeError("law-recovery diagnostic is locked behind a passing confirmation")

    empirical = empirical_scaled_distances()
    q99 = float(np.quantile(empirical, 0.99))
    maximum = float(empirical.max())
    grid = np.linspace(0.0, max(18.0, 1.5 * maximum), 1200)
    exact = exact_law(grid)
    bins = np.linspace(0.0, maximum, 41)
    counts, _ = np.histogram(empirical, bins=bins)
    positive = counts[counts > 0]
    common_threshold = float(np.median(positive))
    bin_index = np.clip(np.digitize(grid, bins) - 1, 0, len(counts) - 1)
    common = (grid <= maximum) & (counts[bin_index] >= common_threshold)
    rare = (grid <= maximum) & (counts[bin_index] < common_threshold)
    stable_relative = exact >= 1e-4

    rows = []
    report = {
        "evaluation_only": True,
        "exact_law": "m_star(s)=s*K1(s)",
        "empirical_s": {
            "count": int(len(empirical)),
            "q99": q99,
            "maximum": maximum,
            "common_bin_count_threshold": common_threshold,
        },
        "models": {},
    }
    law_errors = []
    solution_errors = primary_solution_errors()
    curves = defaultdict(list)
    for method, seed_index, model in load_models():
        with torch.no_grad():
            prediction = model.m_law(torch.from_numpy(grid).float()).numpy()
        relative = np.abs(prediction - exact) / np.maximum(exact, 1e-12)
        in_support = (grid <= maximum) & stable_relative
        common_mask = common & stable_relative
        rare_mask = rare & stable_relative
        summary = {
            "median_relative_error_support": float(np.median(relative[in_support])),
            "max_relative_error_support": float(np.max(relative[in_support])),
            "median_relative_error_common": float(np.median(relative[common_mask])),
            "median_relative_error_rare": float(np.median(relative[rare_mask])),
            "median_relative_error_q99": float(np.median(relative[(grid <= q99) & stable_relative])),
            "max_absolute_tail_error": float(np.max(np.abs(prediction[grid > maximum] - exact[grid > maximum]))),
            "solution_primary_rel_l2": solution_errors[(method, seed_index)],
        }
        report["models"][f"{method}_s{seed_index}"] = summary
        law_errors.append(summary["median_relative_error_q99"])
        curves[method].append(prediction)
        for s, learned, target in zip(grid, prediction, exact):
            rows.append({
                "method": method,
                "seed_index": seed_index,
                "s": float(s),
                "learned_law": float(learned),
                "exact_law": float(target),
            })

    ordered_solution = [
        report["models"][key]["solution_primary_rel_l2"]
        for key in report["models"]
    ]
    report["law_solution_correlation"] = {
        "pearson": float(pearsonr(law_errors, ordered_solution).statistic),
        "spearman": float(spearmanr(law_errors, ordered_solution).statistic),
        "n": len(law_errors),
        "interpretation": "descriptive only; six method-seed points",
    }

    for method, values in curves.items():
        stack = np.asarray(values)
        report["models"][f"{method}_cross_seed"] = {
            "max_standard_deviation_support": float(stack[:, grid <= maximum].std(axis=0).max()),
        }

    csv_path = RESULT / "law_recovery_curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (RESULT / "law_recovery.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(grid, exact, color="black", linewidth=2.0, label="exact")
    for method, color in (("screening_mix8", "#0072B2"), ("radial_mlp", "#D55E00")):
        stack = np.asarray(curves[method])
        axes[0].plot(grid, stack.mean(0), color=color, label=method)
        axes[0].fill_between(
            grid, stack.min(0), stack.max(0), color=color, alpha=0.16,
        )
    axes[0].axvline(q99, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set(xlabel="scaled distance s", ylabel="screening law m(s)", ylim=(-0.05, 1.05))
    axes[0].legend(frameon=False)
    axes[1].hist(empirical, bins=40, color="#666666", alpha=0.8)
    axes[1].axvline(q99, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(xlabel="training scaled distance s", ylabel="sampled pair count")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
