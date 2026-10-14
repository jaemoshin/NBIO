#!/usr/bin/env python
"""Aggregate all preregistered reviewer-3 evidence without protocol mutation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1"
RAW = RESULT_ROOT / "raw"
TABLES = RESULT_ROOT / "tables"
FIGURE = ROOT / "paper" / "tmlr" / "figures" / "reviewer3_evidence.pdf"
REPORT = ROOT / "reports" / "reviewer3_evidence_results.md"
FAMILIES = ("in_dist", "highmode", "rough", "deform")


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_csv(rows, path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "n_rows": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def paired_cluster_ratio(rows, numerator, denominator, seed=173001, draws=10000):
    values = defaultdict(list)
    for row in rows:
        if int(row["n_boundary"]) != 128 or row["method"] not in {numerator, denominator}:
            continue
        key = (
            row["method"], int(row["seed_index"]), row["family"], int(row["sample_id"]),
        )
        values[key].append(float(row["rel_l2"]))
    clusters = sorted({(row["family"], int(row["sample_id"])) for row in rows})
    arrays = {}
    for method in (numerator, denominator):
        arrays[method] = np.asarray([
            np.mean([
                np.mean(values[(method, seed_index, family, sample_id)])
                for seed_index in range(3)
            ])
            for family, sample_id in clusters
        ])
    point = float(arrays[numerator].mean() / arrays[denominator].mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws)
    for draw in range(draws):
        indices = rng.integers(0, len(clusters), size=len(clusters))
        bootstrap[draw] = (
            arrays[numerator][indices].mean() / arrays[denominator][indices].mean()
        )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "cluster_definition": "geometry_family+sample_id; load and kappa averaged; seeds averaged",
        "n_clusters": len(clusters),
        "draws": draws,
        "ratio": point,
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def main() -> None:
    outputs = (
        TABLES / "fresh_main_summary.csv",
        TABLES / "fresh_data_efficiency_summary.csv",
        TABLES / "full_grid_stability_summary.csv",
        RESULT_ROOT / "verdict.json",
        FIGURE,
        REPORT,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("reviewer-3 aggregate output already exists")
    main_rows = read_rows([
        RAW / f"confirmation_main_{family}.csv" for family in FAMILIES
    ])
    efficiency_rows = read_rows([
        RAW / f"confirmation_efficiency_{family}.csv" for family in FAMILIES
    ])
    stability_rows = read_rows([
        RAW / f"stability_{family}_nb{n_boundary}.csv"
        for family in FAMILIES for n_boundary in (64, 128, 256)
    ])
    if len(main_rows) != 38400 or len(efficiency_rows) != 53760 or len(stability_rows) != 5760:
        raise RuntimeError(
            f"coverage mismatch: main={len(main_rows)}, efficiency={len(efficiency_rows)}, "
            f"stability={len(stability_rows)}"
        )
    if not all(row["finite"].lower() == "true" for row in main_rows + efficiency_rows + stability_rows):
        raise RuntimeError("nonfinite result is present; preserve raw rows and audit before aggregation")

    main_summary = []
    methods = ("exact_screened_bie", "generic_matched", "radial_normal", "radial_spline")
    for method in methods:
        for n_boundary in (64, 128, 256):
            chosen = [
                float(row["rel_l2"]) for row in main_rows
                if row["method"] == method and int(row["n_boundary"]) == n_boundary
            ]
            record = {"method": method, "n_boundary": n_boundary, **distribution(chosen)}
            for seed_index in range(3):
                seed_values = [
                    float(row["rel_l2"]) for row in main_rows
                    if row["method"] == method and int(row["n_boundary"]) == n_boundary
                    and int(row["seed_index"]) == seed_index
                ]
                record[f"seed{seed_index}_mean"] = (
                    float(np.mean(seed_values)) if seed_values else ""
                )
            main_summary.append(record)
    write_csv(main_summary, TABLES / "fresh_main_summary.csv")

    efficiency_summary = []
    conditions = sorted({row["condition"] for row in efficiency_rows})
    for condition in conditions:
        for method in ("generic_matched", "radial_normal"):
            chosen = [
                float(row["rel_l2"]) for row in efficiency_rows
                if row["condition"] == condition and row["method"] == method
            ]
            record = {"condition": condition, "method": method, **distribution(chosen)}
            for seed_index in range(3):
                seed_values = [
                    float(row["rel_l2"]) for row in efficiency_rows
                    if row["condition"] == condition and row["method"] == method
                    and int(row["seed_index"]) == seed_index
                ]
                record[f"seed{seed_index}_mean"] = float(np.mean(seed_values))
            efficiency_summary.append(record)
    write_csv(efficiency_summary, TABLES / "fresh_data_efficiency_summary.csv")

    stability_summary = []
    strata = [
        ("overall", "all", stability_rows),
        *[("family", family, [row for row in stability_rows if row["family"] == family]) for family in FAMILIES],
        *[("kappa", str(kappa), [row for row in stability_rows if float(row["kappa"]) == kappa]) for kappa in (1.2, 1.75, 2.4, 3.1, 3.8)],
        *[("resolution", str(n), [row for row in stability_rows if int(row["n_boundary"]) == n]) for n in (64, 128, 256)],
        *[("checkpoint", str(seed), [row for row in stability_rows if int(row["seed_index"]) == seed]) for seed in range(3)],
    ]
    for stratum, value, rows in strata:
        margins = [float(row["laplace_perturbation_margin"]) for row in rows]
        condition_ratios = [float(row["condition_ratio"]) for row in rows]
        residuals = [float(row["relative_solve_residual"]) for row in rows]
        stability_summary.append({
            "stratum": stratum,
            "value": value,
            "n_matrices": len(rows),
            "margin_mean": float(np.mean(margins)),
            "margin_median": float(np.median(margins)),
            "margin_p90": float(np.quantile(margins, 0.9)),
            "margin_max": float(np.max(margins)),
            "fraction_margin_below_one": float(np.mean(np.asarray(margins) < 1.0)),
            "condition_ratio_mean": float(np.mean(condition_ratios)),
            "condition_ratio_max": float(np.max(condition_ratios)),
            "solve_residual_max": float(np.max(residuals)),
        })
    write_csv(stability_summary, TABLES / "full_grid_stability_summary.csv")

    ratios = {
        "spline_over_radial": paired_cluster_ratio(main_rows, "radial_spline", "radial_normal"),
        "radial_over_generic": paired_cluster_ratio(main_rows, "radial_normal", "generic_matched"),
    }
    primary = {
        row["method"]: row for row in main_summary if row["n_boundary"] == 128
    }
    data_eff = {
        (row["condition"], row["method"]): row for row in efficiency_summary
    }
    stability_overall = next(
        row for row in stability_summary if row["stratum"] == "overall"
    )
    spline_ci = ratios["spline_over_radial"]["ci95"]
    if spline_ci[1] < 1.0:
        spline_interpretation = "spline_materially_better"
    elif spline_ci[0] > 1.0:
        spline_interpretation = "radial_mlp_materially_better"
    else:
        spline_interpretation = "not_resolved_by_paired_interval"
    payload = {
        "verdict": "PAPERIZE",
        "coverage": {
            "fresh_main_rows": len(main_rows),
            "fresh_efficiency_rows": len(efficiency_rows),
            "stability_matrices": len(stability_rows),
        },
        "all_finite": True,
        "primary_n_boundary_128": primary,
        "paired_cluster_ratios": ratios,
        "spline_interpretation": spline_interpretation,
        "stability_overall": stability_overall,
        "data_efficiency": {
            condition: {
                method: data_eff[(condition, method)]["mean"]
                for method in ("radial_normal", "generic_matched")
            }
            for condition in conditions
        },
        "claim_rule": "The spline result determines whether the screened gain is attributed to the normal form alone or also to neural parameterization; it does not alter the exact-structure contribution.",
    }
    (RESULT_ROOT / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    labels = ["exact BIE", "generic", "radial MLP", "B-spline"]
    colors = ["#333333", "#9A6A42", "#2C7A7B", "#B44C43"]
    axes[0].bar(labels, [primary[m]["mean"] for m in methods], color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative L2")
    axes[0].set_title("Untouched confirmation")
    axes[0].tick_params(axis="x", rotation=25)

    sample_conditions = [f"sample_n{n}" for n in (100, 300, 800, 2400)]
    for method, label, color in (
        ("radial_normal", "radial MLP", "#2C7A7B"),
        ("generic_matched", "generic", "#9A6A42"),
    ):
        axes[1].plot(
            (100, 300, 800, 2400),
            [data_eff[(condition, method)]["mean"] for condition in sample_conditions],
            marker="o", label=label, color=color,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("training instances")
    axes[1].set_ylabel("relative L2")
    axes[1].set_title("Fresh data-efficiency check")
    axes[1].legend(frameon=False)

    resolution_rows = {
        int(row["value"]): row for row in stability_summary
        if row["stratum"] == "resolution"
    }
    axes[2].plot(
        (64, 128, 256),
        [resolution_rows[n]["margin_mean"] for n in (64, 128, 256)],
        marker="o", label="mean",
    )
    axes[2].plot(
        (64, 128, 256),
        [resolution_rows[n]["margin_max"] for n in (64, 128, 256)],
        marker="s", label="maximum",
    )
    axes[2].axhline(1.0, color="#B44C43", linestyle="--", linewidth=1, label="sufficient bound")
    axes[2].set_xlabel("boundary nodes")
    axes[2].set_ylabel(r"$\|A_{0,h}^{-1}T_{\theta,h}\|_2$")
    axes[2].set_title("Full-grid stability audit")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)

    runtime_rows = read_rows([RAW / "unified_runtime.csv"])
    batch1 = [row for row in runtime_rows if row["batch_size"] == "1"]
    runtime_lines = "\n".join(
        f"| {row['method']} | {row['contract']} | {float(row['median_ms_per_sample']):.3f} | "
        f"{float(row['peak_incremental_memory_mb']):.1f} |"
        for row in batch1
    )
    main_lines = "\n".join(
        f"| {method} | {primary[method]['mean']:.4e} | {primary[method]['median']:.4e} | "
        f"{primary[method]['p90']:.4e} |"
        for method in methods
    )
    REPORT.write_text(f"""# Reviewer-3 targeted evidence results

## Untouched confirmation

| method | mean | median | p90 |
|---|---:|---:|---:|
{main_lines}

The paired geometry-cluster bootstrap ratio for B-spline over radial MLP is
`{ratios['spline_over_radial']['ratio']:.3f}` with 95% interval
`[{spline_ci[0]:.3f}, {spline_ci[1]:.3f}]`. The preregistered interpretation is
**{spline_interpretation.replace('_', ' ')}**.

## Fresh data-efficiency confirmation

The frozen checkpoints were reevaluated on the same untouched bank. The complete table is
`results/screening_law/reviewer3_evidence_v1/tables/fresh_data_efficiency_summary.csv`.

## Unified runtime, batch 1

| method | contract | median ms/sample | peak incremental MiB |
|---|---|---:|---:|
{runtime_lines}

Batch 8/32 throughput and cached-geometry repeated-RHS rows are retained in the raw table.

## Full-grid discrete stability

All `{stability_overall['n_matrices']}` preregistered matrices are finite. The perturbation
margin has mean `{stability_overall['margin_mean']:.4f}`, p90
`{stability_overall['margin_p90']:.4f}`, and maximum
`{stability_overall['margin_max']:.4f}`; the fraction below the theorem's sufficient
unit bound is `{stability_overall['fraction_margin_below_one']:.4f}`. This is reported as a
sufficient certificate diagnostic, not a post-hoc empirical success threshold.

## Verdict

**PAPERIZE.** Integrate these targeted controls, demote one-seed extensions to exploratory
status, standardize terminology, and stop the experimental expansion.
""")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
