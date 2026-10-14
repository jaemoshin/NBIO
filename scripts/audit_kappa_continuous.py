#!/usr/bin/env python
"""Integrity and claim audit for the fresh continuous-kappa holdout."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/kappa_continuous_eval"
REPORT = ROOT / "reports/kappa_continuous_results.md"


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def fmt(value):
    return f"{float(value):.3e}"


def cell(summary, method, kappa, margin, n_b):
    found = [row for row in summary if (
        row["method"] == method and abs(float(row["kappa"]) - kappa) < 1e-12
        and abs(float(row["query_margin"]) - margin) < 1e-12
        and int(row["n_b"]) == n_b
    )]
    if len(found) != 1:
        raise RuntimeError(f"missing cell {method}/{kappa}/{margin}/{n_b}")
    return found[0]


def comparison(analysis, control, regime, margin, n_b):
    found = [row for row in analysis["paired_regime_comparisons"] if (
        row["target"] == "adaptive_sat" and row["control"] == control
        and row["kappa_regime"] == regime
        and abs(float(row["query_margin"]) - margin) < 1e-12
        and int(row["n_b"]) == n_b
    )]
    if len(found) != 1:
        raise RuntimeError(f"missing comparison {control}/{regime}/{margin}/{n_b}")
    return found[0]


def aggregate(raw, method, regime, margin, n_b):
    grouped = defaultdict(list)
    for row in raw:
        if (row["method"] == method and row["kappa_regime"] == regime
                and abs(float(row["query_margin"]) - margin) < 1e-12
                and int(row["n_b"]) == n_b):
            grouped[int(row["sample_id"])].append(float(row["rel_l2"]))
    values = np.asarray([np.mean(value) for _, value in sorted(grouped.items())])
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)), "n_geometry_clusters": len(values),
    }


def significant_better(item):
    return item["bootstrap_mean_difference_ci"][1] < 0


def describe(item):
    change = 100 * item["relative_improvement"]
    direction = f"{change:.1f}% lower" if change >= 0 else f"{-change:.1f}% higher"
    ci = item["bootstrap_mean_difference_ci"]
    significance = "excludes zero" if ci[1] < 0 or ci[0] > 0 else "includes zero"
    return direction, significance


def main():
    raw = read_csv(RESULT / "raw.csv")
    summary = read_csv(RESULT / "summary.csv")
    training = read_csv(RESULT / "training_summary.csv")
    analysis = json.loads((RESULT / "analysis.json").read_text())
    manifest = json.loads((ROOT / "data_cache/kappa_continuous/manifest.json").read_text())

    expected = 2 * 11 * 2 * 3 * 64 + 5 * 3 * 11 * 2 * 3 * 64
    keys = [tuple(row[name] for name in (
        "method", "seed", "kappa", "query_margin", "n_b", "sample_id"
    )) for row in raw]
    if len(raw) != expected or len(keys) != len(set(keys)):
        raise RuntimeError(f"integrity failure rows={len(raw)} expected={expected}")
    if any(not np.isfinite(float(row["rel_l2"])) for row in raw):
        raise RuntimeError("non-finite result")
    expected_methods = {"none_norm", "full_norm", "full_sat", "adaptive_norm", "adaptive_sat"}
    if len(training) != 15 or {row["method"] for row in training} != expected_methods:
        raise RuntimeError("incomplete training matrix")
    if manifest["seeds"]["test"] != 9203:
        raise RuntimeError("unexpected holdout seed")

    primary = {}
    for regime in ("interpolation", "extrapolation"):
        for margin, n_b, label in ((0.12, 128, "bulk"), (0.02, 256, "near")):
            for control in ("none_norm", "full_norm", "full_sat", "adaptive_norm"):
                primary[f"{regime}_{label}_adaptive_sat_vs_{control}"] = comparison(
                    analysis, control, regime, margin, n_b
                )
    interpolation_bulk = primary["interpolation_bulk_adaptive_sat_vs_none_norm"]
    interpolation_near = primary["interpolation_near_adaptive_sat_vs_none_norm"]
    extrapolation_bulk = primary["extrapolation_bulk_adaptive_sat_vs_none_norm"]

    resolution = {}
    for method in ("none_norm", "adaptive_sat"):
        for margin, label in ((0.12, "bulk"), (0.02, "near")):
            for n_b in (64, 128, 256):
                resolution[f"{method}_{label}_nb{n_b}"] = aggregate(
                    raw, method, "interpolation", margin, n_b
                )

    fallback_gap = 0.0
    for method in ("full_norm", "full_sat", "adaptive_norm", "adaptive_sat"):
        for margin in (0.12, 0.02):
            for n_b in (64, 128, 256):
                learned = float(cell(summary, method, 0.0, margin, n_b)["mean_rel_l2"])
                classical = float(cell(
                    summary, "correct_yukawa_bie", 0.0, margin, n_b
                )["mean_rel_l2"])
                fallback_gap = max(fallback_gap, abs(learned - classical))

    if significant_better(interpolation_bulk) and significant_better(interpolation_near):
        verdict = "PAPERIZE"
    elif (significant_better(interpolation_near)
          and interpolation_bulk["bootstrap_mean_difference_ci"][0] <= 0):
        verdict = "NARROW"
    else:
        verdict = "KILL"

    gate_rates = {
        method: [float(row["learned_gate_rate"]) for row in training
                 if row["method"] == method]
        for method in ("adaptive_norm", "adaptive_sat")
    }
    audit = {
        "integrity": {
            "raw_rows": len(raw), "expected_rows": expected,
            "duplicate_keys": 0, "nonfinite_values": 0,
            "training_runs": len(training), "seeds_per_method": 3,
            "test_geometries_per_cell": 64,
        },
        "fresh_holdout": {
            "test_seed": manifest["seeds"]["test"],
            "primary_interpolation_kappas": manifest["primary_interpolation_kappas"],
            "primary_extrapolation_kappas": manifest["primary_extrapolation_kappas"],
            "statement": manifest["freshness"],
        },
        "primary_comparisons": primary,
        "interpolation_resolution_slices": resolution,
        "learned_gate_rates": gate_rates,
        "max_kappa_zero_gap_from_classical": fallback_gap,
        "verdict": verdict,
    }
    (RESULT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    methods = ("correct_yukawa_bie", "fixed_laplace_bie", "none_norm", "full_norm",
               "full_sat", "adaptive_norm", "adaptive_sat")
    labels = ("correct BIE", "fixed BIE", "no prior", "full norm", "full sat",
              "adaptive norm", "adaptive sat")
    kappas = (0.0, 0.125, 0.375, 0.75, 1.25, 1.75, 2.5, 3.5, 4.0, 4.5, 5.5)
    regimes = {0.0: "endpoint", 4.0: "endpoint", 4.5: "extra", 5.5: "extra"}
    lines = [
        "# Continuous-kappa NBIO family result",
        "",
        "## Protocol integrity",
        "",
        "This follow-up was defined after the discrete-anchor failure and uses a new train/validation/test",
        "seed set. Its seven primary interpolation values and two extrapolation values were absent from the",
        "earlier inspected test. Training uses a continuous stratified distribution over `[0,4]`.",
        "",
        "## Bulk results at n_b=128",
        "",
        "| kappa | regime | " + " | ".join(labels) + " |",
        "|---:|---|" + "---:|" * len(labels),
    ]
    for kappa in kappas:
        regime = regimes.get(kappa, "interp")
        values = [fmt(cell(summary, method, kappa, 0.12, 128)["mean_rel_l2"])
                  for method in methods]
        lines.append(
            f"| `{kappa:g}` | {regime} | "
            + " | ".join(f"`{value}`" for value in values) + " |"
        )
    lines += ["", "## Primary paired audit", ""]
    for name, item in primary.items():
        direction, significance = describe(item)
        ci = item["bootstrap_mean_difference_ci"]
        lines.append(
            f"- `{name}`: **{direction}**; geometry-cluster 95% CI "
            f"`[{fmt(ci[0])}, {fmt(ci[1])}]` ({significance}); target win fraction "
            f"`{100 * item['target_win_fraction']:.1f}%`."
        )
    lines += [
        "",
        "## Resolution behavior on fresh interpolation values",
        "",
        "| method | slice | n_b=64 | n_b=128 | n_b=256 |",
        "|---|---|---:|---:|---:|",
    ]
    for method, label in (("none_norm", "no prior"), ("adaptive_sat", "adaptive sat")):
        for slice_name in ("bulk", "near"):
            values = [resolution[f"{method}_{slice_name}_nb{n_b}"]["mean"]
                      for n_b in (64, 128, 256)]
            lines.append(
                f"| {label} | {slice_name} | "
                + " | ".join(f"`{fmt(value)}`" for value in values) + " |"
            )
    lines += [
        "",
        "## Claim boundary",
        "",
        f"The largest aggregate `kappa=0` discrepancy from the classical BIE is `{fmt(fallback_gap)}`.",
        f"Out-of-range extrapolation is reported independently; adaptive-saturating versus no-prior bulk",
        f"extrapolation is **{describe(extrapolation_bulk)[0]}** and is not folded into the interpolation claim.",
        "The exact-kernel BIE remains the numerical ceiling, and near-boundary direct quadrature error is",
        "visible in that control rather than attributed to learning.",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
