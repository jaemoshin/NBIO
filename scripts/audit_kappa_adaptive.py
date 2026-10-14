#!/usr/bin/env python
"""Audit the learned monotone trust-gate follow-up without hiding fixed-gate failures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/kappa_conditioned_adaptive"
REPORT = ROOT / "reports/kappa_adaptive_results.md"


def rows(path):
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
        raise RuntimeError(f"missing unique cell {method}/{kappa}/{margin}/{n_b}")
    return found[0]


def comparison(analysis, control, regime, margin, n_b):
    found = [row for row in analysis["paired_regime_comparisons"] if (
        row["target"] == "adaptive" and row["control"] == control
        and row["kappa_regime"] == regime
        and abs(float(row["query_margin"]) - margin) < 1e-12
        and int(row["n_b"]) == n_b
    )]
    if len(found) != 1:
        raise RuntimeError(f"missing unique comparison {control}/{regime}/{margin}/{n_b}")
    return found[0]


def significant_better(item):
    return item["bootstrap_mean_difference_ci"][1] < 0


def describe(item):
    change = 100 * item["relative_improvement"]
    direction = f"{change:.1f}% lower" if change >= 0 else f"{-change:.1f}% higher"
    ci = item["bootstrap_mean_difference_ci"]
    significance = "excludes zero" if ci[1] < 0 or ci[0] > 0 else "includes zero"
    return direction, significance


def main():
    raw = rows(RESULT / "raw.csv")
    summary = rows(RESULT / "summary.csv")
    training = rows(RESULT / "training_summary.csv")
    analysis = json.loads((RESULT / "analysis.json").read_text())
    expected = 2 * 10 * 2 * 3 * 64 + 5 * 3 * 10 * 2 * 3 * 64
    keys = [tuple(row[name] for name in (
        "method", "seed", "kappa", "query_margin", "n_b", "sample_id"
    )) for row in raw]
    if len(raw) != expected or len(keys) != len(set(keys)):
        raise RuntimeError(f"integrity failure: rows={len(raw)} expected={expected}")
    if any(not np.isfinite(float(row["rel_l2"])) for row in raw):
        raise RuntimeError("non-finite result")
    if len(training) != 15 or {row["method"] for row in training} != {
        "full", "none", "local_both", "local_potential", "adaptive"
    }:
        raise RuntimeError("incomplete training matrix")

    primary = {}
    for regime in ("seen", "interpolation", "extrapolation"):
        for margin, n_b, label in ((0.12, 128, "bulk"), (0.02, 256, "near")):
            for control in ("none", "full", "local_both", "local_potential"):
                primary[f"{regime}_{label}_adaptive_vs_{control}"] = comparison(
                    analysis, control, regime, margin, n_b
                )

    interpolation_bulk = primary["interpolation_bulk_adaptive_vs_none"]
    interpolation_near = primary["interpolation_near_adaptive_vs_none"]
    extrapolation_bulk = primary["extrapolation_bulk_adaptive_vs_none"]
    if significant_better(interpolation_bulk) and significant_better(interpolation_near):
        verdict = "PAPERIZE"
    elif significant_better(interpolation_bulk) or significant_better(interpolation_near):
        verdict = "NARROW"
    else:
        verdict = "KILL"

    gate_rates = [float(row["learned_gate_rate"]) for row in training
                  if row["method"] == "adaptive"]
    fallback_gap = 0.0
    for margin in (0.12, 0.02):
        for n_b in (64, 128, 256):
            adaptive = float(cell(summary, "adaptive", 0.0, margin, n_b)["mean_rel_l2"])
            classical = float(cell(
                summary, "correct_yukawa_bie", 0.0, margin, n_b
            )["mean_rel_l2"])
            fallback_gap = max(fallback_gap, abs(adaptive - classical))

    audit = {
        "integrity": {
            "raw_rows": len(raw), "expected_rows": expected,
            "duplicate_keys": 0, "nonfinite_values": 0,
            "training_runs": len(training), "seeds_per_method": 3,
            "test_geometries_per_cell": 64,
        },
        "learned_gate_rates": gate_rates,
        "max_kappa_zero_gap_from_classical": fallback_gap,
        "primary_comparisons": primary,
        "verdict": verdict,
    }
    (RESULT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Learned trust-gate NBIO across kappa",
        "",
        "## What changed",
        "",
        "The fixed local gates failed between the `kappa=0` and `kappa=1` training anchors. This follow-up",
        "learns one positive monotone decay rate `a` in",
        "",
        "```text",
        "alpha(kappa,r) = exp(-a kappa^2)",
        "                 + (1-exp(-a kappa^2)) exp(-(kappa r/0.32)^4).",
        "```",
        "",
        "It retains the exact Laplace fallback and local high-mismatch singular structure. The failed",
        "fixed-gate results remain in the comparison rather than being removed.",
        "",
        f"Learned gate rates over three seeds: `{', '.join(f'{x:.3f}' for x in gate_rates)}`.",
        "",
        "## Bulk results at n_b=128",
        "",
        "| kappa | regime | correct BIE | fixed BIE | full | no prior | local both | local potential | adaptive |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    regime = {0.0: "seen", 0.25: "interp", 0.5: "interp", 1.0: "seen",
              1.5: "interp", 2.0: "seen", 3.0: "interp", 4.0: "seen",
              5.0: "extra", 6.0: "extra"}
    methods = ("correct_yukawa_bie", "fixed_laplace_bie", "full", "none",
               "local_both", "local_potential", "adaptive")
    for kappa in regime:
        values = [fmt(cell(summary, method, kappa, 0.12, 128)["mean_rel_l2"])
                  for method in methods]
        lines.append(
            f"| `{kappa:g}` | {regime[kappa]} | "
            + " | ".join(f"`{value}`" for value in values) + " |"
        )

    lines += ["", "## Primary paired audit", ""]
    for name, item in primary.items():
        direction, significance = describe(item)
        ci = item["bootstrap_mean_difference_ci"]
        lines.append(
            f"- `{name}`: **{direction}**; geometry-cluster 95% CI "
            f"`[{fmt(ci[0])}, {fmt(ci[1])}]` ({significance}); adaptive win fraction "
            f"`{100 * item['target_win_fraction']:.1f}%`."
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        f"The maximum aggregate `kappa=0` gap from the classical BIE is `{fmt(fallback_gap)}`.",
        "Interpolation and out-of-range extrapolation remain separate claims. Correct-kernel BIE is the",
        "numerical ceiling, and a family checkpoint is not compared as if it were a fixed-parameter specialist.",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
