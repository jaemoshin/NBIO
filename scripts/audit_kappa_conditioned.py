#!/usr/bin/env python
"""Integrity, uncertainty, and claim audit for the conditioned-kappa tranche."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/kappa_conditioned"
REPORT = ROOT / "reports/kappa_conditioned_results.md"


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def fmt(value):
    return f"{float(value):.3e}"


def cell(summary, method, kappa, margin, n_b):
    matches = [row for row in summary if (
        row["method"] == method
        and abs(float(row["kappa"]) - kappa) < 1e-12
        and abs(float(row["query_margin"]) - margin) < 1e-12
        and int(row["n_b"]) == n_b
    )]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one cell for {method}, k={kappa}, margin={margin}, nb={n_b}; "
            f"found {len(matches)}"
        )
    return matches[0]


def comparison(analysis, control, regime, margin, n_b):
    matches = [row for row in analysis["paired_regime_comparisons"] if (
        row["target"] == "local_potential" and row["control"] == control
        and row["kappa_regime"] == regime
        and abs(float(row["query_margin"]) - margin) < 1e-12
        and int(row["n_b"]) == n_b
    )]
    if len(matches) != 1:
        raise RuntimeError(f"missing paired comparison {control}/{regime}/{margin}/{n_b}")
    return matches[0]


def significant_better(item):
    return item["bootstrap_mean_difference_ci"][1] < 0


def direction(item):
    change = 100 * item["relative_improvement"]
    label = f"{change:.1f}% lower" if change >= 0 else f"{-change:.1f}% higher"
    ci = item["bootstrap_mean_difference_ci"]
    significance = "excludes zero" if ci[1] < 0 or ci[0] > 0 else "includes zero"
    return label, significance


def main():
    raw = read_csv(RESULT / "raw.csv")
    summary = read_csv(RESULT / "summary.csv")
    training = read_csv(RESULT / "training_summary.csv")
    analysis = json.loads((RESULT / "analysis.json").read_text())

    expected_rows = 2 * 10 * 2 * 3 * 64 + 4 * 3 * 10 * 2 * 3 * 64
    key_fields = ("method", "seed", "kappa", "query_margin", "n_b", "sample_id")
    keys = [tuple(row[key] for key in key_fields) for row in raw]
    if len(raw) != expected_rows:
        raise RuntimeError(f"raw row count {len(raw)} != {expected_rows}")
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate raw result key")
    if any(not np.isfinite(float(row["rel_l2"])) for row in raw):
        raise RuntimeError("non-finite relative error")
    if len(training) != 12:
        raise RuntimeError(f"expected twelve training runs, found {len(training)}")
    if {int(row["seed"]) for row in training} != {0, 1, 2}:
        raise RuntimeError("missing training seed")
    if {row["method"] for row in training} != {
        "full", "none", "local_both", "local_potential"
    }:
        raise RuntimeError("missing prior control")

    interpolation_bulk = comparison(analysis, "none", "interpolation", 0.12, 128)
    interpolation_near = comparison(analysis, "none", "interpolation", 0.02, 256)
    extrapolation_bulk = comparison(analysis, "none", "extrapolation", 0.12, 128)
    local_vs_full_bulk = comparison(analysis, "full", "interpolation", 0.12, 128)
    local_vs_full_near = comparison(analysis, "full", "interpolation", 0.02, 256)
    local_vs_local_both_bulk = comparison(
        analysis, "local_both", "interpolation", 0.12, 128
    )

    exact_fallback_differences = []
    for mode in ("local_both", "local_potential", "full"):
        for margin in (0.12, 0.02):
            for n_b in (64, 128, 256):
                structured = float(cell(summary, mode, 0.0, margin, n_b)["mean_rel_l2"])
                classical = float(cell(summary, "correct_yukawa_bie", 0.0, margin, n_b)["mean_rel_l2"])
                exact_fallback_differences.append(abs(structured - classical))
    fallback_gap = max(exact_fallback_differences)

    if significant_better(interpolation_bulk) and significant_better(interpolation_near):
        verdict = "PAPERIZE"
    elif significant_better(interpolation_bulk) or significant_better(interpolation_near):
        verdict = "NARROW"
    else:
        verdict = "KILL"

    audit = {
        "integrity": {
            "raw_rows": len(raw), "expected_rows": expected_rows,
            "duplicate_keys": 0, "nonfinite_values": 0,
            "training_runs": len(training), "seeds_per_method": 3,
            "test_geometries_per_cell": 64,
        },
        "primary_comparisons": {
            "local_vs_none_interpolation_bulk_nb128": interpolation_bulk,
            "local_vs_none_interpolation_near_nb256": interpolation_near,
            "local_vs_none_extrapolation_bulk_nb128": extrapolation_bulk,
            "local_vs_full_interpolation_bulk_nb128": local_vs_full_bulk,
            "local_vs_full_interpolation_near_nb256": local_vs_full_near,
            "local_potential_vs_local_both_interpolation_bulk_nb128": local_vs_local_both_bulk,
        },
        "max_mean_error_gap_from_classical_at_kappa_zero": fallback_gap,
        "specialist_context_kappa4_standard_test": {
            "fixed_kappa_local_three_seed_mean": 0.00269792966855069,
            "fixed_kappa_no_prior_three_seed_mean": 0.0028516020781050124,
            "note": "Context only: separate specialist training protocol.",
        },
        "verdict": verdict,
    }
    (RESULT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Parameter-conditioned singular-prior NBIO",
        "",
        "## Protocol",
        "",
        "One checkpoint is trained on `kappa={0,1,2,4}` at `n_b=128`. Held-out tests contain",
        "interpolation values `{0.25,0.5,1.5,3}` and extrapolation values `{5,6}`, each on the same",
        "64 geometry/forcing instances at three boundary resolutions and two query margins. Full-prior,",
        "no-prior, local-both, and local-potential models use equal correction-network capacity and three seeds.",
        "",
        "The prior-bearing models enforce a `(kappa/4)^2` residual envelope, so their `kappa=0` output",
        "is exactly the classical Laplace BIE rather than a learned approximation.",
        "",
        "## Bulk results at the training resolution",
        "",
        "| kappa | regime | correct BIE | fixed Laplace BIE | full prior | no prior | local both | local potential |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    regime = {0.0: "seen", 0.25: "interp", 0.5: "interp", 1.0: "seen",
              1.5: "interp", 2.0: "seen", 3.0: "interp", 4.0: "seen",
              5.0: "extra", 6.0: "extra"}
    for kappa in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        values = [fmt(cell(summary, method, kappa, 0.12, 128)["mean_rel_l2"])
                  for method in ("correct_yukawa_bie", "fixed_laplace_bie", "full", "none",
                                 "local_both", "local_potential")]
        lines.append(
            f"| `{kappa:g}` | {regime[kappa]} | " + " | ".join(f"`{value}`" for value in values) + " |"
        )

    lines += ["", "## Paired claim audit", ""]
    for title, item in (
        ("interpolation bulk, n_b=128: local vs no prior", interpolation_bulk),
        ("interpolation near, n_b=256: local vs no prior", interpolation_near),
        ("extrapolation bulk, n_b=128: local vs no prior", extrapolation_bulk),
        ("interpolation bulk, n_b=128: local vs full prior", local_vs_full_bulk),
        ("interpolation near, n_b=256: local vs full prior", local_vs_full_near),
        ("interpolation bulk, n_b=128: local potential vs local both", local_vs_local_both_bulk),
    ):
        change, significance = direction(item)
        ci = item["bootstrap_mean_difference_ci"]
        lines.append(
            f"- {title}: **{change}**; paired geometry-cluster mean-difference 95% CI "
            f"`[{fmt(ci[0])}, {fmt(ci[1])}]` ({significance}); local win fraction "
            f"`{100 * item['target_win_fraction']:.1f}%`."
        )

    conditioned_k4 = cell(summary, "local_potential", 4.0, 0.12, 128)
    lines += [
        "",
        "## Claim boundary",
        "",
        f"The maximum aggregate discrepancy between either prior-bearing model and the classical BIE at",
        f"`kappa=0` is `{fmt(fallback_gap)}`. The conditioned local model obtains",
        f"`{fmt(conditioned_k4['mean_rel_l2'])}` at the seen `kappa=4`; the separate fixed-`kappa` specialist",
        "obtains `2.70e-3`, so any family-model accuracy cost must be reported rather than hidden.",
        "",
        "Interpolation and extrapolation are reported separately. A positive interpolation result does not",
        "establish out-of-range parameter extrapolation, and the correct-kernel BIE remains the numerical ceiling.",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
