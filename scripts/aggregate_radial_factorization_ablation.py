#!/usr/bin/env python
"""Single-shot locked aggregate for the parameter-matched radial ablation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_paperization_v1"
REPORT = ROOT / "reports" / "radial_factorization_ablation_v1.md"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
METHODS = ("generic_matched", "radial_separate", "radial_normal")
KAPPAS = (1.2, 1.75, 2.4, 3.1, 3.8)
LOADS = ("point_source", "fourier")
RESOLUTIONS = (64, 128, 256)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed_index"] = int(row["seed_index"])
        row["kappa"] = float(row["kappa"])
        row["n_boundary"] = int(row["n_boundary"])
        row["sample_id"] = int(row["sample_id"])
        row["rel_l2"] = float(row["rel_l2"])
        row["finite"] = row["finite"].lower() == "true"
    return rows


def select(rows, method, *, seed=None, resolution=128, family=None, load=None, kappa=None):
    return [row for row in rows if (
        row["method"] == method
        and row["n_boundary"] == resolution
        and (seed is None or row["seed_index"] == seed)
        and (family is None or row["family"] == family)
        and (load is None or row["load_family"] == load)
        and (kappa is None or row["kappa"] == kappa)
    )]


def mean_error(rows, method, **filters):
    chosen = select(rows, method, **filters)
    if not chosen:
        raise RuntimeError(f"empty selection: {method} {filters}")
    return float(np.mean([row["rel_l2"] for row in chosen]))


def cluster_values(rows, method):
    grouped = defaultdict(list)
    for row in select(rows, method):
        key = (row["family"], row["load_family"], row["kappa"], row["sample_id"])
        grouped[key].append(row["rel_l2"])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def ratio_bootstrap(numerator: dict, denominator: dict, seed=112001, draws=5000):
    keys = sorted(numerator)
    if keys != sorted(denominator):
        raise RuntimeError("cluster alignment failed")
    num = np.asarray([numerator[key] for key in keys])
    den = np.asarray([denominator[key] for key in keys])
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws)
    for index in range(draws):
        sample = rng.integers(0, len(keys), size=len(keys))
        ratios[index] = num[sample].mean() / den[sample].mean()
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    outputs = (RESULT_ROOT / "aggregate_summary.csv", RESULT_ROOT / "verdict.json", REPORT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("write-once aggregate output already exists")
    rows = []
    coverage = {}
    for family in FAMILIES:
        family_rows = read_rows(RESULT_ROOT / "raw" / f"{family}.csv")
        coverage[family] = len(family_rows)
        if len(family_rows) != 9600:
            raise RuntimeError(f"row-count mismatch for {family}: {len(family_rows)}")
        rows.extend(family_rows)
    all_finite = all(row["finite"] and np.isfinite(row["rel_l2"]) for row in rows)
    methods = METHODS + ("exact_yukawa_bie",)
    primary = {method: mean_error(rows, method) for method in methods}

    g1_ratio = primary["radial_normal"] / primary["generic_matched"]
    g2_ratio = primary["radial_normal"] / primary["radial_separate"]
    g1_seed_ratios = [
        mean_error(rows, "radial_normal", seed=seed)
        / mean_error(rows, "generic_matched", seed=seed)
        for seed in range(3)
    ]
    g2_seed_ratios = [
        mean_error(rows, "radial_normal", seed=seed)
        / mean_error(rows, "radial_separate", seed=seed)
        for seed in range(3)
    ]
    g1_ci = ratio_bootstrap(
        cluster_values(rows, "radial_normal"), cluster_values(rows, "generic_matched"),
    )
    g2_ci = ratio_bootstrap(
        cluster_values(rows, "radial_normal"), cluster_values(rows, "radial_separate"),
        seed=112002,
    )
    g1_pass = g1_ratio <= 0.8 and max(g1_seed_ratios) < 1.0 and g1_ci[1] < 1.0
    g2_pass = g2_ratio <= 0.8 and max(g2_seed_ratios) < 1.0 and g2_ci[1] < 1.0
    g3_ratio = primary["radial_normal"] / primary["exact_yukawa_bie"]
    g3_pass = g3_ratio <= 3.0

    with (RESULT_ROOT / "raw" / "stability.csv").open(newline="") as handle:
        stability = list(csv.DictReader(handle))
    if len(stability) != 360:
        raise RuntimeError(f"stability row-count mismatch: {len(stability)}")
    radial_stability = [row for row in stability if row["method"] == "radial_normal"]
    max_residual = max(float(row["relative_residual"]) for row in radial_stability)
    max_condition_ratio = max(float(row["condition_ratio"]) for row in radial_stability)
    g4_pass = all_finite and max_residual <= 1e-5 and max_condition_ratio <= 1.5

    hidden_failures = []
    for family in FAMILIES:
        for load in LOADS:
            for kappa in KAPPAS:
                for resolution in RESOLUTIONS:
                    normal = mean_error(
                        rows, "radial_normal", family=family, load=load,
                        kappa=kappa, resolution=resolution,
                    )
                    generic = mean_error(
                        rows, "generic_matched", family=family, load=load,
                        kappa=kappa, resolution=resolution,
                    )
                    separate = mean_error(
                        rows, "radial_separate", family=family, load=load,
                        kappa=kappa, resolution=resolution,
                    )
                    if normal > 2.0 * generic and normal > 2.0 * separate:
                        hidden_failures.append({
                            "family": family, "load": load, "kappa": kappa,
                            "n_boundary": resolution, "radial_normal": normal,
                            "generic_matched": generic, "radial_separate": separate,
                        })

    if g1_pass and g2_pass and g3_pass and g4_pass and not hidden_failures:
        outcome, verdict = "A", "PAPERIZE"
    elif g1_pass and g3_pass and g4_pass and not hidden_failures:
        outcome, verdict = "B", "PAPERIZE"
    elif g4_pass:
        outcome, verdict = "C", "NARROW"
    else:
        outcome, verdict = "D", "KILL"

    summary_rows = []
    for method in methods:
        for family in FAMILIES:
            for load in LOADS:
                for kappa in KAPPAS:
                    for resolution in RESOLUTIONS:
                        values = [
                            row["rel_l2"] for row in select(
                                rows, method, family=family, load=load,
                                kappa=kappa, resolution=resolution,
                            )
                        ]
                        summary_rows.append({
                            "method": method, "family": family, "load_family": load,
                            "kappa": kappa, "n_boundary": resolution,
                            "n_rows": len(values), "mean_rel_l2": np.mean(values),
                            "median_rel_l2": np.median(values),
                            "p90_rel_l2": np.quantile(values, 0.9),
                        })
    write_csv(summary_rows, RESULT_ROOT / "aggregate_summary.csv")
    result = {
        "protocol": "radial_factorization_ablation_v1",
        "coverage": coverage,
        "all_finite": all_finite,
        "primary_mean_rel_l2": primary,
        "G1_radial_vs_generic": {
            "ratio": g1_ratio, "seed_ratios": g1_seed_ratios,
            "cluster_bootstrap_ci95": g1_ci, "pass": g1_pass,
        },
        "G2_normal_form_vs_separate": {
            "ratio": g2_ratio, "seed_ratios": g2_seed_ratios,
            "cluster_bootstrap_ci95": g2_ci, "pass": g2_pass,
        },
        "G3_exact_proximity": {"ratio": g3_ratio, "pass": g3_pass},
        "G4_stability": {
            "max_relative_residual": max_residual,
            "max_condition_ratio": max_condition_ratio,
            "pass": g4_pass,
        },
        "hidden_failures": hidden_failures,
        "outcome": outcome,
        "verdict": verdict,
    }
    (RESULT_ROOT / "verdict.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    REPORT.write_text(f"""# Parameter-matched radial factorization ablation

Verdict: **{verdict}** (Outcome {outcome})

This fresh, blinded test matches the generic and radial models to within 20 parameters and uses identical
data, batches, optimizer updates, and checkpoint selection.

| method | parameters | primary mean rel-L2 |
|---|---:|---:|
| exact Yukawa BIE | 0 | `{primary['exact_yukawa_bie']:.4e}` |
| generic pairwise correction | 2,221 | `{primary['generic_matched']:.4e}` |
| radial `(kappa,r)` | 2,241 | `{primary['radial_separate']:.4e}` |
| radial normal form `kappa*r` | 2,241 | **`{primary['radial_normal']:.4e}`** |

- G1 radial normal form versus matched generic: **{'PASS' if g1_pass else 'FAIL'}**; ratio
  `{g1_ratio:.4f}`, clustered 95% interval `{g1_ci[0]:.4f}-{g1_ci[1]:.4f}`.
- G2 dimensionless collapse versus separate inputs: **{'PASS' if g2_pass else 'FAIL'}**; ratio
  `{g2_ratio:.4f}`, clustered 95% interval `{g2_ci[0]:.4f}-{g2_ci[1]:.4f}`.
- G3 exact-BIE proximity: **{'PASS' if g3_pass else 'FAIL'}**; ratio `{g3_ratio:.3f}`.
- G4 stability: **{'PASS' if g4_pass else 'FAIL'}**; maximum solve residual
  `{max_residual:.3e}`, maximum condition ratio `{max_condition_ratio:.3f}`.
- Hidden resolved failures: `{len(hidden_failures)}`.

Raw rows, checkpoint manifests, training curves, and the machine-readable verdict are under
`results/screening_law/radial_paperization_v1/`.
""")
    print(json.dumps({"outcome": outcome, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
