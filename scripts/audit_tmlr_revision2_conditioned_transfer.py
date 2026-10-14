#!/usr/bin/env python
"""Aggregate and audit the conditioned learned-correction OOD transfer tranche."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tmlr_revision2"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
METHODS = (
    "correct_yukawa_bie",
    "fixed_laplace_bie",
    "full_prior",
    "no_prior",
)


def read_rows() -> list[dict]:
    rows = []
    for family in FAMILIES:
        path = RESULTS / f"conditioned_transfer_{family}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row.update({
                    "train_data_seed": int(row["train_data_seed"]),
                    "kappa": float(row["kappa"]),
                    "query_margin": float(row["query_margin"]),
                    "n_b": int(row["n_b"]),
                    "reference_n_b": int(row["reference_n_b"]),
                    "sample_id": int(row["sample_id"]),
                    "rel_l2": float(row["rel_l2"]),
                    "min_radius": float(row["min_radius"]),
                    "max_radius": float(row["max_radius"]),
                    "max_abs_curvature": float(row["max_abs_curvature"]),
                })
                rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("refusing to write empty table")
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["geometry_family"], row["load_family"],
            row["kappa"], row["kappa_regime"], row["query_margin"], row["n_b"],
        )
        groups[key].append(row)
    output = []
    for index, (key, group) in enumerate(sorted(groups.items())):
        method, family, load, kappa, regime, margin, n_b = key
        per_instance = defaultdict(list)
        per_seed = defaultdict(list)
        for row in group:
            per_instance[row["sample_id"]].append(row["rel_l2"])
            per_seed[row["train_data_seed"]].append(row["rel_l2"])
        values = np.asarray([np.mean(value) for value in per_instance.values()])
        seed_means = np.asarray([np.mean(value) for value in per_seed.values()])
        ci = bootstrap_ci(values, 20260722 + index)
        output.append({
            "method": method,
            "geometry_family": family,
            "load_family": load,
            "kappa": kappa,
            "kappa_regime": regime,
            "query_margin": margin,
            "n_b": n_b,
            "n_instances": len(values),
            "n_training_data_seeds": 0 if method.endswith("bie") else len(per_seed),
            "mean_rel_l2": float(values.mean()),
            "median_rel_l2": float(np.median(values)),
            "p90_rel_l2": float(np.quantile(values, 0.9)),
            "std_instances": float(values.std(ddof=1)),
            "training_seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "bootstrap_ci_low": ci[0],
            "bootstrap_ci_high": ci[1],
        })
    return output


def clustered_values(
    rows: list[dict], method: str, regime: str, load: str,
    margin: float, n_b: int, family: str | None = None,
) -> dict[tuple[str, int], float]:
    groups = defaultdict(list)
    for row in rows:
        if row["method"] != method or row["kappa_regime"] != regime:
            continue
        if row["load_family"] != load or row["query_margin"] != margin or row["n_b"] != n_b:
            continue
        if family is not None and row["geometry_family"] != family:
            continue
        groups[(row["geometry_family"], row["sample_id"])].append(row["rel_l2"])
    return {key: float(np.mean(value)) for key, value in groups.items()}


def paired_comparison(
    rows: list[dict], regime: str, load: str, margin: float, n_b: int,
    seed: int, family: str | None = None,
) -> dict:
    full = clustered_values(rows, "full_prior", regime, load, margin, n_b, family)
    none = clustered_values(rows, "no_prior", regime, load, margin, n_b, family)
    keys = sorted(set(full) & set(none))
    diff = np.asarray([full[key] - none[key] for key in keys])
    full_mean = float(np.mean([full[key] for key in keys]))
    none_mean = float(np.mean([none[key] for key in keys]))
    return {
        "geometry_family": family or "all",
        "kappa_regime": regime,
        "load_family": load,
        "query_margin": margin,
        "n_b": n_b,
        "n_geometry_clusters": len(keys),
        "full_prior_mean_rel_l2": full_mean,
        "no_prior_mean_rel_l2": none_mean,
        "full_prior_relative_improvement": (none_mean - full_mean) / max(none_mean, 1e-30),
        "full_prior_win_fraction": float(np.mean(diff < 0.0)),
        "mean_paired_difference": float(diff.mean()),
        "paired_difference_bootstrap_ci": bootstrap_ci(diff, seed),
    }


def audit(rows: list[dict], summaries: list[dict]) -> dict:
    expected = 4 * 2 * 5 * 2 * 3 * 24 * (2 + 2 * 3)
    nonfinite = sum(not np.isfinite(row["rel_l2"]) for row in rows)
    keys = [
        (
            row["method"], row["train_data_seed"], row["geometry_family"],
            row["load_family"], row["kappa"], row["query_margin"], row["n_b"],
            row["sample_id"],
        )
        for row in rows
    ]
    comparisons = []
    seed = 2026072200
    for family in (None,) + FAMILIES:
        for regime in ("in_range", "out_of_range"):
            for load in ("point_source", "fourier"):
                for margin in (0.12, 0.02):
                    for n_b in (64, 128, 256):
                        comparisons.append(paired_comparison(
                            rows, regime, load, margin, n_b, seed, family
                        ))
                        seed += 1

    summary_lookup = {
        (
            row["method"], row["geometry_family"], row["load_family"],
            row["kappa"], row["query_margin"], row["n_b"],
        ): row["mean_rel_l2"]
        for row in summaries
    }
    resolution = []
    beat_wrong = []
    for family in FAMILIES:
        for load in ("point_source", "fourier"):
            for kappa in (0.2, 1.1, 2.2, 3.8):
                for margin in (0.12, 0.02):
                    full_128 = summary_lookup[("full_prior", family, load, kappa, margin, 128)]
                    full_256 = summary_lookup[("full_prior", family, load, kappa, margin, 256)]
                    wrong = summary_lookup[("fixed_laplace_bie", family, load, kappa, margin, 128)]
                    resolution.append({
                        "geometry_family": family,
                        "load_family": load,
                        "kappa": kappa,
                        "query_margin": margin,
                        "full_prior_nb256_over_nb128": full_256 / max(full_128, 1e-30),
                    })
                    beat_wrong.append(full_128 < wrong)
    bulk_resolution = [
        row["full_prior_nb256_over_nb128"]
        for row in resolution if row["query_margin"] == 0.12
    ]
    primary = [
        row for row in comparisons
        if row["geometry_family"] == "all" and row["kappa_regime"] == "in_range"
        and row["query_margin"] in (0.12, 0.02) and row["n_b"] in (128, 256)
    ]
    return {
        "integrity": {
            "expected_raw_rows": expected,
            "actual_raw_rows": len(rows),
            "unique_keys": len(set(keys)),
            "nonfinite_rel_l2": nonfinite,
            "methods": sorted(set(row["method"] for row in rows)),
            "pass": len(rows) == expected and len(set(keys)) == expected and nonfinite == 0,
        },
        "primary_paired_comparisons": primary,
        "all_paired_comparisons": comparisons,
        "resolution_ratios": resolution,
        "kill_criteria": {
            "fraction_in_range_cells_beating_wrong_laplace": float(np.mean(beat_wrong)),
            "fraction_in_range_bulk_cells_nb256_over_nb128_below_5": float(
                np.mean(np.asarray(bulk_resolution) < 5.0)
            ),
            "conditioned_transfer_claim_pass": bool(
                np.mean(beat_wrong) > 0.5 and np.mean(np.asarray(bulk_resolution) < 5.0) > 0.5
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/tmlr_revision2")
    args = parser.parse_args()
    global RESULTS
    RESULTS = ROOT / args.results_dir
    rows = read_rows()
    summaries = summary(rows)
    result = audit(rows, summaries)
    write_csv(rows, RESULTS / "conditioned_transfer_raw.csv")
    write_csv(summaries, RESULTS / "conditioned_transfer_summary.csv")
    (RESULTS / "conditioned_transfer_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "integrity": result["integrity"],
        "kill_criteria": result["kill_criteria"],
        "primary_paired_comparisons": result["primary_paired_comparisons"],
    }, indent=2))


if __name__ == "__main__":
    main()
