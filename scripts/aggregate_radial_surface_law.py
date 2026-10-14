#!/usr/bin/env python
"""Aggregate raw radial-surface rows and evaluate the frozen tranche gates."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np


def read_rows(patterns):
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                for key in (
                    "model_seed", "params", "case_seed",
                    "kappa", "rel_l2", "solve_residual", "condition_number",
                ):
                    row[key] = float(row[key])
                row["source_file"] = path
                rows.append(row)
    if not rows:
        raise SystemExit("no raw rows found")
    return paths, rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["split"], row["geometry"], row["kappa"],
        )
        groups[key].append(row["rel_l2"])
    result = []
    for key, values in sorted(groups.items()):
        array = np.asarray(values)
        result.append({
            "method": key[0],
            "split": key[1],
            "geometry": key[2],
            "kappa": key[3],
            "n": len(array),
            "mean_rel_l2": float(array.mean()),
            "median_rel_l2": float(np.median(array)),
            "std_rel_l2": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "p90_rel_l2": float(np.quantile(array, 0.9)),
            "max_rel_l2": float(array.max()),
        })
    return result


def seed_means(rows, method, split, geometry):
    values = defaultdict(list)
    for row in rows:
        if (
            row["method"] == method
            and row["split"] == split
            and row["geometry"] == geometry
        ):
            values[int(row["model_seed"])].append(row["rel_l2"])
    return {seed: float(np.mean(items)) for seed, items in values.items()}


def pooled_seed_means(rows, method, split):
    values = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["split"] == split:
            values[int(row["model_seed"])].append(row["rel_l2"])
    return {seed: float(np.mean(items)) for seed, items in values.items()}


def gate_a(rows):
    generic_torus = seed_means(rows, "generic_pairwise", "fixed", "torus")
    generic_sphere = seed_means(rows, "generic_pairwise", "fixed", "sphere")
    radial_sphere = seed_means(rows, "radial_product", "fixed", "sphere")
    common = sorted(set(generic_sphere) & set(radial_sphere))
    ratios = [radial_sphere[s] / generic_sphere[s] for s in common]
    return {
        "n_seeds": len(common),
        "generic_torus_mean": float(np.mean(list(generic_torus.values()))),
        "generic_sphere_mean": float(np.mean(list(generic_sphere.values()))),
        "radial_sphere_mean": float(np.mean(list(radial_sphere.values()))),
        "radial_to_generic_seed_ratios": ratios,
        "reproduction_below_1e-2": float(np.mean(list(generic_torus.values()))) < 1e-2,
        "radial_all_seed_wins": bool(ratios) and all(ratio < 1 for ratio in ratios),
        "radial_at_least_3x": bool(ratios) and float(np.mean(ratios)) < 1 / 3,
    }


def gate_b(rows):
    methods = ("generic_kappa_pairwise", "radial_separate", "radial_product")
    pooled = {
        method: pooled_seed_means(rows, method, "interpolation")
        for method in methods
    }
    common = sorted(set.intersection(*(set(values) for values in pooled.values())))
    best_control = {
        seed: min(pooled["generic_kappa_pairwise"][seed], pooled["radial_separate"][seed])
        for seed in common
    }
    ratios = [pooled["radial_product"][seed] / best_control[seed] for seed in common]

    closures = {}
    for geometry in ("torus", "sphere", "two_sphere"):
        radial = seed_means(rows, "radial_product", "interpolation", geometry)
        floor = seed_means(rows, "laplace_floor", "interpolation", geometry)
        seeds = sorted(set(radial) & set(floor))
        closures[geometry] = [
            1 - radial[seed] / floor[seed] for seed in seeds
        ]
    max_residual = max(row["solve_residual"] for row in rows)
    all_finite = all(
        math.isfinite(row["rel_l2"])
        and math.isfinite(row["solve_residual"])
        and math.isfinite(row["condition_number"])
        for row in rows
    )
    return {
        "n_seeds": len(common),
        "pooled_interpolation_seed_means": pooled,
        "radial_to_best_control_seed_ratios": ratios,
        "gap_closure_by_geometry": closures,
        "max_solve_residual": max_residual,
        "radial_all_seed_wins": bool(ratios) and all(ratio < 1 for ratio in ratios),
        "radial_ratio_below_0_33": bool(ratios) and float(np.mean(ratios)) < 0.33,
        "every_geometry_closes_80pct": bool(closures) and all(
            values and float(np.mean(values)) >= 0.8
            for values in closures.values()
        ),
        "finite_and_residual_below_1e-7": all_finite and max_residual < 1e-7,
    }


def gate_b2(rows):
    mixture = pooled_seed_means(rows, "radial_mixture", "interpolation")
    generic = pooled_seed_means(rows, "generic_kappa_pairwise", "interpolation")
    product = pooled_seed_means(rows, "radial_product", "interpolation")
    separate = pooled_seed_means(rows, "radial_separate", "interpolation")
    common = sorted(set(mixture) & set(generic) & set(product) & set(separate))
    best_control = {
        seed: min(generic[seed], product[seed], separate[seed])
        for seed in common
    }
    ratios = [mixture[seed] / best_control[seed] for seed in common]
    closures = {}
    for geometry in ("torus", "sphere", "two_sphere"):
        learned = seed_means(rows, "radial_mixture", "interpolation", geometry)
        floor = seed_means(rows, "laplace_floor", "interpolation", geometry)
        seeds = sorted(set(learned) & set(floor))
        closures[geometry] = [
            1 - learned[seed] / floor[seed] for seed in seeds
        ]
    two_sphere_rows = [
        row["rel_l2"] for row in rows
        if row["method"] == "radial_mixture"
        and row["split"] == "interpolation"
        and row["geometry"] == "two_sphere"
    ]
    residuals = [
        row["solve_residual"] for row in rows
        if row["method"] == "radial_mixture"
    ]
    paired = {}
    for row in rows:
        if (
            row["split"] == "interpolation"
            and row["method"] in {"radial_mixture", "generic_kappa_pairwise"}
        ):
            key = (
                int(row["model_seed"]),
                row["geometry"],
                int(row["case_seed"]),
                float(row["kappa"]),
            )
            paired.setdefault(key, {})[row["method"]] = row["rel_l2"]
    paired = {
        key: value for key, value in paired.items()
        if len(value) == 2
    }
    cluster_values = defaultdict(lambda: defaultdict(list))
    for (seed, geometry, case_seed, _), value in paired.items():
        cluster_values[seed][(geometry, case_seed)].append(value)
    rng = np.random.default_rng(20260725)
    seeds = sorted(cluster_values)
    draws = []
    if seeds:
        for _ in range(10000):
            numerator = []
            denominator = []
            for sampled_seed in rng.choice(seeds, len(seeds), replace=True):
                clusters = sorted(cluster_values[int(sampled_seed)])
                for sampled_cluster in rng.choice(
                    len(clusters), len(clusters), replace=True,
                ):
                    values = cluster_values[int(sampled_seed)][
                        clusters[int(sampled_cluster)]
                    ]
                    numerator.extend(v["radial_mixture"] for v in values)
                    denominator.extend(v["generic_kappa_pairwise"] for v in values)
            draws.append(float(np.mean(numerator) / np.mean(denominator)))
    ratio_ci = {
        "estimate": float(
            np.mean([v["radial_mixture"] for v in paired.values()])
            / np.mean([v["generic_kappa_pairwise"] for v in paired.values()])
        ) if paired else None,
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ] if draws else [None, None],
        "clusters": sum(len(value) for value in cluster_values.values()),
    }
    return {
        "n_seeds": len(common),
        "pooled_interpolation_seed_means": {
            "radial_mixture": mixture,
            "generic_kappa_pairwise": generic,
            "radial_product": product,
            "radial_separate": separate,
        },
        "mixture_to_best_control_seed_ratios": ratios,
        "gap_closure_by_geometry": closures,
        "max_two_sphere_interpolation_error": max(two_sphere_rows, default=None),
        "max_solve_residual": max(residuals, default=None),
        "paired_hierarchical_ratio": ratio_ci,
        "all_seed_wins": bool(ratios) and all(ratio < 1 for ratio in ratios),
        "mean_ratio_below_0_33": bool(ratios) and float(np.mean(ratios)) < 0.33,
        "every_geometry_closes_90pct": bool(closures) and all(
            values and float(np.mean(values)) >= 0.9
            for values in closures.values()
        ),
        "two_sphere_below_0_1": bool(two_sphere_rows)
        and max(two_sphere_rows) < 0.1,
        "residual_below_1e-7": bool(residuals) and max(residuals) < 1e-7,
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    paths, rows = read_rows(args.patterns)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "raw_results.csv"), rows)
    summary = summarize(rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    modes = {row["split"] for row in rows}
    aggregate = {
        "source_files": paths,
        "n_rows": len(rows),
        "model_seeds": sorted({int(row["model_seed"]) for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "splits": sorted(modes),
        "max_solve_residual": max(row["solve_residual"] for row in rows),
        "summary": summary,
    }
    if "fixed" in modes:
        aggregate["gate_a"] = gate_a(rows)
    if "interpolation" in modes:
        aggregate["gate_b"] = gate_b(rows)
        if "radial_mixture" in aggregate["methods"]:
            aggregate["gate_b2"] = gate_b2(rows)
    with open(os.path.join(args.out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
