#!/usr/bin/env python3
"""Recompute the surface inverse-design aggregates from the M14/M15 raw files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface_root", type=Path)
    parser.add_argument("out", type=Path)
    return parser.parse_args()


def load_jsons(path: Path) -> list[dict]:
    files = sorted(path.glob("seed_*.json"))
    if len(files) != 3:
        raise RuntimeError(f"expected three seed files in {path}, found {len(files)}")
    return [json.loads(file.read_text()) for file in files]


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    args = parse_args()
    root = args.surface_root.resolve()
    m14 = load_jsons(root / "results" / "m14_derivative_control_fleet")
    m15 = load_jsons(root / "results" / "m15_indep_reference")

    m14_rows: dict[str, list[dict]] = defaultdict(list)
    for seed in m14:
        for row in seed["methods"]:
            m14_rows[row["method"]].append(row)
    m14_summary = {}
    for method, rows in sorted(m14_rows.items()):
        m14_summary[method] = {
            "median_theta_relerr": median([row["theta_relerr"] for row in rows]),
            "median_true_probe_mse": median(
                [row["true_probe_mse"] for row in rows]
            ),
            "median_seconds": median([row["seconds"] for row in rows]),
        }
    qbx = m14_summary["qbx"]
    derivative = m14_summary["derivative_constrained"]
    m14_summary["qbx_vs_derivative"] = {
        "theta_error_advantage": (
            derivative["median_theta_relerr"] / qbx["median_theta_relerr"]
        ),
        "true_mse_advantage": (
            derivative["median_true_probe_mse"] / qbx["median_true_probe_mse"]
        ),
        "qbx_wins_all_seeds_theta": all(
            next(row for row in seed["methods"] if row["method"] == "qbx")[
                "theta_relerr"
            ]
            < next(
                row
                for row in seed["methods"]
                if row["method"] == "derivative_constrained"
            )["theta_relerr"]
            for seed in m14
        ),
        "derivative_jvp_cos_median_over_seeds": median(
            [seed["derivative_validation"]["jvp_cos_median"] for seed in m14]
        ),
    }

    best_by_seed_condition: dict[str, dict[str, dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for seed in m15:
        seed_key = str(seed["seed"])
        for row in seed["rows"]:
            key = best_by_seed_condition[seed_key][row["condition"]][row["method"]]
            key.append(row["theta_relerr"])

    conditions = sorted({row["condition"] for seed in m15 for row in seed["rows"]})
    methods = sorted({row["method"] for seed in m15 for row in seed["rows"]})
    m15_summary = {}
    for condition in conditions:
        m15_summary[condition] = {}
        for method in methods:
            best = [
                min(best_by_seed_condition[str(seed["seed"])][condition][method])
                for seed in m15
            ]
            m15_summary[condition][method] = {
                "median_best_of_3_theta_relerr": median(best),
                "per_seed_best_of_3_theta_relerr": best,
            }
        m15_summary[condition]["qbx_vs_derivative"] = {
            "theta_error_advantage": (
                m15_summary[condition]["derivative"][
                    "median_best_of_3_theta_relerr"
                ]
                / m15_summary[condition]["qbx"][
                    "median_best_of_3_theta_relerr"
                ]
            ),
            "qbx_wins_all_seeds": all(
                min(
                    best_by_seed_condition[str(seed["seed"])][condition]["qbx"]
                )
                < min(
                    best_by_seed_condition[str(seed["seed"])][condition][
                        "derivative"
                    ]
                )
                for seed in m15
            ),
        }

    payload = {
        "m14_paired_box_constrained": {
            "n_seeds": len(m14),
            "summary": m14_summary,
        },
        "m15_independent_reference": {
            "n_seeds": len(m15),
            "coarse_fine_gap_mse": {
                "max": max(seed["coarse_fine_gap_mse"] for seed in m15),
                "per_seed": [seed["coarse_fine_gap_mse"] for seed in m15],
            },
            "best_of_3_summary": m15_summary,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
