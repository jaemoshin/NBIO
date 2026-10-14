#!/usr/bin/env python
"""Two-way paired bootstrap over training seed and geometry cluster."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1" / "raw"
OUTPUT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1" / "hierarchical_bootstrap.json"
FAMILIES = ("in_dist", "highmode", "rough", "deform")


def read(prefix):
    rows = []
    for family in FAMILIES:
        with (RAW / f"{prefix}_{family}.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def paired_ratio(rows, numerator, denominator, condition=None, draws=100000):
    chosen = [
        row for row in rows
        if row["method"] in {numerator, denominator}
        and int(row["n_boundary"]) == 128
        and (condition is None or row["condition"] == condition)
    ]
    clusters = sorted({(row["family"], int(row["sample_id"])) for row in chosen})
    values = defaultdict(list)
    for row in chosen:
        values[(
            row["method"], int(row["seed_index"]), row["family"], int(row["sample_id"]),
        )].append(float(row["rel_l2"]))
    arrays = {}
    for method in (numerator, denominator):
        arrays[method] = np.asarray([
            [np.mean(values[(method, seed, family, sample)]) for family, sample in clusters]
            for seed in range(3)
        ])
    point = arrays[numerator].mean() / arrays[denominator].mean()
    rng = np.random.default_rng(173101)
    ratios = np.empty(draws)
    for draw in range(draws):
        seeds = rng.integers(0, 3, 3)
        geometries = rng.integers(0, len(clusters), len(clusters))
        ratios[draw] = (
            arrays[numerator][seeds][:, geometries].mean()
            / arrays[denominator][seeds][:, geometries].mean()
        )
    return {
        "ratio": float(point),
        "ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))],
        "draws": draws,
        "training_seeds": 3,
        "geometry_clusters": len(clusters),
        "resampling": "paired training-seed indices and paired geometry clusters",
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    main_rows = read("confirmation_main")
    efficiency_rows = read("confirmation_efficiency")
    output = {
        "spline_over_radial": paired_ratio(main_rows, "radial_spline", "radial_normal"),
        "radial_over_generic": paired_ratio(main_rows, "radial_normal", "generic_matched"),
        "sample_n100_radial_over_generic": paired_ratio(
            efficiency_rows, "radial_normal", "generic_matched", "sample_n100",
        ),
        "noise5_radial_over_generic": paired_ratio(
            efficiency_rows, "radial_normal", "generic_matched", "sensor_q32_noise0p05",
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
