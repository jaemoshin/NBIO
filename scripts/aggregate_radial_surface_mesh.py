#!/usr/bin/env python
"""Aggregate radial-mixture mesh cells against the frozen generic correction."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def bootstrap_median(values, seed=20260725):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        array[rng.integers(0, len(array), size=(10000, len(array)))],
        axis=1,
    )
    return {
        "median": float(np.median(array)),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-pattern", required=True)
    parser.add_argument("--old-raw", required=True)
    parser.add_argument("--old-summary", required=True)
    parser.add_argument("--ceiling-pattern")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    new_rows = []
    source_hashes = None
    paths = sorted(glob.glob(args.new_pattern))
    for path in paths:
        payload = json.load(open(path))
        if source_hashes is None:
            source_hashes = payload["source_hashes"]
        elif source_hashes != payload["source_hashes"]:
            raise RuntimeError(f"source hash mismatch: {path}")
        new_rows.extend(payload["rows"])

    old_rows = list(csv.DictReader(open(args.old_raw)))
    old_by_cell = {
        (int(row["seed"]), row["mesh"]): row
        for row in old_rows
        if row["status"] == "measured"
    }
    valid_meshes = {
        row["mesh"] for row in csv.DictReader(open(args.old_summary))
        if row["category"] == "valid"
    }
    expected = {(seed, mesh) for seed in range(4) for mesh in valid_meshes}
    observed = {(int(row["eval_seed"]), row["mesh"]) for row in new_rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"mesh coverage mismatch missing={missing} extra={extra}")

    paired = []
    for row in new_rows:
        key = (int(row["eval_seed"]), row["mesh"])
        old = old_by_cell[key]
        for metric in ("known_kernel", "floor"):
            if not np.isclose(
                float(row[metric]), float(old[metric]), rtol=1e-10, atol=1e-12,
            ):
                raise RuntimeError(f"{metric} mismatch for {key}")
        generic = float(old["corr_transfer"])
        floor = float(row["floor"])
        mixture = float(row["radial_mixture"])
        public_row = {
            key: value for key, value in row.items() if key != "checkpoint"
        }
        paired.append({
            **public_row,
            "generic_pairwise": generic,
            "generic_gap_closed_pct": 100 * (1 - generic / floor),
            "mixture_beats_generic": mixture < generic,
            "mixture_beats_floor": mixture < floor,
        })

    by_mesh = defaultdict(list)
    for row in paired:
        by_mesh[row["mesh"]].append(row)
    mesh_rows = []
    for mesh, rows in sorted(by_mesh.items()):
        mesh_rows.append({
            "mesh": mesh,
            "vertices": rows[0]["vertices"],
            "n_seeds": len(rows),
            "floor_mean": float(np.mean([r["floor"] for r in rows])),
            "generic_mean": float(np.mean([r["generic_pairwise"] for r in rows])),
            "mixture_mean": float(np.mean([r["radial_mixture"] for r in rows])),
            "generic_gap_closed_pct": float(np.mean([
                r["generic_gap_closed_pct"] for r in rows
            ])),
            "mixture_gap_closed_pct": float(np.mean([
                r["gap_closed_pct"] for r in rows
            ])),
            "mixture_seed_wins": sum(r["mixture_beats_generic"] for r in rows),
        })

    generic_closure = [row["generic_gap_closed_pct"] for row in mesh_rows]
    mixture_closure = [row["mixture_gap_closed_pct"] for row in mesh_rows]
    aggregate = {
        "n_files": len(paths),
        "n_meshes": len(mesh_rows),
        "n_cells": len(paired),
        "source_hashes": source_hashes,
        "generic_gap_closure": bootstrap_median(generic_closure),
        "mixture_gap_closure": bootstrap_median(mixture_closure),
        "median_closure_gain_pp": float(
            np.median(mixture_closure) - np.median(generic_closure)
        ),
        "mixture_cell_wins": int(sum(r["mixture_beats_generic"] for r in paired)),
        "mixture_floor_wins": int(sum(r["mixture_beats_floor"] for r in paired)),
        "generic_losing_cells": int(sum(
            float(r["generic_pairwise"]) >= float(r["floor"]) for r in paired
        )),
        "mixture_losing_cells": int(sum(not r["mixture_beats_floor"] for r in paired)),
        "max_mixture_error": float(max(r["radial_mixture"] for r in paired)),
        "max_solve_residual": float(max(r["solve_residual"] for r in paired)),
    }
    aggregate["gate_c"] = {
        "median_gain_at_least_10pp": aggregate["median_closure_gain_pp"] >= 10,
        "fewer_losing_cells": (
            aggregate["mixture_losing_cells"] < aggregate["generic_losing_cells"]
        ),
    }

    ceiling_rows = []
    if args.ceiling_pattern:
        paired_by_cell = {
            (int(row["eval_seed"]), row["mesh"]): row for row in paired
        }
        for path in sorted(glob.glob(args.ceiling_pattern)):
            payload = json.load(open(path))
            for row in payload["rows"]:
                key = (int(row["eval_seed"]), row["mesh"])
                if key not in paired_by_cell:
                    raise RuntimeError(f"unexpected ceiling cell: {key}")
                matched = paired_by_cell[key]
                for metric in ("known_kernel", "floor"):
                    if not np.isclose(
                        float(row[metric]),
                        float(matched[metric]),
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        raise RuntimeError(
                            f"ceiling {metric} mismatch for {key}"
                        )
                exact = float(row["exact_radial_split"])
                mixture = float(matched["radial_mixture"])
                floor = float(row["floor"])
                ceiling_rows.append({
                    **row,
                    "radial_mixture": mixture,
                    "generic_pairwise": float(matched["generic_pairwise"]),
                    "mixture_minus_exact_abs": abs(mixture - exact),
                    "exact_beats_floor": exact < floor,
                    "mixture_beats_floor": mixture < floor,
                })
        aggregate["ceiling_diagnostic"] = {
            "n_cells": len(ceiling_rows),
            "meshes": sorted({row["mesh"] for row in ceiling_rows}),
            "exact_losing_cells": int(sum(
                not row["exact_beats_floor"] for row in ceiling_rows
            )),
            "mixture_losing_cells": int(sum(
                not row["mixture_beats_floor"] for row in ceiling_rows
            )),
            "failure_sets_identical": {
                (row["mesh"], int(row["eval_seed"]))
                for row in ceiling_rows if not row["exact_beats_floor"]
            } == {
                (row["mesh"], int(row["eval_seed"]))
                for row in ceiling_rows if not row["mixture_beats_floor"]
            },
            "median_abs_error_difference": float(np.median([
                row["mixture_minus_exact_abs"] for row in ceiling_rows
            ])),
            "max_abs_error_difference": float(max(
                row["mixture_minus_exact_abs"] for row in ceiling_rows
            )),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("raw_results.csv", paired), ("mesh_summary.csv", mesh_rows)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if ceiling_rows:
        with open(
            os.path.join(args.out_dir, "ceiling_diagnostic.csv"),
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(ceiling_rows[0]))
            writer.writeheader()
            writer.writerows(ceiling_rows)
    with open(os.path.join(args.out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
