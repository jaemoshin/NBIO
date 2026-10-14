#!/usr/bin/env python
"""Aggregate the final TMLR seed-replication checks from committed result JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    with (ROOT / path).open() as handle:
        return json.load(handle)


def seed_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(array.size),
        "seed_values": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    yukawa_paths = {
        "laplace_split_correction": [
            "results/nbio_yuk_s0/eval.json",
            "results/nbio_yuk_split_s1/eval.json",
            "results/nbio_yuk_split_s2/eval.json",
        ],
        "nosplit_learned_kernel": [
            "results/nbio_yuk_nosplit_s0/eval.json",
            "results/nbio_yuk_nosplit_s1/eval.json",
            "results/nbio_yuk_nosplit_s2/eval.json",
        ],
    }
    yukawa = {
        method: seed_summary([load(path)["eval"]["yukawa_test"]["rel_l2"] for path in paths])
        for method, paths in yukawa_paths.items()
    }

    joint_paths = {
        "nbio3d_joint": [
            "results/next_steps_3d_joint.json",
            "results/next_steps_3d_joint_s1.json",
            "results/next_steps_3d_joint_s2.json",
        ],
        "deepset3d_joint": [
            "results/next_steps_3d_joint_deepset.json",
            "results/next_steps_3d_joint_deepset_s1.json",
            "results/next_steps_3d_joint_deepset_s2.json",
        ],
    }
    slices = list(load(joint_paths["nbio3d_joint"][0])["eval"])
    surface_3d = {
        method: {
            split: seed_summary([load(path)["eval"][split]["mean_rel_l2"] for path in paths])
            for split in slices
        }
        for method, paths in joint_paths.items()
    }
    structure = load("results/next_steps_3d_joint_structure.json")
    surface_3d["nbio3d_structure"] = {
        split: {
            "n_seeds": 1,
            "seed_values": [row["mean_rel_l2"]],
            "mean": row["mean_rel_l2"],
            "sample_std": 0.0,
            "min": row["mean_rel_l2"],
            "max": row["mean_rel_l2"],
            "deterministic_control": True,
        }
        for split, row in structure["eval"].items()
    }

    output = {
        "protocol": "frozen final TMLR replication; no post-hoc hyperparameter changes",
        "yukawa_kappa4": yukawa,
        "surface_3d_joint": surface_3d,
        "source_files": yukawa_paths | joint_paths | {
            "nbio3d_structure": ["results/next_steps_3d_joint_structure.json"]
        },
    }
    json_path = RESULTS / "tmlr_final_multiseed.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")

    table_path = RESULTS / "tables" / "tmlr_final_multiseed.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "method", "slice", "n_seeds", "mean_rel_l2", "sample_std", "min", "max"],
            lineterminator="\n",
        )
        writer.writeheader()
        for method, row in yukawa.items():
            writer.writerow({
                "experiment": "yukawa_kappa4",
                "method": method,
                "slice": "test",
                "n_seeds": row["n_seeds"],
                "mean_rel_l2": row["mean"],
                "sample_std": row["sample_std"],
                "min": row["min"],
                "max": row["max"],
            })
        for method, method_rows in surface_3d.items():
            for split, row in method_rows.items():
                writer.writerow({
                    "experiment": "surface_3d_joint",
                    "method": method,
                    "slice": split,
                    "n_seeds": row["n_seeds"],
                    "mean_rel_l2": row["mean"],
                    "sample_std": row["sample_std"],
                    "min": row["min"],
                    "max": row["max"],
                })
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {table_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
