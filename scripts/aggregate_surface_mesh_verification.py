#!/usr/bin/env python3
"""Aggregate the frozen corrected 51-mesh surface-transfer verification.

The input directory must contain one JSON per seed/mesh:
  seed_<seed>_<mesh>.json

The script records every expected cell, including missing, non-watertight,
unevaluable, and invalid-core cases. Headline transfer statistics use only the
pre-registered validity gates inherited from the original mesh study.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


SEEDS = (0, 1, 2, 3)
MIN_PROBES = 20
KNOWN_VALID = 2e-2
GAP_MIN_FLOOR = 2.5e-2

GRAPHICS = (
    "spot",
    "cow",
    "cheburashka",
    "homer",
    "fandisk",
    "armadillo",
    "igea",
    "horse-vox",
    "rocker-arm",
    "teapot-vox",
    "ogre-vox",
    "nefertiti-vox",
    "lucy",
    "bimba-vox",
    "happy-vox",
    "max-planck-vox",
    "beetle-alt-vox",
)
THINGI10K = (
    "thingi_103538",
    "thingi_1228189",
    "thingi_1312952",
    "thingi_1620054",
    "thingi_1623064_ps",
    "thingi_1646897",
    "thingi_200010",
    "thingi_388482",
    "thingi_472103",
    "thingi_472182",
    "thingi_509311",
    "thingi_55177",
    "thingi_56265",
    "thingi_67919_ps",
    "thingi_72658",
    "thingi_78976_ps",
    "thingi_94144_ps",
    "thingi_94147_ps",
    "thingi_98939_ps",
    "thingi_98976",
)
ENGINEERING = (
    "fusion_45109_f1cebd33_3",
    "fusion_136172_f73f1078_8",
    "fusion_32220_1fd19c5e_71",
    "fusion_34673_5d261268_1",
    "fusion_41032_ed481084_4",
    "fusion_31360_a1accb4b_65",
    "bentheimer32",
    "blade_iea15",
    "blade_iea34",
    "car_708",
    "car_083",
    "car_427",
    "bentheimer20z2",
    "bentheimer16z25",
)
MESHES = GRAPHICS + THINGI10K + ENGINEERING


def source_of(name: str) -> str:
    if name.startswith("thingi_"):
        return "Thingi10K"
    if name.startswith("fusion_"):
        return "Fusion360"
    if name.startswith("car_"):
        return "ShapeNet-Car"
    if name.startswith("blade_"):
        return "IEA blade"
    if name.startswith("bentheimer"):
        return "microCT"
    return "graphics"


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_median(values: list[float], seed: int = 20260725) -> dict:
    if not values:
        return {"median": None, "ci95": [None, None], "n": 0}
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        x[rng.integers(0, len(x), size=(10000, len(x)))],
        axis=1,
    )
    return {
        "median": float(np.median(x)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n": int(len(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    args = parser.parse_args()

    if len(MESHES) != 51 or len(set(MESHES)) != 51:
        raise RuntimeError("frozen corpus must contain exactly 51 unique meshes")

    raw_rows: list[dict] = []
    hashes_by_seed: dict[int, dict] = {}
    for seed in SEEDS:
        for mesh in MESHES:
            path = args.raw_dir / f"seed_{seed}_{mesh}.json"
            row = {
                "seed": seed,
                "mesh": mesh,
                "source": source_of(mesh),
                "path": path.name,
                "_found": path.exists(),
                "status": "missing",
                "n_probe": None,
                "known_kernel": None,
                "floor": None,
                "corr_transfer": None,
                "gap_closed_pct": None,
                "corr_beats_floor": None,
            }
            if not path.exists():
                raw_rows.append(row)
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                row["status"] = f"unreadable:{type(exc).__name__}"
                raw_rows.append(row)
                continue
            hashes = payload.get("source_hashes")
            if hashes is not None:
                hashes_by_seed.setdefault(seed, hashes)
                if hashes_by_seed[seed] != hashes:
                    row["status"] = "source_hash_mismatch"
                    raw_rows.append(row)
                    continue
            value = payload.get(mesh)
            if not isinstance(value, dict):
                row["status"] = "mesh_row_missing"
                raw_rows.append(row)
                continue
            if value.get("watertight") is False:
                row["status"] = "non_watertight"
                raw_rows.append(row)
                continue
            if (
                value.get("not_evaluated_reason")
                == "zero_accepted_interior_probes"
                and int(value.get("n_probe", -1)) == 0
            ):
                row.update(
                    {
                        "status": "no_probes",
                        "n_probe": 0,
                        "vertices": value.get("V"),
                        "genus": value.get("genus"),
                        "tau_h_p5": finite_or_none(value.get("tau_h", {}).get("p5")),
                    }
                )
                raw_rows.append(row)
                continue
            required = ("known_kernel", "floor", "corr_transfer")
            if any(key not in value for key in required):
                row["status"] = "metrics_missing"
                raw_rows.append(row)
                continue
            row.update(
                {
                    "status": "measured",
                    "n_probe": int(value.get("n_probe", 0)),
                    "vertices": value.get("V"),
                    "genus": value.get("genus"),
                    "tau_h_p5": finite_or_none(value.get("tau_h", {}).get("p5")),
                    "known_kernel": finite_or_none(value["known_kernel"]),
                    "floor": finite_or_none(value["floor"]),
                    "corr_transfer": finite_or_none(value["corr_transfer"]),
                }
            )
            metrics = (row["known_kernel"], row["floor"], row["corr_transfer"])
            if any(value is None for value in metrics):
                row["status"] = "nonfinite"
            elif row["floor"] <= 0.0:
                row["status"] = (
                    "no_probes" if row["n_probe"] == 0 else "nonpositive_floor"
                )
            else:
                row["gap_closed_pct"] = 100.0 * (
                    1.0 - row["corr_transfer"] / row["floor"]
                )
                row["corr_beats_floor"] = row["corr_transfer"] < row["floor"]
            raw_rows.append(row)

    mesh_rows: list[dict] = []
    for mesh in MESHES:
        rows = [row for row in raw_rows if row["mesh"] == mesh]
        measured = [row for row in rows if row["status"] == "measured"]
        observed = [
            row for row in rows if row["status"] in {"measured", "no_probes"}
        ]
        out = {
            "mesh": mesh,
            "source": source_of(mesh),
            "seed_cells": len(rows),
            "observed_cells": len(observed),
            "measured_cells": len(measured),
            "category": "incomplete",
        }
        if observed:
            out.update(
                {
                    "vertices": observed[0].get("vertices"),
                    "genus": observed[0].get("genus"),
                    "n_probe_median": float(
                        np.median([r["n_probe"] for r in observed])
                    ),
                    "tau_h_p5": observed[0].get("tau_h_p5"),
                }
            )
        if measured:
            out.update(
                {
                    "known_kernel_mean": float(np.mean([r["known_kernel"] for r in measured])),
                    "floor_mean": float(np.mean([r["floor"] for r in measured])),
                    "corr_transfer_mean": float(
                        np.mean([r["corr_transfer"] for r in measured])
                    ),
                    "paired_seed_wins": int(sum(bool(r["corr_beats_floor"]) for r in measured)),
                    "gap_closed_seed_min_pct": float(
                        min(r["gap_closed_pct"] for r in measured)
                    ),
                    "gap_closed_seed_max_pct": float(
                        max(r["gap_closed_pct"] for r in measured)
                    ),
                }
            )
            out["gap_closed_pct"] = 100.0 * (
                1.0 - out["corr_transfer_mean"] / out["floor_mean"]
            )
        if len(observed) == len(SEEDS):
            if out["n_probe_median"] < MIN_PROBES:
                out["category"] = "excluded_low_probe"
            elif len(measured) != len(SEEDS):
                out["category"] = "invalid_metrics"
            elif out["known_kernel_mean"] >= KNOWN_VALID:
                out["category"] = "invalid_core"
            elif out["floor_mean"] < GAP_MIN_FLOOR:
                out["category"] = "small_gap"
            else:
                out["category"] = "valid"
        mesh_rows.append(out)

    valid = [row for row in mesh_rows if row["category"] == "valid"]
    valid_cells = [
        row
        for row in raw_rows
        if row["mesh"] in {item["mesh"] for item in valid} and row["status"] == "measured"
    ]
    summary = {
        "protocol": {
            "seeds": list(SEEDS),
            "n_meshes": len(MESHES),
            "expected_cells": len(MESHES) * len(SEEDS),
            "min_probes": MIN_PROBES,
            "known_valid_threshold": KNOWN_VALID,
            "minimum_floor": GAP_MIN_FLOOR,
        },
        "coverage": {
            "found_files": sum(bool(row["_found"]) for row in raw_rows),
            "status_counts": dict(Counter(row["status"] for row in raw_rows)),
            "mesh_category_counts": dict(Counter(row["category"] for row in mesh_rows)),
            "source_hash_seed_count": len(hashes_by_seed),
        },
        "valid_transfer": {
            "mesh_gap_closed_pct": bootstrap_median(
                [row["gap_closed_pct"] for row in valid]
            ),
            "positive_over_20pct_meshes": int(
                sum(row["gap_closed_pct"] > 20.0 for row in valid)
            ),
            "paired_seed_wins": int(sum(bool(row["corr_beats_floor"]) for row in valid_cells)),
            "paired_seed_cells": len(valid_cells),
            "worst_seed_gap_closed_pct": (
                float(min(row["gap_closed_pct"] for row in valid_cells))
                if valid_cells
                else None
            ),
            "worst_mesh_gap_closed_pct": (
                float(min(row["gap_closed_pct"] for row in valid)) if valid else None
            ),
        },
        "source_hashes_by_seed": hashes_by_seed,
        "manifest": [
            {"seed": seed, "mesh": mesh, "source": source_of(mesh)}
            for seed in SEEDS
            for mesh in MESHES
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_fields = [
        "seed",
        "mesh",
        "source",
        "status",
        "n_probe",
        "vertices",
        "genus",
        "tau_h_p5",
        "known_kernel",
        "floor",
        "corr_transfer",
        "gap_closed_pct",
        "corr_beats_floor",
        "path",
    ]
    mesh_fields = [
        "mesh",
        "source",
        "category",
        "seed_cells",
        "observed_cells",
        "measured_cells",
        "vertices",
        "genus",
        "n_probe_median",
        "tau_h_p5",
        "known_kernel_mean",
        "floor_mean",
        "corr_transfer_mean",
        "gap_closed_pct",
        "paired_seed_wins",
        "gap_closed_seed_min_pct",
        "gap_closed_seed_max_pct",
    ]
    write_csv(args.out_dir / "raw_cells.csv", raw_rows, raw_fields)
    write_csv(args.out_dir / "mesh_summary.csv", mesh_rows, mesh_fields)
    (args.out_dir / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.mirror_dir:
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        for name in ("raw_cells.csv", "mesh_summary.csv", "aggregate.json"):
            shutil.copy2(args.out_dir / name, args.mirror_dir / name)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
