#!/usr/bin/env python3
"""Write-once sealing for fresh holdout data and combined checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_endpoint_v1"
OLD_RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_paperization_v1"
DATA_ROOT = ROOT / "data_cache" / "radial_endpoint_v1" / "holdout"
DATA_MANIFEST = ROOT / "protocols" / "radial_endpoint_data_manifest.json"
CHECKPOINT_MANIFEST = RESULT_ROOT / "checkpoint_manifest.json"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
LOADS = ("point_source", "fourier")
KAPPAS = (1.3, 1.9, 2.65, 3.35, 3.95)
RESOLUTIONS = (64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def seal_data() -> None:
    if DATA_MANIFEST.exists():
        raise FileExistsError(DATA_MANIFEST)
    files = []
    for family in FAMILIES:
        for load in LOADS:
            for kappa in KAPPAS:
                for resolution in RESOLUTIONS:
                    path = DATA_ROOT / f"{load}_{family}_k{tag(kappa)}_nb{resolution}.pt"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    shard = torch.load(path, map_location="cpu", weights_only=False)
                    if shard["Xb"].shape[0] != 32 or shard["Xb"].shape[1] != resolution:
                        raise RuntimeError(f"bad shard shape: {path}")
                    files.append({
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path),
                        "n_geometries": 32,
                    })
    DATA_MANIFEST.write_text(json.dumps({
        "status": "SEALED_BEFORE_EVALUATION",
        "files": files,
        "families": FAMILIES,
        "loads": LOADS,
        "kappas": KAPPAS,
        "resolutions": RESOLUTIONS,
        "family_seeds": (114201, 114202, 114203, 114204),
    }, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(files)} fresh holdout shards")


def seal_checkpoints() -> None:
    if CHECKPOINT_MANIFEST.exists():
        raise FileExistsError(CHECKPOINT_MANIFEST)
    old_manifest = json.loads((OLD_RESULT_ROOT / "checkpoint_manifest.json").read_text())
    records = []
    for record in old_manifest["records"]:
        if record["method"] not in {"generic_matched", "radial_separate", "radial_normal"}:
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"old checkpoint hash mismatch: {path}")
        records.append(record)
    for method in ("generic_no_prior_matched", "radial_endpoint"):
        for seed_index in range(3):
            path = RESULT_ROOT / "training" / f"{method}_s{seed_index}" / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint["config"]["method"] != method:
                raise RuntimeError(f"method mismatch: {path}")
            records.append({
                "method": method,
                "seed_index": seed_index,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "n_parameters": sum(value.numel() for value in checkpoint["state_dict"].values()),
                "validation_rel_l2": checkpoint["validation_rel_l2"],
            })
    if len(records) != 15:
        raise RuntimeError(f"expected 15 combined checkpoints, found {len(records)}")
    CHECKPOINT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_MANIFEST.write_text(json.dumps({
        "status": "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT",
        "records": records,
    }, indent=2, sort_keys=True) + "\n")
    print("sealed 15 old/new checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("data", "checkpoints"))
    args = parser.parse_args()
    if args.stage == "data":
        seal_data()
    else:
        seal_checkpoints()


if __name__ == "__main__":
    main()
