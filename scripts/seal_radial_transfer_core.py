#!/usr/bin/env python3
"""Seal fresh data or trained checkpoints before transfer-core evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
)
PROTOCOL = ROOT / "protocols" / "radial_transfer_core_v1.json"
DATA_ROOT = ROOT / "data_cache" / "radial_transfer_core_v1"
DATA_MANIFEST = ROOT / "protocols" / "radial_transfer_core_data_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seal_data() -> None:
    records = []
    expected = {"asymptotic": 120, "transformer": 120}
    for study, count in expected.items():
        paths = sorted((DATA_ROOT / study / "holdout").glob("*.pt"))
        if len(paths) != count:
            raise RuntimeError(f"{study}: expected {count} shards, found {len(paths)}")
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            required = {"Xb", "Nb", "wb", "curv", "g", "Xq", "uq", "kappa"}
            if set(payload) != required:
                raise RuntimeError(f"unexpected keys in {path}: {sorted(payload)}")
            if len(payload["uq"]) != 24:
                raise RuntimeError(f"unexpected sample count in {path}")
            records.append({
                "study": study,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    write_once({
        "status": "SEALED_BEFORE_HOLDOUT_EVALUATION",
        "protocol_sha256": sha256(PROTOCOL),
        "n_files": len(records),
        "files": records,
    }, DATA_MANIFEST)


def seal_checkpoints(component: str) -> None:
    specification = {
        "asymptotic": ("radial_asymptotic", 1),
        "transformer": ("kappa_boundary_transformer", 3),
        "transformer_v2": ("kappa_boundary_transformer_v2", 3),
    }
    method, expected = specification[component]
    component_protocol = (
        ROOT / "protocols" / "radial_transformer_normalized_v2.json"
        if component == "transformer_v2" else PROTOCOL
    )
    component_root = RESULT_ROOT / {
        "asymptotic": "asymptotic_smoke",
        "transformer": "transformer",
        "transformer_v2": "transformer_v2",
    }[component]
    records = []
    for run in sorted((component_root / "training").glob(f"{method}_s*")):
        checkpoint_path = run / "best.pt"
        manifest_path = run / "run_manifest.json"
        if not checkpoint_path.exists() or not manifest_path.exists():
            continue
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        run_manifest = json.loads(manifest_path.read_text())
        if sha256(checkpoint_path) != run_manifest["checkpoint_sha256"]:
            raise RuntimeError(f"run-manifest hash mismatch: {checkpoint_path}")
        records.append({
            "method": method,
            "seed_index": int(checkpoint["config"]["seed_index"]),
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": sha256(checkpoint_path),
            "validation_rel_l2": float(checkpoint["validation_rel_l2"]),
            "update": int(checkpoint["update"]),
        })
    if len(records) != expected:
        raise RuntimeError(f"{component}: expected {expected} checkpoints, found {len(records)}")
    write_once({
        "status": "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT",
        "component": component,
        "protocol_sha256": sha256(component_protocol),
        "records": records,
    }, component_root / "checkpoint_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("data", "asymptotic", "transformer", "transformer_v2"),
        required=True,
    )
    args = parser.parse_args()
    if args.component == "data":
        seal_data()
    else:
        seal_checkpoints(args.component)


if __name__ == "__main__":
    main()
