#!/usr/bin/env python
"""Seal spline checkpoints and untouched confirmation tensors before evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols" / "reviewer3_evidence_v1.json"
RESULT_ROOT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1"
DATA_ROOT = ROOT / "data_cache" / "reviewer3_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    checkpoint_output = RESULT_ROOT / "spline_checkpoint_manifest.json"
    data_output = RESULT_ROOT / "confirmation_data_manifest.json"
    if checkpoint_output.exists() or data_output.exists():
        raise FileExistsError("reviewer-3 artifact manifest already exists")

    checkpoints = []
    for seed_index in range(3):
        run = RESULT_ROOT / "training" / f"spline_s{seed_index}"
        checkpoint = run / "best.pt"
        manifest = json.loads((run / "run_manifest.json").read_text())
        if manifest["smoke"] or manifest["max_updates"] != 7500:
            raise RuntimeError(f"non-confirmatory training run: {run}")
        if manifest["n_parameters"] != 2241 or manifest["examples_seen"] != 240000:
            raise RuntimeError(f"training contract mismatch: {run}")
        if manifest["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError(f"checkpoint hash mismatch: {run}")
        checkpoints.append({
            "method": "radial_spline",
            "seed_index": seed_index,
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "n_parameters": 2241,
            "best_validation_rel_l2": manifest["best_validation_rel_l2"],
        })

    data_files = sorted(DATA_ROOT.glob("*.pt"))
    metadata_files = sorted(DATA_ROOT.glob("*_geometry_metadata.json"))
    if len(data_files) != 120 or len(metadata_files) != 4:
        raise RuntimeError(
            f"confirmation coverage mismatch: {len(data_files)} tensors, "
            f"{len(metadata_files)} metadata files"
        )
    expected_seeds = set(protocol["fresh_confirmation"]["family_seeds"])
    old_seeds = {110201, 110202, 110203, 110204}
    if expected_seeds & old_seeds:
        raise RuntimeError("fresh and original family seed blocks overlap")
    fingerprints = []
    for path in metadata_files:
        payload = json.loads(path.read_text())
        if payload["seed"] not in expected_seeds:
            raise RuntimeError(f"unexpected family seed in {path}")
        fingerprints.extend(payload["geometry_fingerprints"])
    if len(fingerprints) != 128 or len(set(fingerprints)) != 128:
        raise RuntimeError("confirmation geometry fingerprints are not unique")

    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output.write_text(json.dumps({
        "status": "CHECKPOINTS_FROZEN_BEFORE_CONFIRMATION",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "records": checkpoints,
    }, indent=2, sort_keys=True) + "\n")
    data_output.write_text(json.dumps({
        "status": "UNTOUCHED_CONFIRMATION_DATA_SEALED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "seed_overlap_with_original": False,
        "unique_geometry_fingerprints": len(set(fingerprints)),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in data_files + metadata_files
        ],
    }, indent=2, sort_keys=True) + "\n")
    print("sealed 3 spline checkpoints and 128-geometry confirmation bank")


if __name__ == "__main__":
    main()
