#!/usr/bin/env python
"""Seal all matched-ablation checkpoints before holdout evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_paperization_v1"
METHODS = ("generic_matched", "radial_separate", "radial_normal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = RESULT_ROOT / "checkpoint_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    records = []
    for method in METHODS:
        for seed_index in range(3):
            run = RESULT_ROOT / "training" / f"{method}_s{seed_index}"
            checkpoint = run / "best.pt"
            manifest = json.loads((run / "run_manifest.json").read_text())
            expected_parameters = 2221 if method == "generic_matched" else 2241
            if manifest["n_parameters"] != expected_parameters:
                raise RuntimeError(f"parameter contract failed: {run}")
            if manifest["max_updates"] != 7500 or manifest["examples_seen"] != 240000:
                raise RuntimeError(f"training contract failed: {run}")
            if sha256(checkpoint) != manifest["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
            records.append({
                "method": method,
                "seed_index": seed_index,
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": sha256(checkpoint),
                "n_parameters": manifest["n_parameters"],
                "best_validation_rel_l2": manifest["best_validation_rel_l2"],
            })
    output.write_text(json.dumps({
        "status": "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT",
        "records": records,
    }, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(records)} checkpoints")


if __name__ == "__main__":
    main()
