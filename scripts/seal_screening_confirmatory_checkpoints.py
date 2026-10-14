#!/usr/bin/env python
"""Freeze all checkpoint hashes before the blinded confirmation is evaluated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "confirmatory_v1"
METHODS = ("full_prior", "no_prior", "screening_mix8", "radial_mlp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = RESULT_ROOT / "checkpoint_manifest.json"
    if output.exists():
        raise FileExistsError(f"write-once checkpoint manifest exists: {output}")
    records = []
    for method in METHODS:
        for seed_index in range(3):
            run = RESULT_ROOT / "training" / f"{method}_s{seed_index}"
            checkpoint = run / "best.pt"
            manifest_path = run / "run_manifest.json"
            if not checkpoint.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"incomplete training run: {run}")
            manifest = json.loads(manifest_path.read_text())
            if manifest["max_updates"] != 7500 or manifest["batch_size"] != 32:
                raise RuntimeError(f"unequal computational contract: {run}")
            if manifest["checkpoint_sha256"] != sha256(checkpoint):
                raise RuntimeError(f"checkpoint changed after training: {checkpoint}")
            records.append({
                "method": method,
                "seed_index": seed_index,
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": sha256(checkpoint),
                "n_parameters": manifest["n_parameters"],
                "best_validation_rel_l2": manifest["best_validation_rel_l2"],
                "max_updates": manifest["max_updates"],
                "examples_seen": manifest["examples_seen"],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "CHECKPOINTS_FROZEN_BEFORE_CONFIRMATION",
        "records": records,
    }, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(records)} checkpoints in {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
