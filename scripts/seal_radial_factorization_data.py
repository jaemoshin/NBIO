#!/usr/bin/env python
"""Hash-seal the fresh ablation data without reading holdout tensor values."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_cache" / "radial_factorization_ablation_v1"
OUTPUT = ROOT / "protocols" / "radial_factorization_ablation_data_manifest.json"
KAPPAS = (1.2, 1.75, 2.4, 3.1, 3.8)
FAMILIES = ("in_dist", "highmode", "rough", "deform")
LOADS = ("point_source", "fourier")
RESOLUTIONS = (64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    expected = {
        DATA_ROOT / f"data_s{seed}" / name
        for seed in range(3)
        for name in ("yukawa_train.pt", "yukawa_val.pt")
    }
    expected.update({
        DATA_ROOT / "holdout" / f"{load}_{family}_k{tag(kappa)}_nb{n_boundary}.pt"
        for load in LOADS
        for family in FAMILIES
        for kappa in KAPPAS
        for n_boundary in RESOLUTIONS
    })
    observed = set(DATA_ROOT.glob("data_s*/*.pt")) | set(
        (DATA_ROOT / "holdout").glob("*.pt")
    )
    if expected != observed:
        missing = sorted(str(path.relative_to(ROOT)) for path in expected - observed)
        extra = sorted(str(path.relative_to(ROOT)) for path in observed - expected)
        raise RuntimeError(f"data coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    records = []
    for path in sorted(expected):
        records.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": "holdout" if "holdout" in path.parts else (
                "training" if path.name == "yukawa_train.pt" else "validation"
            ),
        })
    OUTPUT.write_text(json.dumps({
        "protocol": "radial_factorization_ablation_v1",
        "status": "SEALED_BEFORE_TRAINING",
        "train_seeds": [110001, 110002, 110003],
        "validation_seeds": [110101, 110102, 110103],
        "family_seeds": [110201, 110202, 110203, 110204],
        "kappa_test": list(KAPPAS),
        "families": list(FAMILIES),
        "loads": list(LOADS),
        "resolutions": list(RESOLUTIONS),
        "query_margin": 0.12,
        "instances_per_family": 32,
        "files": records,
    }, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(records)} data files without reading holdout values")


if __name__ == "__main__":
    main()
