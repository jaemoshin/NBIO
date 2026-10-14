#!/usr/bin/env python
"""Validate names and freeze hashes without reading confirmation values."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_cache" / "screening_confirmatory_v1"
OUTPUT = ROOT / "protocols" / "screening_law_confirmatory_data_manifest.json"
KAPPAS = (0.15, 0.45, 0.85, 1.35, 1.9, 2.55, 3.25, 3.9, 4.5, 5.25, 6.0)
FAMILIES = ("in_dist", "highmode", "rough", "deform")
LOADS = ("point_source", "fourier")
MARGINS = ("m120", "m020")
RESOLUTIONS = (64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kappa_tag(kappa: float) -> str:
    return str(kappa).replace(".", "p")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once data manifest exists: {OUTPUT}")
    expected = {
        DATA_ROOT / f"data_s{seed_index}" / filename
        for seed_index in range(3)
        for filename in ("yukawa_train.pt", "yukawa_val.pt", "yukawa_test.pt")
    }
    expected.update({
        DATA_ROOT / "confirmation" / (
            f"{load}_{family}_k{kappa_tag(kappa)}_{margin}_nb{resolution}.pt"
        )
        for load in LOADS
        for family in FAMILIES
        for kappa in KAPPAS
        for margin in MARGINS
        for resolution in RESOLUTIONS
    })
    observed = set(DATA_ROOT.glob("data_s*/*.pt")) | set(
        (DATA_ROOT / "confirmation").glob("*.pt")
    )
    missing = sorted(str(path.relative_to(ROOT)) for path in expected - observed)
    unexpected = sorted(str(path.relative_to(ROOT)) for path in observed - expected)
    if missing or unexpected:
        raise RuntimeError(f"data coverage mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
    generator_manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    files = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": (
                "confirmation" if "confirmation" in path.parts
                else "pipeline_check" if path.name == "yukawa_test.pt"
                else "training" if path.name == "yukawa_train.pt"
                else "validation"
            ),
        }
        for path in sorted(expected)
    ]
    payload = {
        "protocol": "screening_law_confirmatory_v1",
        "status": "SEALED_BEFORE_TRAINING",
        "generator_manifest": generator_manifest,
        "train_range": [0.0, 4.0],
        "train_count_per_seed": 2400,
        "validation_count_per_seed": 400,
        "confirmation_count_per_family": 24,
        "kappa_test": list(KAPPAS[:8]),
        "kappa_extra": list(KAPPAS[8:]),
        "families": list(FAMILIES),
        "load_families": list(LOADS),
        "margins": [0.12, 0.02],
        "resolutions": list(RESOLUTIONS),
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(files)} files without reading confirmation values")


if __name__ == "__main__":
    main()
