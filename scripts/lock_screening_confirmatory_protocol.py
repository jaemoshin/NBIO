#!/usr/bin/env python
"""Create the immutable machine-readable and human-readable protocol before training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocols"
JSON_PATH = PROTOCOL_DIR / "screening_law_confirmatory_v1.json"
MARKDOWN_PATH = PROTOCOL_DIR / "screening_law_confirmatory_v1.md"
DATA_MANIFEST = PROTOCOL_DIR / "screening_law_confirmatory_data_manifest.json"
PROTOCOL_SOURCE = ROOT / "PROTOCOL_CONFIRMATORY_V1.md"
CODE_FILES = (
    "src/models/nbio_screening_law.py",
    "src/models/nbio_kappa_conditioned.py",
    "src/models/__init__.py",
    "scripts/confirmatory_gen_data.py",
    "scripts/seal_screening_confirmatory_data.py",
    "scripts/train_screening_confirmatory.py",
    "scripts/seal_screening_confirmatory_checkpoints.py",
    "scripts/eval_screening_confirmatory_family.py",
    "scripts/aggregate_screening_confirmatory.py",
    "scripts/submit_screening_confirmatory.sh",
    "scripts/test_screening_confirmatory_model.py",
    "scripts/test_screening_confirmatory_aggregate.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if JSON_PATH.exists() or MARKDOWN_PATH.exists():
        raise FileExistsError("confirmatory_v1 protocol is write-once and already exists")
    data = json.loads(DATA_MANIFEST.read_text())
    if data["status"] != "SEALED_BEFORE_TRAINING":
        raise RuntimeError("data are not sealed")
    code = {
        path: sha256(ROOT / path)
        for path in CODE_FILES
    }
    protocol = {
        "protocol": "screening_law_confirmatory_v1",
        "status": "LOCKED_BEFORE_TRAINING",
        "branch": "research/nbio-screening-law-confirmatory",
        "safe_base": "nbio-tmlr-safe-freeze@3835d1c",
        "manuscript_frozen": True,
        "exploratory_checkpoints_reused": False,
        "data_manifest_path": str(DATA_MANIFEST.relative_to(ROOT)),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "protocol_source_path": str(PROTOCOL_SOURCE.relative_to(ROOT)),
        "protocol_source_sha256": sha256(PROTOCOL_SOURCE),
        "code_sha256": code,
        "methods": {
            "primary_constrained": "screening_mix8",
            "secondary_factorization": "radial_mlp",
            "generic_controls": ["full_prior", "no_prior"],
            "numerical_ceiling": "correct_yukawa_bie",
            "excluded_from_primary": ["screening_mix16"],
        },
        "training": {
            "data_seed_indices": [0, 1, 2],
            "data_seeds": [91001, 91002, 91003],
            "validation_seeds": [92001, 92002, 92003],
            "model_initialization_seeds": [51001, 51002, 51003],
            "batch_order_seeds": [52001, 52002, 52003],
            "samples_per_seed": 2400,
            "validation_samples_per_seed": 400,
            "batch_size": 32,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "schedule": "cosine per optimizer update",
            "optimizer_updates": 7500,
            "examples_seen": 240000,
            "validation_every_updates": 375,
            "checkpoint_rule": "lowest validation mean per-instance relative L2",
            "same_batches_within_paired_seed": True,
            "generic_control_feature_mode": "smooth",
            "dtype": "float32",
            "allow_tf32": False,
        },
        "confirmation": {
            "geometry_family_seeds": [95001, 95002, 95003, 95004],
            "geometry_families": ["in_dist", "highmode", "rough", "deform"],
            "instances_per_family": 24,
            "loads": ["point_source", "fourier"],
            "kappa_test": [0.15, 0.45, 0.85, 1.35, 1.9, 2.55, 3.25, 3.9],
            "kappa_primary": [1.35, 1.9, 2.55, 3.25, 3.9],
            "kappa_extra": [4.5, 5.25, 6.0],
            "query_margins": [0.12, 0.02],
            "boundary_resolutions": [64, 128, 256],
            "fourier_reference_resolution": 512,
            "primary_families": ["in_dist", "highmode", "rough", "deform"],
            "ood_families": ["highmode", "rough", "deform"],
            "underresolved_stress": {"query_margin": 0.02, "boundary_resolution": 64},
            "peek_policy": (
                "No confirmation values or per-family aggregates may be inspected until all 12 "
                "validation-selected checkpoint hashes are frozen and all family evaluations finish."
            ),
        },
        "metrics_and_gates": {
            "primary_metric": "mean per-instance relative L2",
            "uncertainty": "paired family/geometry-cluster bootstrap, 5000 draws, 95% interval",
            "H1": (
                "radial_mlp pooled-four-family bulk kappa[1,4] at n_boundary=128 <= 0.8 * "
                "min(full_prior,no_prior); every seed beats its better paired control; "
                "family/geometry-cluster bootstrap ratio upper bound < 1"
            ),
            "H2_competitive": "screening_mix8 bulk kappa[1,4] <= 1.5 * radial_mlp, aggregate and every seed",
            "H2_extrapolation": (
                "mean extra/in-range degradation is lower than radial_mlp, aggregate and every "
                "seed; per-kappa generic-control ratios are reported but not gated"
            ),
            "near_boundary": (
                "pooled-four-family mix8 near n=256 <= 0.6 * no_prior near n=256 and mix8 "
                "n=256 <= mix8 n=128, aggregate and every seed"
            ),
            "stability": {
                "nonfinite_predictions": 0,
                "failed_solves": 0,
                "max_relative_residual": 1e-5,
                "max_condition_ratio": 1.5,
                "hidden_failure": (
                    "no resolved cell where candidate mean error exceeds 2x both generic controls; "
                    "radial claim checks in-range cells, mix8 safety checks in-range and extra cells"
                ),
                "discrete_perturbation_margin": "reported, not a success gate",
            },
            "outcomes": {
                "A": "radial and mixture candidates pass",
                "B": "radial candidate passes; mixture candidate fails",
                "C": "mixture candidate passes; radial candidate fails",
                "D": "neither passes; kill method development",
            },
            "paperize_outcomes": ["A", "B", "C"],
        },
    }
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    MARKDOWN_PATH.write_text("""# Screening-Law NBIO confirmatory v1 protocol

Status: **LOCKED BEFORE TRAINING**

Branch: `research/nbio-screening-law-confirmatory`
Safe base: `nbio-tmlr-safe-freeze` (`3835d1c`)

The current manuscript remains frozen. The killed exploratory tranche supplies only the prospective
hypothesis; no exploratory checkpoint, confirmation value, or post-hoc threshold is reused.

## Frozen models

- Primary constrained candidate: J8 positive monotone screening mixture.
- Secondary factorization control: unconstrained radial MLP.
- Generic learned controls: conditioned full Laplace prior and conditioned no-prior BIE.
- Numerical ceiling: exact Yukawa BIE. It neither initializes nor supervises learned models.
- J16 is excluded from primary confirmation.

## Equal training contract

Each method uses 2,400 examples, batch size 32, Adam at 1e-3, cosine scheduling, 7,500 optimizer updates,
240,000 examples seen, validation every 375 updates, float32 with TF32 disabled, and the same deterministic
batch order within each paired seed. Checkpoints are selected only by validation mean per-instance relative
L2.

## Blinded confirmation

Fresh kappa values are `{0.15,0.45,0.85,1.35,1.9,2.55,3.25,3.9}` with extra-range safety values
`{4.5,5.25,6.0}`. The matrix contains in-distribution, high-mode, rough, and deformed geometries; point-source
and kappa-independent Fourier traces; margins 0.12 and 0.02; and boundary resolutions 64, 128, and 256.
The `margin=0.02,n=64` cell is an underresolution stress test and is not a primary ranking cell.

No confirmation value may be inspected until all 12 training runs complete, validation-selected checkpoint
hashes are frozen, and all family evaluations finish. One automated aggregation then issues Outcome A-D.

## Gates

- H1: radial MLP is at least 20% better than the better generic control on primary bulk error pooled across
  all four geometry families at n=128, every seed improves, and the family/geometry-cluster bootstrap ratio
  has upper 95% bound below one.
- H2: J8 is within 1.5x of radial MLP in range and its extra/in-range degradation is lower than radial MLP
  in aggregate and every seed. Per-kappa generic-control ratios are reported but are not success gates.
- Near boundary: J8 at n=256 is at least 40% better than no-prior and no worse than J8 at n=128, in aggregate
  and every seed.
- Stability: zero nonfinite predictions and failed solves, residual below 1e-5, condition ratio at most 1.5,
  and no resolved hidden cell more than 2x worse than both generic controls.

Outcome A supports radial factorization and the constrained safety trade-off. Outcome B supports radial
factorization only. Outcome C supports the constrained operating point only. Outcome D kills further method
development and leaves the frozen paper unchanged.

The machine-readable protocol and all data/code hashes are in `protocols/screening_law_confirmatory_v1.json`
and `protocols/screening_law_confirmatory_data_manifest.json`.
""")
    print(f"locked protocol: {JSON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
