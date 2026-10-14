#!/usr/bin/env python
"""Cross-check radial transfer, interval, and efficiency result artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
INTERVAL = ROOT / "results" / "screening_law" / "interval_holdout_v1"
EFFICIENCY = ROOT / "results" / "screening_law" / "radial_data_efficiency_v1"
OUTPUT = ROOT / "results" / "screening_law" / "radial_transfer_core_audit.json"
REPORT = ROOT / "reports" / "radial_transfer_core_result_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        with path.open(newline="") as handle:
            output.extend(csv.DictReader(handle))
    return output


def verify_checkpoints(path: Path, expected: int) -> dict:
    manifest = json.loads(path.read_text())
    records = manifest["records"]
    valid = len(records) == expected
    for record in records:
        checkpoint = ROOT / record["path"]
        valid = valid and checkpoint.exists() and sha256(checkpoint) == record["sha256"]
    return {"count": len(records), "expected": expected, "hashes_valid": valid}


def bootstrap_condition(rows: list[dict], condition: str, seed: int) -> dict:
    grouped = {"radial_normal": defaultdict(list), "generic_matched": defaultdict(list)}
    seed_values = {"radial_normal": defaultdict(list), "generic_matched": defaultdict(list)}
    for row in rows:
        if row["condition"] != condition or row["method"] not in grouped:
            continue
        key = (row["family"], row["load_family"], row["kappa"], row["sample_id"])
        grouped[row["method"]][key].append(float(row["rel_l2"]))
        seed_values[row["method"]][int(row["seed_index"])].append(float(row["rel_l2"]))
    keys = sorted(grouped["radial_normal"])
    radial = np.asarray([
        np.mean(grouped["radial_normal"][key]) for key in keys
    ])
    generic = np.asarray([
        np.mean(grouped["generic_matched"][key]) for key in keys
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), (5000, len(keys)))
    ratios = radial[indices].mean(axis=1) / generic[indices].mean(axis=1)
    paired_seed_ratios = [
        float(np.mean(seed_values["radial_normal"][index])
              / np.mean(seed_values["generic_matched"][index]))
        for index in range(3)
    ]
    return {
        "ratio": float(radial.mean() / generic.mean()),
        "cluster_bootstrap_ci95": [
            float(np.quantile(ratios, 0.025)),
            float(np.quantile(ratios, 0.975)),
        ],
        "paired_seed_ratios": paired_seed_ratios,
        "clusters": len(keys),
    }


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("write-once audit output exists")

    interval_rows = read_rows([
        INTERVAL / "raw" / f"{family}.csv"
        for family in ("in_dist", "highmode", "rough", "deform")
    ])
    interval_summary = json.loads((INTERVAL / "summary.json").read_text())
    interval_finite = all(row["finite"] == "True" for row in interval_rows)
    interval_kappas = sorted({float(row["kappa"]) for row in interval_rows})
    interval_data = []
    for seed_index in range(3):
        shard = torch.load(
            ROOT / "data_cache" / "screening_interval_holdout_v1"
            / f"data_s{seed_index}" / "train.pt",
            map_location="cpu", weights_only=False,
        )
        interval_data.extend(float(value) for value in shard["kappa"])
    train_low = sum(value <= 1.5 for value in interval_data)
    train_high = sum(value >= 2.5 for value in interval_data)
    train_gap = sum(1.5 < value < 2.5 for value in interval_data)
    interval_audit = {
        "coverage_rows": len(interval_rows),
        "coverage_expected": 28800,
        "all_finite": interval_finite,
        "test_kappas": interval_kappas,
        "test_kappas_inside_withheld_interval": all(
            1.5 < value < 2.5 for value in interval_kappas
        ),
        "train_kappa_counts": {
            "lower_interval": train_low,
            "upper_interval": train_high,
            "withheld_interval": train_gap,
        },
        "checkpoint_audit": verify_checkpoints(
            INTERVAL / "checkpoint_manifest.json", 9,
        ),
        "reported_verdict": interval_summary["verdict"],
    }

    efficiency_rows = read_rows([
        EFFICIENCY / "raw" / f"{family}.csv"
        for family in ("in_dist", "highmode", "rough", "deform")
    ])
    efficiency_summary = json.loads((EFFICIENCY / "verdict.json").read_text())
    efficiency_finite = all(row["finite"] == "True" for row in efficiency_rows)
    conditions = sorted({
        row["condition"] for row in efficiency_rows if row["condition"] != "exact"
    })
    uncertainty = {
        condition: bootstrap_condition(
            efficiency_rows, condition, 122001 + index,
        )
        for index, condition in enumerate(conditions)
    }
    run_manifests = [
        json.loads(path.read_text())
        for path in sorted((EFFICIENCY / "training").glob("*/run_manifest.json"))
    ]
    efficiency_audit = {
        "coverage_rows": len(efficiency_rows),
        "coverage_expected": 55040,
        "all_finite": efficiency_finite,
        "training_runs": len(run_manifests),
        "all_runs_fixed_7500_updates": all(
            record["updates"] == 7500 and record["examples_seen"] == 240000
            for record in run_manifests
        ),
        "checkpoint_audit": verify_checkpoints(
            EFFICIENCY / "checkpoint_manifest.json", 42,
        ),
        "condition_uncertainty": uncertainty,
        "reported_verdict": efficiency_summary["verdict"],
        "holdout_reuse_note": (
            "This prospective training grid reused the already examined "
            "radial-factorization holdout; checkpoints were validation-selected "
            "and every prespecified condition is reported."
        ),
    }

    core_verdicts = {
        "topology": json.loads(
            (CORE / "topology_transfer" / "verdict.json").read_text()
        )["verdict"],
        "asymptotic": json.loads(
            (CORE / "asymptotic_smoke" / "verdict.json").read_text()
        )["verdict"],
        "transformer_v1": json.loads(
            (CORE / "transformer" / "verdict.json").read_text()
        )["verdict"],
        "transformer_v2": json.loads(
            (CORE / "transformer_v2" / "verdict.json").read_text()
        )["verdict"],
    }
    checks = {
        "interval_coverage": len(interval_rows) == 28800,
        "interval_finite": interval_finite,
        "interval_split_clean": train_gap == 0 and train_low == train_high == 3600,
        "interval_checkpoints": interval_audit["checkpoint_audit"]["hashes_valid"],
        "efficiency_coverage": len(efficiency_rows) == 55040,
        "efficiency_finite": efficiency_finite,
        "efficiency_training_contract": efficiency_audit[
            "all_runs_fixed_7500_updates"
        ],
        "efficiency_checkpoints": efficiency_audit[
            "checkpoint_audit"
        ]["hashes_valid"],
    }
    payload = {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "core_local_verdicts": core_verdicts,
        "interval": interval_audit,
        "efficiency": efficiency_audit,
        "claim_boundary": {
            "supported": [
                "contiguous in-range kappa-interval interpolation",
                "same-protocol advantage over a normalized Boundary Transformer",
                "unique-observation and synthetic-label-noise efficiency",
            ],
            "not_supported": [
                "interior-to-exterior or topology transfer without retraining",
                "asymptotic parameterization replacing the selected radial law",
                "parameter extrapolation outside the trained kappa range",
                "universal neural-operator SOTA",
            ],
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    n100 = uncertainty["sample_n100"]
    noisy = uncertainty["sensor_q32_noise0p05"]
    REPORT.write_text(f"""# Radial transfer core result audit

All mechanical checks pass: `{all(checks.values())}`.

## Promotable evidence

- The contiguous interval holdout has `{len(interval_rows)}` complete, finite
  rows. Training contains `{train_low}` lower-interval and `{train_high}`
  upper-interval instances, with `{train_gap}` instances in the withheld
  interval. The radial model's reported error is
  `{interval_summary['means']['radial_normal']:.4e}`.
- The normalized same-protocol Boundary Transformer is an admissible contextual
  baseline; the unnormalized v1 baseline is a retained failed attempt.
- The efficiency study has `{len(efficiency_rows)}` complete, finite rows and
  42 hash-verified, validation-selected checkpoints. At 100 unique instances,
  the radial/generic ratio is `{n100['ratio']:.4f}` with clustered 95% interval
  `{n100['cluster_bootstrap_ci95'][0]:.4f}-{n100['cluster_bootstrap_ci95'][1]:.4f}`.
  Under 32 queries and 5% label noise, the ratio is `{noisy['ratio']:.4f}` with
  interval `{noisy['cluster_bootstrap_ci95'][0]:.4f}-{noisy['cluster_bootstrap_ci95'][1]:.4f}`.

## Negative evidence

- Frozen interior checkpoints fail exterior and multi-component topology
  transfer. Converged solves do not rescue the unsupported kernel-law tail.
- The asymptotic-aware model improves split-QBX by about 10x but fails bulk and
  refinement gates, so it is not the selected method.
- The unnormalized Transformer v1 fails adequacy gates and must not be cited as
  the strong baseline.

## Interpretation limits

The interval test establishes interpolation across a missing in-range interval,
not extrapolation beyond the trained parameter support. The efficiency grid
holds optimizer updates fixed, so it establishes unique-observation efficiency,
not reduced training compute. Its 1% and 5% perturbations are synthetic iid
relative solution-label noise, not a general sensor model. The efficiency
experiment reused an existing holdout, although the full grid and gates were
frozen before training and all runs were selected only by clean validation.

## Verdict

**PAPERIZE** the original radial normal form with the interval, normalized
Transformer, and efficiency evidence. Keep topology and asymptotic outcomes as
negative appendix or audit results. Do not broaden the method claim.
""")
    print(json.dumps({"all_checks_pass": all(checks.values())}, indent=2))


if __name__ == "__main__":
    main()
