#!/usr/bin/env python3
"""Aggregate radial transfer-core work packages without changing frozen metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402

RESULT_ROOT = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
)
FAMILIES = ("in_dist", "highmode", "rough", "deform")
REGIMES = (
    "single_obstacle_exterior",
    "two_hole_interior",
    "two_obstacle_exterior",
    "four_obstacle_exterior",
)


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        for key in (
            "seed_index", "n_boundary", "sample_id",
            "n_boundary_per_component", "n_components", "n_boundary_total",
            "geometry_id",
        ):
            if key in row:
                row[key] = int(row[key])
        for key in (
            "kappa", "distance", "rel_l2", "relative_residual",
            "condition_number", "exact_condition_number", "condition_ratio",
        ):
            if key in row:
                row[key] = float(row[key])
        if "finite" in row:
            row["finite"] = row["finite"].lower() == "true"
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def grouped_summary(
    rows: list[dict], group_keys: tuple[str, ...],
) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row["rel_l2"])
    summary = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        array = np.asarray(values)
        summary.append({
            **dict(zip(group_keys, key)),
            "n_rows": len(array),
            "mean_rel_l2": float(array.mean()),
            "median_rel_l2": float(np.median(array)),
            "p90_rel_l2": float(np.quantile(array, 0.9)),
            "max_rel_l2": float(array.max()),
        })
    return summary


def mean_for(rows, method, *, resolution_key, resolution) -> float:
    values = [
        row["rel_l2"] for row in rows
        if row["method"] == method and row[resolution_key] == resolution
    ]
    if not values:
        raise RuntimeError(f"empty mean selection: {method} {resolution}")
    return float(np.mean(values))


def aggregate_asymptotic() -> None:
    root = RESULT_ROOT / "asymptotic_smoke"
    outputs = (
        root / "bulk_summary.csv",
        root / "qbx_summary.csv",
        root / "verdict.json",
        ROOT / "reports" / "radial_asymptotic_smoke.md",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("asymptotic aggregate already exists")
    bulk = read_rows([root / "bulk_raw" / f"{family}.csv" for family in FAMILIES])
    qbx = read_rows([
        root / "qbx_raw" / f"{family}.csv" for family in ("in_dist", "rough")
    ])
    stability = read_rows([
        root / "bulk_raw" / f"{family}_stability.csv" for family in FAMILIES
    ])
    expected_bulk = 4 * 3600
    expected_qbx = 2 * 4320
    if len(bulk) != expected_bulk or len(qbx) != expected_qbx:
        raise RuntimeError(
            f"coverage mismatch bulk={len(bulk)}/{expected_bulk} "
            f"qbx={len(qbx)}/{expected_qbx}"
        )
    all_finite = all(row["finite"] for row in bulk + qbx)
    candidate = mean_for(
        bulk, "radial_asymptotic", resolution_key="n_boundary", resolution=128,
    )
    original = mean_for(
        bulk, "radial_normal", resolution_key="n_boundary", resolution=128,
    )
    refinement = {
        resolution: mean_for(
            bulk, "radial_asymptotic",
            resolution_key="n_boundary", resolution=resolution,
        )
        for resolution in (128, 256, 512)
    }
    qbx_candidate = float(np.mean([
        row["rel_l2"] for row in qbx
        if row["method"] == "radial_asymptotic_split_qbx"
    ]))
    qbx_original = float(np.mean([
        row["rel_l2"] for row in qbx
        if row["method"] == "radial_normal_split_qbx"
    ]))
    candidate_stability = [
        row for row in stability if row["method"] == "radial_asymptotic"
    ]
    primary_conditions = [
        row["condition_ratio"] for row in candidate_stability
        if row["n_boundary"] == 128
    ]
    residuals = [row["relative_residual"] for row in candidate_stability]
    manifest = json.loads((root / "checkpoint_manifest.json").read_text())
    record = manifest["records"][0]
    checkpoint = torch.load(
        ROOT / record["path"], map_location="cpu", weights_only=False,
    )
    model = build_model(checkpoint["config"]).double()
    model.load_state_dict(checkpoint["state_dict"])
    endpoint = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    endpoint_value = model.m_law(endpoint)
    endpoint_derivative = torch.autograd.grad(
        endpoint_value.sum(), endpoint,
    )[0]
    endpoint_error = max(
        abs(float(endpoint_value) - 1.0),
        abs(float(endpoint_derivative)),
    )
    gates = {
        "A1_bulk_within_1p25x_original": candidate / original <= 1.25,
        "A2_split_qbx_at_least_2x_better": qbx_candidate / qbx_original <= 0.5,
        "A3_strict_refinement": (
            refinement[128] > refinement[256] > refinement[512]
        ),
        "A4_condition_ratio_below_1p1": max(primary_conditions) < 1.1,
        "A5_no_failed_or_nonfinite_solves": (
            all_finite
            and all(np.isfinite(value) and value <= 1e-5 for value in residuals)
        ),
        "A6_endpoint_constraints": endpoint_error <= 1e-10,
    }
    verdict = "CONTINUE" if all(gates.values()) else "KILL"
    write_csv(
        grouped_summary(
            bulk,
            ("method", "family", "load_family", "kappa", "n_boundary"),
        ),
        root / "bulk_summary.csv",
    )
    write_csv(
        grouped_summary(
            qbx,
            (
                "method", "family", "load_family", "kappa",
                "distance", "n_boundary",
            ),
        ),
        root / "qbx_summary.csv",
    )
    payload = {
        "component": "asymptotic_smoke",
        "coverage": {
            "bulk_rows": len(bulk),
            "qbx_rows": len(qbx),
            "stability_rows": len(stability),
        },
        "bulk_n_boundary_128": {
            "radial_asymptotic": candidate,
            "radial_normal": original,
            "ratio": candidate / original,
        },
        "bulk_refinement": refinement,
        "split_qbx": {
            "radial_asymptotic": qbx_candidate,
            "radial_normal": qbx_original,
            "ratio": qbx_candidate / qbx_original,
        },
        "stability": {
            "max_condition_ratio_at_128": max(primary_conditions),
            "max_relative_residual": max(residuals),
        },
        "endpoint": {
            "value": float(endpoint_value),
            "derivative": float(endpoint_derivative),
            "max_error": endpoint_error,
        },
        "gates": gates,
        "verdict": verdict,
    }
    write_json(payload, root / "verdict.json")
    report = ROOT / "reports" / "radial_asymptotic_smoke.md"
    report.write_text(f"""# Asymptotic-aware radial smoke

Verdict: **{verdict}**

| metric | asymptotic | original radial | ratio |
|---|---:|---:|---:|
| bulk rel-L2 at `n_b=128` | `{candidate:.4e}` | `{original:.4e}` | `{candidate / original:.3f}` |
| pooled split-QBX rel-L2 | `{qbx_candidate:.4e}` | `{qbx_original:.4e}` | `{qbx_candidate / qbx_original:.3f}` |

Candidate refinement: `128={refinement[128]:.4e}`,
`256={refinement[256]:.4e}`, `512={refinement[512]:.4e}`.

Endpoint value/derivative: `{float(endpoint_value):.12g}` /
`{float(endpoint_derivative):.3e}`. Maximum condition ratio at 128:
`{max(primary_conditions):.6f}`. Maximum solve residual:
`{max(residuals):.3e}`.

Preregistered gates:

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```

Per protocol, the candidate advances to paired three-seed training only if
every gate passes.
""")


def aggregate_topology() -> None:
    root = RESULT_ROOT / "topology_transfer"
    outputs = (
        root / "summary.csv",
        root / "verdict.json",
        ROOT / "reports" / "radial_topology_transfer.md",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("topology aggregate already exists")
    rows = read_rows([root / "raw" / f"{regime}.csv" for regime in REGIMES])
    stability = read_rows([
        root / "raw" / f"{regime}_stability.csv" for regime in REGIMES
    ])
    expected = 4 * 2880
    if len(rows) != expected:
        raise RuntimeError(f"topology coverage mismatch {len(rows)} != {expected}")
    primary = {
        method: mean_for(
            rows, method,
            resolution_key="n_boundary_per_component", resolution=128,
        )
        for method in (
            "exact_yukawa_bie", "radial_normal",
            "generic_matched", "generic_no_prior_matched",
        )
    }
    ratios = {
        method: primary["radial_normal"] / primary[method]
        for method in (
            "exact_yukawa_bie", "generic_matched", "generic_no_prior_matched",
        )
    }
    all_finite = all(row["finite"] for row in rows)
    residuals = [row["relative_residual"] for row in stability]
    gates = {
        "T1_radial_within_4x_exact": ratios["exact_yukawa_bie"] <= 4.0,
        "T2_at_least_10x_better_than_generic_full": (
            ratios["generic_matched"] <= 0.1
        ),
        "T3_at_least_10x_better_than_prior_free": (
            ratios["generic_no_prior_matched"] <= 0.1
        ),
        "T4_no_nonfinite_or_failed_solve": (
            all_finite and all(
                np.isfinite(value) and value <= 1e-5 for value in residuals
            )
        ),
    }
    verdict = (
        "PAPERIZE" if all(gates.values())
        else "NARROW" if gates["T2_at_least_10x_better_than_generic_full"]
        and gates["T3_at_least_10x_better_than_prior_free"]
        and gates["T4_no_nonfinite_or_failed_solve"]
        else "KILL"
    )
    summary = grouped_summary(
        rows,
        (
            "method", "regime", "load_family", "kappa",
            "n_boundary_per_component",
        ),
    )
    write_csv(summary, root / "summary.csv")
    regime_primary = {}
    for regime in REGIMES:
        regime_rows = [row for row in rows if row["regime"] == regime]
        regime_primary[regime] = {
            method: mean_for(
                regime_rows, method,
                resolution_key="n_boundary_per_component", resolution=128,
            )
            for method in primary
        }
    payload = {
        "component": "topology_transfer",
        "coverage_rows": len(rows),
        "primary_mean_rel_l2": primary,
        "primary_ratios": ratios,
        "per_regime_primary": regime_primary,
        "max_relative_residual": max(residuals),
        "gates": gates,
        "verdict": verdict,
    }
    write_json(payload, root / "verdict.json")
    table = "\n".join(
        f"| {regime} | {values['exact_yukawa_bie']:.3e} | "
        f"{values['radial_normal']:.3e} | {values['generic_matched']:.3e} | "
        f"{values['generic_no_prior_matched']:.3e} |"
        for regime, values in regime_primary.items()
    )
    (ROOT / "reports" / "radial_topology_transfer.md").write_text(
        f"""# Frozen interior-to-exterior and topology transfer

Verdict: **{verdict}**

All learned checkpoints were trained only on simply connected interior
Dirichlet problems. No checkpoint was retrained or selected here.

| regime | exact BIE | radial law | generic prior | generic prior-free |
|---|---:|---:|---:|---:|
{table}

Pooled radial/exact ratio: `{ratios['exact_yukawa_bie']:.3f}`.
Radial/generic-prior ratio: `{ratios['generic_matched']:.4f}`.
Radial/prior-free ratio: `{ratios['generic_no_prior_matched']:.4f}`.
Maximum solve residual: `{max(residuals):.3e}`.

```json
{json.dumps(gates, indent=2, sort_keys=True)}
```
"""
    )


def aggregate_transformer(version: str = "v1") -> None:
    if version not in {"v1", "v2"}:
        raise ValueError(version)
    suffix = "transformer" if version == "v1" else "transformer_v2"
    transformer_method = (
        "kappa_boundary_transformer"
        if version == "v1" else "kappa_boundary_transformer_v2"
    )
    report_name = (
        "radial_transformer_baseline.md"
        if version == "v1" else "radial_transformer_baseline_v2.md"
    )
    root = RESULT_ROOT / suffix
    outputs = (
        root / "summary.csv",
        root / "verdict.json",
        ROOT / "reports" / report_name,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("transformer aggregate already exists")
    rows = read_rows([root / "bulk_raw" / f"{family}.csv" for family in FAMILIES])
    expected = 4 * 9360
    if len(rows) != expected:
        raise RuntimeError(f"transformer coverage mismatch {len(rows)} != {expected}")
    methods = (
        "exact_yukawa_bie", "radial_normal", "generic_matched",
        "generic_no_prior_matched", transformer_method,
    )
    primary = {
        method: mean_for(
            rows, method, resolution_key="n_boundary", resolution=128,
        )
        for method in methods
    }
    seed_means = {
        method: {
            seed: float(np.mean([
                row["rel_l2"] for row in rows
                if row["method"] == method
                and row["seed_index"] == seed
                and row["n_boundary"] == 128
            ]))
            for seed in (0, 1, 2)
        }
        for method in methods if method != "exact_yukawa_bie"
    }
    write_csv(
        grouped_summary(
            rows,
            ("method", "family", "load_family", "kappa", "n_boundary"),
        ),
        root / "summary.csv",
    )
    transformer_over_radial = (
        primary[transformer_method] / primary["radial_normal"]
    )
    adequacy = {
        "mean_below_0p2": primary[transformer_method] < 0.2,
        "all_seed_means_below_0p25": all(
            value < 0.25 for value in seed_means[transformer_method].values()
        ),
    }
    verdict = "PAPERIZE" if all(adequacy.values()) else "KILL"
    payload = {
        "component": f"transformer_baseline_{version}",
        "coverage_rows": len(rows),
        "primary_mean_rel_l2": primary,
        "seed_mean_rel_l2": seed_means,
        "transformer_over_radial": transformer_over_radial,
        "adequacy_gates": adequacy,
        "all_finite": all(row["finite"] for row in rows),
        "verdict": verdict,
    }
    write_json(payload, root / "verdict.json")
    (ROOT / "reports" / report_name).write_text(
        f"""# Same-protocol kappa-conditioned Boundary Transformer {version}

| method | mean rel-L2 at `n_b=128` |
|---|---:|
| exact modified-Helmholtz BIE | `{primary['exact_yukawa_bie']:.4e}` |
| radial normal form | `{primary['radial_normal']:.4e}` |
| generic Laplace-prior correction | `{primary['generic_matched']:.4e}` |
| generic prior-free learned kernel | `{primary['generic_no_prior_matched']:.4e}` |
| kappa-conditioned Boundary Transformer {version} | `{primary[transformer_method]:.4e}` |

The amortized Transformer/radial error ratio is
`{transformer_over_radial:.2f}x`. Its online latency is reported separately
from NBIO assembly and solution because the computational contracts differ.

Per-seed means:

```json
{json.dumps(seed_means, indent=2, sort_keys=True)}
```

Adequacy gates: `{json.dumps(adequacy, sort_keys=True)}`.

Verdict: **{verdict}** as a same-protocol contextual baseline; this does not
convert the paper into a universal neural-operator SOTA claim.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("asymptotic", "topology", "transformer", "transformer_v2"),
        required=True,
    )
    args = parser.parse_args()
    {
        "asymptotic": aggregate_asymptotic,
        "topology": aggregate_topology,
        "transformer": aggregate_transformer,
        "transformer_v2": lambda: aggregate_transformer("v2"),
    }[args.component]()


if __name__ == "__main__":
    main()
