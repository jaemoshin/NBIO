#!/usr/bin/env python3
"""Aggregate the frozen endpoint-regular confirmation and registered gates."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models import build_model


RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_endpoint_v1"
REPORT = ROOT / "reports" / "radial_endpoint_revision_results.md"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
METHODS = (
    "generic_matched", "generic_no_prior_matched", "radial_separate",
    "radial_normal", "radial_endpoint", "exact_yukawa_bie",
)
LEARNED = METHODS[:-1]
KAPPAS = (1.3, 1.9, 2.65, 3.35, 3.95)
LOADS = ("point_source", "fourier")
RESOLUTIONS = (64, 128, 256)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed_index"] = int(row["seed_index"])
        row["kappa"] = float(row["kappa"])
        row["n_boundary"] = int(row["n_boundary"])
        row["sample_id"] = int(row["sample_id"])
        row["rel_l2"] = float(row["rel_l2"])
        row["finite"] = row["finite"].lower() == "true"
    return rows


def chosen(rows, method, *, resolution=128, seed=None, family=None, load=None, kappa=None):
    return [row for row in rows if (
        row["method"] == method and row["n_boundary"] == resolution
        and (seed is None or row["seed_index"] == seed)
        and (family is None or row["family"] == family)
        and (load is None or row["load_family"] == load)
        and (kappa is None or row["kappa"] == kappa)
    )]


def mean_error(rows, method, **filters) -> float:
    values = [row["rel_l2"] for row in chosen(rows, method, **filters)]
    if not values:
        raise RuntimeError(f"empty selection: {method} {filters}")
    return float(np.mean(values))


def hierarchical_ratio_ci(rows, numerator, denominator, seed, draws=5000):
    lookup = {}
    for row in rows:
        if row["n_boundary"] != 128 or row["method"] not in {numerator, denominator}:
            continue
        key = (
            row["method"], row["seed_index"], row["family"], row["sample_id"],
            row["load_family"], row["kappa"],
        )
        lookup[key] = row["rel_l2"]
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws)
    seeds = np.arange(3)
    for draw in range(draws):
        sampled_seeds = rng.choice(seeds, size=3, replace=True)
        num_values, den_values = [], []
        for family in FAMILIES:
            sampled_geometries = rng.integers(0, 32, size=32)
            for learned_seed in sampled_seeds:
                for sample_id in sampled_geometries:
                    for load in LOADS:
                        for kappa in KAPPAS:
                            num_values.append(lookup[(
                                numerator, int(learned_seed), family, int(sample_id), load, kappa,
                            )])
                            den_values.append(lookup[(
                                denominator, int(learned_seed), family, int(sample_id), load, kappa,
                            )])
        ratios[draw] = np.mean(num_values) / np.mean(den_values)
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def endpoint_derivatives() -> dict[str, list[float]]:
    manifest = json.loads((RESULT_ROOT / "checkpoint_manifest.json").read_text())
    values = defaultdict(list)
    for record in manifest["records"]:
        if record["method"] not in {"radial_normal", "radial_endpoint"}:
            continue
        checkpoint = torch.load(ROOT / record["path"], map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"]).double()
        model.load_state_dict(checkpoint["state_dict"])
        s0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.grad(model.m_law(s0).sum(), s0)[0]
        values[record["method"]].append(float(derivative))
    return dict(values)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    outputs = (RESULT_ROOT / "summary.csv", RESULT_ROOT / "verdict.json", REPORT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("write-once endpoint aggregate exists")
    rows = []
    coverage = {}
    for family in FAMILIES:
        family_rows = read_rows(RESULT_ROOT / "raw" / f"{family}.csv")
        coverage[family] = len(family_rows)
        if len(family_rows) != 15360:
            raise RuntimeError(f"row-count mismatch for {family}: {len(family_rows)}")
        rows.extend(family_rows)
    all_finite = all(row["finite"] and np.isfinite(row["rel_l2"]) for row in rows)
    primary = {method: mean_error(rows, method) for method in METHODS}
    seed_means = {
        method: [mean_error(rows, method, seed=seed) for seed in range(3)]
        for method in LEARNED
    }
    derivatives = endpoint_derivatives()
    derivative_pass = max(abs(value) for value in derivatives["radial_endpoint"]) <= 1e-10

    ratios = {
        control: primary["radial_endpoint"] / primary[control]
        for control in ("generic_matched", "generic_no_prior_matched", "radial_normal", "exact_yukawa_bie")
    }
    generic_seed_pass = all(
        seed_means["radial_endpoint"][seed] < seed_means[control][seed]
        for control in ("generic_matched", "generic_no_prior_matched")
        for seed in range(3)
    )
    ci_full = hierarchical_ratio_ci(rows, "radial_endpoint", "generic_matched", 115001)
    ci_none = hierarchical_ratio_ci(rows, "radial_endpoint", "generic_no_prior_matched", 115002)

    with (RESULT_ROOT / "raw" / "stability.csv").open(newline="") as handle:
        stability = list(csv.DictReader(handle))
    endpoint_stability = [row for row in stability if row["method"] == "radial_endpoint"]
    max_residual = max(float(row["relative_residual"]) for row in endpoint_stability)
    max_condition_ratio = max(float(row["condition_ratio"]) for row in endpoint_stability)
    margins = [float(row["laplace_perturbation_margin"]) for row in endpoint_stability]

    hidden_failures = []
    for family in FAMILIES:
        for load in LOADS:
            for kappa in KAPPAS:
                for resolution in RESOLUTIONS:
                    endpoint = mean_error(
                        rows, "radial_endpoint", family=family, load=load,
                        kappa=kappa, resolution=resolution,
                    )
                    full = mean_error(
                        rows, "generic_matched", family=family, load=load,
                        kappa=kappa, resolution=resolution,
                    )
                    none = mean_error(
                        rows, "generic_no_prior_matched", family=family, load=load,
                        kappa=kappa, resolution=resolution,
                    )
                    if endpoint > 2 * full and endpoint > 2 * none:
                        hidden_failures.append({
                            "family": family, "load": load, "kappa": kappa,
                            "n_boundary": resolution, "radial_endpoint": endpoint,
                            "generic_matched": full, "generic_no_prior_matched": none,
                        })

    gates = {
        "R1_endpoint_derivative": derivative_pass,
        "R2_both_generic_ratios_below_0p8": (
            ratios["generic_matched"] <= 0.8
            and ratios["generic_no_prior_matched"] <= 0.8
            and generic_seed_pass and ci_full[1] < 1 and ci_none[1] < 1
        ),
        "R3_accuracy_preserved_and_near_exact": (
            ratios["radial_normal"] <= 1.5 and ratios["exact_yukawa_bie"] <= 3.0
        ),
        "R4_no_hidden_or_nonfinite_failure": all_finite and not hidden_failures,
        "R5_stability": max_residual <= 1e-5 and max_condition_ratio <= 1.5,
    }
    verdict = "PAPERIZE" if all(gates.values()) else (
        "NARROW" if gates["R1_endpoint_derivative"] and gates["R2_both_generic_ratios_below_0p8"] else "KILL"
    )

    summary_rows = []
    for method in METHODS:
        for family in FAMILIES:
            for load in LOADS:
                for kappa in KAPPAS:
                    for resolution in RESOLUTIONS:
                        values = [row["rel_l2"] for row in chosen(
                            rows, method, family=family, load=load,
                            kappa=kappa, resolution=resolution,
                        )]
                        summary_rows.append({
                            "method": method, "family": family, "load_family": load,
                            "kappa": kappa, "n_boundary": resolution, "n_rows": len(values),
                            "mean_rel_l2": float(np.mean(values)),
                            "median_rel_l2": float(np.median(values)),
                            "p90_rel_l2": float(np.quantile(values, 0.9)),
                        })
    write_csv(summary_rows, RESULT_ROOT / "summary.csv")
    result = {
        "protocol": "radial_endpoint_v1",
        "coverage": coverage,
        "all_finite": all_finite,
        "primary_resolution": 128,
        "n_test_geometries": 128,
        "n_geometry_load_kappa_instances_per_seed": 1280,
        "primary_mean_rel_l2": primary,
        "seed_mean_rel_l2": seed_means,
        "endpoint_derivatives": derivatives,
        "ratios": ratios,
        "hierarchical_bootstrap_ci95": {
            "endpoint_over_generic_full": ci_full,
            "endpoint_over_generic_no_prior": ci_none,
        },
        "stability": {
            "max_relative_residual": max_residual,
            "max_condition_ratio": max_condition_ratio,
            "laplace_perturbation_margin_mean": float(np.mean(margins)),
            "laplace_perturbation_margin_median": float(np.median(margins)),
            "laplace_perturbation_margin_p90": float(np.quantile(margins, 0.9)),
            "laplace_perturbation_margin_max": float(np.max(margins)),
            "fraction_margin_below_one": float(np.mean(np.asarray(margins) < 1)),
        },
        "hidden_failures": hidden_failures,
        "gates": gates,
        "verdict": verdict,
    }
    (RESULT_ROOT / "verdict.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    REPORT.write_text(f"""# Endpoint-regular radial NBIO confirmation

Verdict: **{verdict}**

The endpoint-matched law enforces `m(0)=1` and `m'(0)=0` without changing the 2,241-parameter count.
It is trained on the original sealed train/validation data and evaluated on a fresh holdout of 128 geometries,
1,280 geometry/load/kappa instances per seed at the primary `n_b=128`, and separate `64/256` resolution rows.

| method | primary mean rel-L2 |
|---|---:|
| exact modified-Helmholtz BIE | `{primary['exact_yukawa_bie']:.4e}` |
| generic Laplace-prior correction | `{primary['generic_matched']:.4e}` |
| generic prior-free learned kernel | `{primary['generic_no_prior_matched']:.4e}` |
| radial separate `(kappa,r)` | `{primary['radial_separate']:.4e}` |
| original radial `kappa*r` | `{primary['radial_normal']:.4e}` |
| **endpoint-regular radial `kappa*r`** | **`{primary['radial_endpoint']:.4e}`** |

- Endpoint derivatives, old/new: `{derivatives}`.
- Endpoint/full-prior ratio `{ratios['generic_matched']:.4f}`, hierarchical 95% interval
  `{ci_full[0]:.4f}-{ci_full[1]:.4f}`.
- Endpoint/no-prior ratio `{ratios['generic_no_prior_matched']:.4f}`, hierarchical 95% interval
  `{ci_none[0]:.4f}-{ci_none[1]:.4f}`.
- Endpoint/original-radial ratio `{ratios['radial_normal']:.4f}`; endpoint/exact ratio
  `{ratios['exact_yukawa_bie']:.3f}`.
- Maximum residual `{max_residual:.3e}`; maximum condition ratio `{max_condition_ratio:.3f}`.
- Laplace perturbation margin mean/p90/max: `{np.mean(margins):.3f}` / `{np.quantile(margins, 0.9):.3f}` /
  `{np.max(margins):.3f}`. This is a sufficient-condition diagnostic, not a required gate.
- Hidden resolved failures: `{len(hidden_failures)}`.

The bootstrap resamples training seed and geometry hierarchically. All loads and kappa values, and each
instance's complete 256-query error vector, remain attached to the resampled geometry.
""")
    print(json.dumps({"verdict": verdict, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
