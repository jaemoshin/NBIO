#!/usr/bin/env python
"""Single-shot locked aggregation and verdict for confirmatory_v1."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "confirmatory_v1"
REPORT_PATH = ROOT / "reports" / "screening_law_confirmatory_v1_TRANCHE_SUMMARY.md"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
METHODS = ("full_prior", "no_prior", "screening_mix8", "radial_mlp")
KAPPA_IN_PRIMARY = (1.35, 1.9, 2.55, 3.25, 3.9)
KAPPA_IN_ALL = (0.15, 0.45, 0.85) + KAPPA_IN_PRIMARY
KAPPA_EXTRA = (4.5, 5.25, 6.0)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("kappa", "query_margin", "rel_l2"):
            row[key] = float(row[key])
        for key in ("seed_index", "n_boundary", "sample_id"):
            row[key] = int(row[key])
        row["finite"] = row["finite"] == "True"
    return rows


def selected(rows, method, *, family=None, kappas=None, margin=None,
             resolutions=None, loads=None, seed=None):
    output = []
    for row in rows:
        if row["method"] != method:
            continue
        if family is not None and row["family"] != family:
            continue
        if kappas is not None and row["kappa"] not in kappas:
            continue
        if margin is not None and abs(row["query_margin"] - margin) > 1e-12:
            continue
        if resolutions is not None and row["n_boundary"] not in resolutions:
            continue
        if loads is not None and row["load_family"] not in loads:
            continue
        if seed is not None and row["seed_index"] != seed:
            continue
        output.append(row)
    if not output:
        raise RuntimeError(f"empty selection for {method}")
    return output


def mean_error(rows, method, **filters):
    return float(np.mean([row["rel_l2"] for row in selected(rows, method, **filters)]))


def cluster_values(rows, method, **filters):
    grouped = defaultdict(list)
    for row in selected(rows, method, **filters):
        grouped[(row["family"], row["sample_id"])].append(row["rel_l2"])
    return {cluster: float(np.mean(values)) for cluster, values in grouped.items()}


def ratio_bootstrap(target: dict, control: dict, seed=63001, draws=5000):
    keys = sorted(set(target) & set(control))
    target_values = np.asarray([target[key] for key in keys])
    control_values = np.asarray([control[key] for key in keys])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    ratios = target_values[indices].mean(axis=1) / np.maximum(
        control_values[indices].mean(axis=1), 1e-15,
    )
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["method"], row["family"], row["load_family"], row["kappa"],
            row["query_margin"], row["n_boundary"],
        )].append(row["rel_l2"])
    output = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values)
        output.append({
            "method": key[0],
            "family": key[1],
            "load_family": key[2],
            "kappa": key[3],
            "query_margin": key[4],
            "n_boundary": key[5],
            "n_rows": len(array),
            "mean_rel_l2": float(array.mean()),
            "median_rel_l2": float(np.median(array)),
            "p90_rel_l2": float(np.quantile(array, 0.9)),
            "std_rel_l2": float(array.std(ddof=1)),
        })
    return output


def write_csv(rows, path):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hidden_failures(rows, method, include_out_of_range):
    grouped = defaultdict(list)
    for row in rows:
        if row["n_boundary"] == 64:
            continue
        if not include_out_of_range and row["kappa"] not in KAPPA_IN_ALL:
            continue
        if include_out_of_range and row["kappa"] not in KAPPA_IN_ALL + KAPPA_EXTRA:
            continue
        key = (
            row["family"], row["load_family"], row["kappa"],
            row["query_margin"], row["n_boundary"], row["method"],
        )
        grouped[key].append(row["rel_l2"])
    failures = []
    cell_keys = sorted({key[:-1] for key in grouped})
    for cell in cell_keys:
        candidate = float(np.mean(grouped[cell + (method,)]))
        full = float(np.mean(grouped[cell + ("full_prior",)]))
        none = float(np.mean(grouped[cell + ("no_prior",)]))
        if candidate > 2.0 * full and candidate > 2.0 * none:
            failures.append({
                "family": cell[0], "load_family": cell[1], "kappa": cell[2],
                "query_margin": cell[3], "n_boundary": cell[4],
                "candidate": candidate, "full_prior": full, "no_prior": none,
            })
    return failures


def main():
    output_paths = (
        RESULT_ROOT / "aggregate_summary.csv",
        RESULT_ROOT / "verdict.json",
        REPORT_PATH,
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError("write-once aggregate output already exists")
    rows = []
    expected = {family: 41184 for family in FAMILIES}
    coverage = {}
    for family in FAMILIES:
        family_rows = read_csv(RESULT_ROOT / "raw" / f"{family}.csv")
        coverage[family] = len(family_rows)
        if len(family_rows) != expected[family]:
            raise RuntimeError(f"row-count mismatch for {family}: {len(family_rows)}")
        rows.extend(family_rows)

    all_finite = all(row["finite"] and np.isfinite(row["rel_l2"]) for row in rows)
    common_primary = dict(
        kappas=KAPPA_IN_PRIMARY, margin=0.12, resolutions=(128,),
        loads=("point_source", "fourier"),
    )
    bulk = {
        method: mean_error(rows, method, **common_primary) for method in METHODS
    }
    best_control = min(("full_prior", "no_prior"), key=lambda method: bulk[method])
    h1_ratio = bulk["radial_mlp"] / bulk[best_control]
    h1_seed_ratios = []
    for seed in range(3):
        radial = mean_error(rows, "radial_mlp", seed=seed, **common_primary)
        full = mean_error(rows, "full_prior", seed=seed, **common_primary)
        none = mean_error(rows, "no_prior", seed=seed, **common_primary)
        h1_seed_ratios.append(radial / min(full, none))
    h1_ci = ratio_bootstrap(
        cluster_values(rows, "radial_mlp", **common_primary),
        cluster_values(rows, best_control, **common_primary),
    )
    h1_pass = h1_ratio <= 0.8 and max(h1_seed_ratios) < 1.0 and h1_ci[1] < 1.0

    h2_competitive_ratio = bulk["screening_mix8"] / bulk["radial_mlp"]
    h2_seed_competitive = []
    inrange_seed = {}
    for seed in range(3):
        mix = mean_error(rows, "screening_mix8", seed=seed, **common_primary)
        radial = mean_error(rows, "radial_mlp", seed=seed, **common_primary)
        h2_seed_competitive.append(mix / radial)
        inrange_seed[seed] = {"mix": mix, "radial": radial}

    oor_guard = {}
    oor_seed_guard = []
    mix_oor_values = []
    radial_oor_values = []
    degradation_seed = []
    for kappa in KAPPA_EXTRA:
        filters = dict(
            kappas=(kappa,), margin=0.12, resolutions=(128,),
            loads=("point_source", "fourier"),
        )
        values = {method: mean_error(rows, method, **filters) for method in METHODS}
        guard_ratio = values["screening_mix8"] / min(values["full_prior"], values["no_prior"])
        oor_guard[str(kappa)] = {"errors": values, "guard_ratio": guard_ratio,
                                "pass": guard_ratio <= 2.0}
        mix_oor_values.append(values["screening_mix8"])
        radial_oor_values.append(values["radial_mlp"])
        for seed in range(3):
            mix_seed = mean_error(rows, "screening_mix8", seed=seed, **filters)
            full_seed = mean_error(rows, "full_prior", seed=seed, **filters)
            none_seed = mean_error(rows, "no_prior", seed=seed, **filters)
            oor_seed_guard.append({
                "kappa": kappa,
                "seed": seed,
                "ratio": mix_seed / min(full_seed, none_seed),
                "pass": mix_seed <= 2.0 * min(full_seed, none_seed),
            })
    for seed in range(3):
        mix_extra = np.mean([
            mean_error(rows, "screening_mix8", seed=seed, kappas=(kappa,), margin=0.12,
                       resolutions=(128,), loads=("point_source", "fourier"))
            for kappa in KAPPA_EXTRA
        ])
        radial_extra = np.mean([
            mean_error(rows, "radial_mlp", seed=seed, kappas=(kappa,), margin=0.12,
                       resolutions=(128,), loads=("point_source", "fourier"))
            for kappa in KAPPA_EXTRA
        ])
        degradation_seed.append({
            "seed": seed,
            "mix": mix_extra / inrange_seed[seed]["mix"],
            "radial": radial_extra / inrange_seed[seed]["radial"],
        })
    mix_degradation = float(np.mean(mix_oor_values)) / bulk["screening_mix8"]
    radial_degradation = float(np.mean(radial_oor_values)) / bulk["radial_mlp"]
    h2_pass = (
        h2_competitive_ratio <= 1.5
        and max(h2_seed_competitive) <= 1.5
        and mix_degradation < radial_degradation
        and all(row["mix"] < row["radial"] for row in degradation_seed)
    )

    near_filters = dict(
        kappas=KAPPA_IN_PRIMARY, margin=0.02,
        loads=("point_source", "fourier"),
    )
    near_mix_128 = mean_error(rows, "screening_mix8", resolutions=(128,), **near_filters)
    near_mix_256 = mean_error(rows, "screening_mix8", resolutions=(256,), **near_filters)
    near_none_256 = mean_error(rows, "no_prior", resolutions=(256,), **near_filters)
    near_seed_pass = []
    for seed in range(3):
        mix128 = mean_error(rows, "screening_mix8", seed=seed, resolutions=(128,), **near_filters)
        mix256 = mean_error(rows, "screening_mix8", seed=seed, resolutions=(256,), **near_filters)
        none256 = mean_error(rows, "no_prior", seed=seed, resolutions=(256,), **near_filters)
        near_seed_pass.append(mix256 <= 0.6 * none256 and mix256 <= mix128)
    near_pass = (
        near_mix_256 <= 0.6 * near_none_256
        and near_mix_256 <= near_mix_128
        and all(near_seed_pass)
    )

    with (RESULT_ROOT / "raw" / "stability.csv").open(newline="") as handle:
        stability = list(csv.DictReader(handle))
    if len(stability) != 1056:
        raise RuntimeError(f"stability row-count mismatch: {len(stability)}")
    for row in stability:
        for key in ("relative_residual", "condition_ratio", "discrete_perturbation_margin"):
            row[key] = float(row[key])
    max_residual = max(row["relative_residual"] for row in stability)
    max_condition_ratio = max(row["condition_ratio"] for row in stability)
    radial_hidden = hidden_failures(rows, "radial_mlp", include_out_of_range=False)
    mix_hidden = hidden_failures(rows, "screening_mix8", include_out_of_range=True)
    common_stability_pass = all_finite and max_residual < 1e-5 and max_condition_ratio <= 1.5
    radial_pass = h1_pass and common_stability_pass and not radial_hidden
    mixture_pass = h2_pass and near_pass and common_stability_pass and not mix_hidden

    if radial_pass and mixture_pass:
        outcome = "A"
        verdict = "PAPERIZE"
    elif radial_pass:
        outcome = "B"
        verdict = "PAPERIZE"
    elif mixture_pass:
        outcome = "C"
        verdict = "PAPERIZE"
    else:
        outcome = "D"
        verdict = "KILL"

    exact_ceiling = mean_error(
        rows, "correct_yukawa_bie", **common_primary,
    )
    ood_bulk = {
        family: {
            method: mean_error(rows, method, family=family, **common_primary)
            for method in METHODS + ("correct_yukawa_bie",)
        }
        for family in ("highmode", "rough", "deform")
    }
    result = {
        "protocol": "screening_law_confirmatory_v1",
        "coverage": coverage,
        "all_finite": all_finite,
        "primary_bulk": bulk,
        "exact_bie_ceiling": exact_ceiling,
        "ood_bulk": ood_bulk,
        "H1_radial_factorization": {
            "best_generic_control": best_control,
            "aggregate_ratio": h1_ratio,
            "seed_ratios": h1_seed_ratios,
            "family_geometry_cluster_bootstrap_ratio_ci95": h1_ci,
            "pass": h1_pass,
        },
        "H2_monotonicity_tradeoff": {
            "inrange_mix_to_radial_ratio": h2_competitive_ratio,
            "seed_competitive_ratios": h2_seed_competitive,
            "out_of_range_guard": oor_guard,
            "out_of_range_seed_guard": oor_seed_guard,
            "mix_degradation_ratio": mix_degradation,
            "radial_degradation_ratio": radial_degradation,
            "seed_degradation_ratios": degradation_seed,
            "pass": h2_pass,
        },
        "near_boundary": {
            "mix_n128": near_mix_128,
            "mix_n256": near_mix_256,
            "no_prior_n256": near_none_256,
            "seed_pass": near_seed_pass,
            "pass": near_pass,
        },
        "stability": {
            "max_relative_residual": max_residual,
            "max_condition_ratio": max_condition_ratio,
            "max_discrete_perturbation_margin": max(
                row["discrete_perturbation_margin"] for row in stability
            ),
            "radial_hidden_failures": radial_hidden,
            "mixture_hidden_failures": mix_hidden,
            "common_pass": common_stability_pass,
        },
        "radial_candidate_pass": radial_pass,
        "mixture_candidate_pass": mixture_pass,
        "outcome": outcome,
        "verdict": verdict,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(summarize(rows), RESULT_ROOT / "aggregate_summary.csv")
    (RESULT_ROOT / "verdict.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(f"""# Screening-Law NBIO confirmatory v1

Verdict: **{verdict}** (Outcome {outcome})

This is the single automated report from the prospectively locked confirmation. The manuscript remained
frozen while the study ran.

## Primary results

| method | in-range bulk rel-L2 |
|---|---:|
| exact Yukawa BIE | `{exact_ceiling:.4e}` |
| generic full prior | `{bulk['full_prior']:.4e}` |
| generic no prior | `{bulk['no_prior']:.4e}` |
| radial MLP | `{bulk['radial_mlp']:.4e}` |
| monotone J8 | `{bulk['screening_mix8']:.4e}` |

- H1 radial factorization: **{'PASS' if h1_pass else 'FAIL'}**; ratio to the better generic control
  `{h1_ratio:.3f}`, 95% geometry-cluster bootstrap interval `{h1_ci[0]:.3f}-{h1_ci[1]:.3f}`.
- H2 monotonicity trade-off: **{'PASS' if h2_pass else 'FAIL'}**; in-range J8/radial ratio
  `{h2_competitive_ratio:.3f}`, extrapolation degradation J8 `{mix_degradation:.2f}` versus radial
  `{radial_degradation:.2f}`.
- Refined near-boundary criterion: **{'PASS' if near_pass else 'FAIL'}**; J8 n=256
  `{near_mix_256:.4e}` versus no-prior `{near_none_256:.4e}`.
- Stability: residual max `{max_residual:.3e}`, condition-ratio max `{max_condition_ratio:.3f}`.
- Hidden resolved failures: radial `{len(radial_hidden)}`, J8 `{len(mix_hidden)}`.

## Claim decision

Outcome A supports radial factorization plus a constrained extrapolation trade-off. Outcome B supports only
radial normal-form factorization. Outcome C supports two operating points only if the constrained model's
safety claim passes. Outcome D kills further screening-law development and leaves the frozen NBIO paper
unchanged.

Raw outputs and the full gate record are under `results/screening_law/confirmatory_v1/`.
""")
    print(json.dumps({"outcome": outcome, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
