#!/usr/bin/env python
"""Independent audit of the locked Screening-Law NBIO confirmation.

The locked aggregator is preserved verbatim. This audit corrects one mechanical
implementation error: stability must be evaluated for each claimed candidate,
not pooled over candidates and generic controls.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "screening_law" / "confirmatory_v1"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
METHODS = ("full_prior", "no_prior", "screening_mix8", "radial_mlp")
PRIMARY_KAPPAS = (1.35, 1.9, 2.55, 3.25, 3.9)
IN_RANGE_KAPPAS = (0.15, 0.45, 0.85) + PRIMARY_KAPPAS
EXTRA_KAPPAS = (4.5, 5.25, 6.0)


def read_rows() -> list[dict]:
    rows = []
    for family in FAMILIES:
        path = RESULT_ROOT / "raw" / f"{family}.csv"
        with path.open(newline="") as handle:
            family_rows = list(csv.DictReader(handle))
        if len(family_rows) != 41184:
            raise RuntimeError(f"unexpected coverage for {family}: {len(family_rows)}")
        for row in family_rows:
            row["seed_index"] = int(row["seed_index"])
            row["sample_id"] = int(row["sample_id"])
            row["n_boundary"] = int(row["n_boundary"])
            row["kappa"] = float(row["kappa"])
            row["query_margin"] = float(row["query_margin"])
            row["rel_l2"] = float(row["rel_l2"])
            row["finite"] = row["finite"] == "True"
        rows.extend(family_rows)
    return rows


def select(rows: list[dict], method: str, **filters) -> list[dict]:
    selected = []
    for row in rows:
        if row["method"] != method:
            continue
        if any(row[key] not in value if isinstance(value, tuple) else row[key] != value
               for key, value in filters.items()):
            continue
        selected.append(row)
    if not selected:
        raise RuntimeError(f"empty selection for {method}: {filters}")
    return selected


def mean_error(rows: list[dict], method: str, **filters) -> float:
    return float(np.mean([row["rel_l2"] for row in select(rows, method, **filters)]))


def clustered(rows: list[dict], method: str, **filters) -> dict[tuple, float]:
    values = defaultdict(list)
    for row in select(rows, method, **filters):
        values[(row["family"], row["sample_id"])].append(row["rel_l2"])
    return {key: float(np.mean(group)) for key, group in values.items()}


def bootstrap_ratio(target: dict, control: dict, draws: int = 5000) -> list[float]:
    keys = sorted(set(target) & set(control))
    target_values = np.asarray([target[key] for key in keys])
    control_values = np.asarray([control[key] for key in keys])
    rng = np.random.default_rng(63001)
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    ratios = target_values[indices].mean(1) / control_values[indices].mean(1)
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def hidden_failures(rows: list[dict], method: str, kappas: tuple[float, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["n_boundary"] == 64 or row["kappa"] not in kappas:
            continue
        key = (row["family"], row["load_family"], row["kappa"],
               row["query_margin"], row["n_boundary"], row["method"])
        grouped[key].append(row["rel_l2"])
    failures = []
    cells = sorted({key[:-1] for key in grouped})
    for cell in cells:
        candidate = float(np.mean(grouped[cell + (method,)]))
        full = float(np.mean(grouped[cell + ("full_prior",)]))
        none = float(np.mean(grouped[cell + ("no_prior",)]))
        if candidate > 2 * full and candidate > 2 * none:
            failures.append({"cell": cell, "candidate": candidate, "full": full, "none": none})
    return failures


def stability_by_method() -> dict[str, dict]:
    with (RESULT_ROOT / "raw" / "stability.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1056:
        raise RuntimeError(f"unexpected stability coverage: {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    output = {}
    for method, method_rows in grouped.items():
        max_residual = max(float(row["relative_residual"]) for row in method_rows)
        max_condition_ratio = max(float(row["condition_ratio"]) for row in method_rows)
        output[method] = {
            "n_rows": len(method_rows),
            "max_relative_residual": max_residual,
            "max_condition_ratio": max_condition_ratio,
            "max_discrete_perturbation_margin": max(
                float(row["discrete_perturbation_margin"]) for row in method_rows
            ),
            "pass": max_residual < 1e-5 and max_condition_ratio <= 1.5,
        }
    return output


def main() -> None:
    rows = read_rows()
    if not all(row["finite"] and np.isfinite(row["rel_l2"]) for row in rows):
        raise RuntimeError("nonfinite confirmation result")

    primary = {
        "kappa": PRIMARY_KAPPAS,
        "query_margin": 0.12,
        "n_boundary": (128,),
        "load_family": ("point_source", "fourier"),
    }
    bulk = {method: mean_error(rows, method, **primary) for method in METHODS}
    best_control = min(("full_prior", "no_prior"), key=bulk.get)
    h1_ratio = bulk["radial_mlp"] / bulk[best_control]
    h1_seed_ratios = []
    for seed in range(3):
        radial = mean_error(rows, "radial_mlp", seed_index=seed, **primary)
        full = mean_error(rows, "full_prior", seed_index=seed, **primary)
        none = mean_error(rows, "no_prior", seed_index=seed, **primary)
        h1_seed_ratios.append(radial / min(full, none))
    h1_ci = bootstrap_ratio(
        clustered(rows, "radial_mlp", **primary),
        clustered(rows, best_control, **primary),
    )
    h1_pass = h1_ratio <= 0.8 and max(h1_seed_ratios) < 1 and h1_ci[1] < 1

    h2_ratio = bulk["screening_mix8"] / bulk["radial_mlp"]
    h2_seed_ratios = [
        mean_error(rows, "screening_mix8", seed_index=seed, **primary)
        / mean_error(rows, "radial_mlp", seed_index=seed, **primary)
        for seed in range(3)
    ]
    degradation = {}
    degradation_seeds = []
    for method in ("screening_mix8", "radial_mlp"):
        extra = np.mean([
            mean_error(rows, method, kappa=(kappa,), query_margin=0.12,
                       n_boundary=(128,), load_family=("point_source", "fourier"))
            for kappa in EXTRA_KAPPAS
        ])
        degradation[method] = float(extra / bulk[method])
    for seed in range(3):
        item = {"seed": seed}
        for method in ("screening_mix8", "radial_mlp"):
            extra = np.mean([
                mean_error(rows, method, seed_index=seed, kappa=(kappa,), query_margin=0.12,
                           n_boundary=(128,), load_family=("point_source", "fourier"))
                for kappa in EXTRA_KAPPAS
            ])
            item[method] = float(extra / mean_error(rows, method, seed_index=seed, **primary))
        degradation_seeds.append(item)
    h2_pass = (
        h2_ratio <= 1.5
        and max(h2_seed_ratios) <= 1.5
        and degradation["screening_mix8"] < degradation["radial_mlp"]
        and all(item["screening_mix8"] < item["radial_mlp"] for item in degradation_seeds)
    )

    near = {
        "kappa": PRIMARY_KAPPAS,
        "query_margin": 0.02,
        "load_family": ("point_source", "fourier"),
    }
    near_mix128 = mean_error(rows, "screening_mix8", n_boundary=(128,), **near)
    near_mix256 = mean_error(rows, "screening_mix8", n_boundary=(256,), **near)
    near_none256 = mean_error(rows, "no_prior", n_boundary=(256,), **near)
    near_seed_pass = []
    for seed in range(3):
        mix128 = mean_error(rows, "screening_mix8", seed_index=seed, n_boundary=(128,), **near)
        mix256 = mean_error(rows, "screening_mix8", seed_index=seed, n_boundary=(256,), **near)
        none256 = mean_error(rows, "no_prior", seed_index=seed, n_boundary=(256,), **near)
        near_seed_pass.append(mix256 <= 0.6 * none256 and mix256 <= mix128)
    near_pass = near_mix256 <= 0.6 * near_none256 and near_mix256 <= near_mix128 \
        and all(near_seed_pass)

    stability = stability_by_method()
    radial_hidden = hidden_failures(rows, "radial_mlp", IN_RANGE_KAPPAS)
    mix_hidden = hidden_failures(rows, "screening_mix8", IN_RANGE_KAPPAS + EXTRA_KAPPAS)
    radial_pass = h1_pass and stability["radial_mlp"]["pass"] and not radial_hidden
    mixture_pass = h2_pass and near_pass and stability["screening_mix8"]["pass"] and not mix_hidden
    outcome = "A" if radial_pass and mixture_pass else "B" if radial_pass else "C" if mixture_pass else "D"
    verdict = "PAPERIZE" if outcome in {"A", "B", "C"} else "KILL"

    locked = json.loads((RESULT_ROOT / "verdict.json").read_text())
    result = {
        "audit_scope": "candidate-specific correction of locked stability aggregation",
        "locked_verdict_preserved": locked["verdict"],
        "locked_outcome_preserved": locked["outcome"],
        "coverage": {family: 41184 for family in FAMILIES},
        "bulk": bulk,
        "H1": {"ratio": h1_ratio, "seed_ratios": h1_seed_ratios, "ci95": h1_ci,
               "pass": h1_pass},
        "H2": {"ratio": h2_ratio, "seed_ratios": h2_seed_ratios,
               "degradation": degradation, "seed_degradation": degradation_seeds,
               "pass": h2_pass},
        "near": {"mix_n128": near_mix128, "mix_n256": near_mix256,
                 "no_prior_n256": near_none256, "seed_pass": near_seed_pass,
                 "pass": near_pass},
        "stability_by_method": stability,
        "radial_hidden_failures": radial_hidden,
        "mixture_hidden_failures": mix_hidden,
        "corrected_radial_candidate_pass": radial_pass,
        "corrected_mixture_candidate_pass": mixture_pass,
        "corrected_outcome": outcome,
        "corrected_verdict": verdict,
    }
    output = RESULT_ROOT / "independent_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": outcome, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
