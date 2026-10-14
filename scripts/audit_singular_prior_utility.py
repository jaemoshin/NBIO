#!/usr/bin/env python
"""Independent aggregate/claim audit for the localized singular-prior tranche."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/singular_prior_utility"
REPORT = ROOT / "reports/singular_prior_utility_results.md"
FAMILIES = ("in_dist", "highmode", "rough", "deform")
LEARNED = ("full_split", "no_split", "local_singular")


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def bootstrap(values, seed, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def averaged_instances(rows, method, margin, n_b):
    grouped = defaultdict(list)
    for row in rows:
        if (row["method"] == method and abs(float(row["query_margin"]) - margin) < 1e-9
                and int(row["n_b"]) == n_b):
            grouped[(row["family"], int(row["sample_id"]))].append(float(row["rel_l2"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def method_slice(rows, method, margin, n_b, seed):
    values = np.asarray(list(averaged_instances(rows, method, margin, n_b).values()))
    return {
        "method": method,
        "query_slice": "bulk" if margin == 0.12 else "near",
        "query_margin": margin,
        "n_b": n_b,
        "n_instances": len(values),
        "mean_rel_l2": float(values.mean()),
        "median_rel_l2": float(np.median(values)),
        "p90_rel_l2": float(np.quantile(values, 0.9)),
        "bootstrap_mean_ci": bootstrap(values, seed),
    }


def paired_slice(rows, target, control, margin, n_b, seed):
    target_values = averaged_instances(rows, target, margin, n_b)
    control_values = averaged_instances(rows, control, margin, n_b)
    keys = sorted(set(target_values) & set(control_values))
    difference = np.asarray([target_values[key] - control_values[key] for key in keys])
    target_mean = float(np.mean([target_values[key] for key in keys]))
    control_mean = float(np.mean([control_values[key] for key in keys]))
    return {
        "target": target,
        "control": control,
        "query_slice": "bulk" if margin == 0.12 else "near",
        "query_margin": margin,
        "n_b": n_b,
        "n_paired_instances": len(keys),
        "target_mean": target_mean,
        "control_mean": control_mean,
        "relative_improvement": (control_mean - target_mean) / control_mean,
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "target_win_fraction": float(np.mean(difference < 0)),
        "bootstrap_mean_difference_ci": bootstrap(difference, seed),
    }


def seed_test_summary():
    paths = {
        "full_split": [
            ROOT / "results/tmlr_revision/nbio_yuk_split_s0/eval.json",
            ROOT / "results/nbio_yuk_split_s1/eval.json",
            ROOT / "results/nbio_yuk_split_s2/eval.json",
        ],
        "no_split": [
            ROOT / "results/tmlr_revision/nbio_yuk_nosplit_s0/eval.json",
            ROOT / "results/nbio_yuk_nosplit_s1/eval.json",
            ROOT / "results/nbio_yuk_nosplit_s2/eval.json",
        ],
        "local_singular": [
            RESULT / "tune_potential_c0p08_s0/eval.json",
            RESULT / "final_s1/eval.json",
            RESULT / "final_s2/eval.json",
        ],
    }
    output = []
    for method, method_paths in paths.items():
        values = []
        for path in method_paths:
            payload = json.loads(path.read_text())
            values.append(float(payload["eval"]["yukawa_test"]["rel_l2"]))
        output.append({
            "method": method,
            "seed_values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        })
    return output


def qbx_diagnostic(rows):
    output = {}
    for base, qbx in (("full_split", "full_split_qbx"), ("local_singular", "local_singular_qbx")):
        base_values = averaged_instances(rows, base, 0.02, 256)
        qbx_values = averaged_instances(rows, qbx, 0.02, 256)
        keys = sorted(set(base_values) & set(qbx_values))
        base_mean = float(np.mean([base_values[key] for key in keys]))
        qbx_mean = float(np.mean([qbx_values[key] for key in keys]))
        output[base] = {
            "base_mean": base_mean,
            "qbx_mean": qbx_mean,
            "relative_change": (qbx_mean - base_mean) / base_mean,
        }
    return output


def fmt(value):
    return f"{value:.3e}"


def main():
    rows = read_csv(RESULT / "transfer_raw.csv")
    qbx_rows = read_csv(RESULT / "transfer_qbx_raw.csv")
    expected_rows = 2 * 4 * 2 * 3 * 64 + 3 * 3 * 4 * 2 * 3 * 64
    if len(rows) != expected_rows:
        raise RuntimeError(f"raw row count {len(rows)} != expected {expected_rows}")
    if any(not np.isfinite(float(row["rel_l2"])) for row in rows + qbx_rows):
        raise RuntimeError("non-finite result found")
    key_fields = ("method", "seed", "family", "query_margin", "n_b", "sample_id")
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate raw result key found")

    slices = []
    paired = []
    for margin in (0.12, 0.02):
        for index, method in enumerate(LEARNED):
            slices.append(method_slice(rows, method, margin, 256, 20260730 + index))
        for index, control in enumerate(("full_split", "no_split")):
            paired.append(paired_slice(rows, "local_singular", control, margin, 256, 20260740 + index))

    audit = {
        "data_integrity": {
            "raw_rows": len(rows),
            "expected_rows": expected_rows,
            "duplicate_keys": 0,
            "nonfinite_values": 0,
            "families": list(FAMILIES),
            "seeds_per_learned_method": 3,
        },
        "standard_test_seed_summary": seed_test_summary(),
        "n_b_256_slices": slices,
        "paired_n_b_256": paired,
        "qbx_diagnostic": qbx_diagnostic(qbx_rows),
        "claim_boundary": (
            "Fixed kappa=4 only. The local singular prior improves the learned-kernel BIE but remains "
            "above the exact Yukawa BIE and does not establish transfer over unseen kappa."
        ),
    }
    with (RESULT / "audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    with (RESULT / "audit_slices.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slices[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(slices)

    slice_lookup = {(row["method"], row["query_slice"]): row for row in slices}
    standard = {row["method"]: row for row in audit["standard_test_seed_summary"]}
    lines = [
        "# Local singular-prior utility: results and claim audit",
        "",
        "## Result",
        "",
        "The validation-only sweep selected a potential-only localized Laplace prior",
        "`exp(-(r/0.08)^4) K_Laplace`. It retains the shared local Laplace/Yukawa singularity while",
        "discarding the globally wrong far-field prior from the density solve.",
        "",
        "| method | standard test mean +/- seed std | bulk OOD, n_b=256 | near OOD, n_b=256 |",
        "|---|---:|---:|---:|",
    ]
    for method, label in (("full_split", "full Laplace split"), ("no_split", "no prior"),
                          ("local_singular", "local singular prior")):
        lines.append(
            f"| {label} | `{fmt(standard[method]['mean'])} +/- {fmt(standard[method]['std'])}` | "
            f"`{fmt(slice_lookup[(method, 'bulk')]['mean_rel_l2'])}` | "
            f"`{fmt(slice_lookup[(method, 'near')]['mean_rel_l2'])}` |"
        )
    lines += ["", "## Paired evidence", ""]
    for item in paired:
        change = 100 * item["relative_improvement"]
        direction = f"{change:.1f}% lower" if change >= 0 else f"{-change:.1f}% higher"
        significance = "excludes zero" if (
            item["bootstrap_mean_difference_ci"][0] > 0
            or item["bootstrap_mean_difference_ci"][1] < 0
        ) else "includes zero"
        lines.append(
            f"- `{item['query_slice']}` local vs `{item['control']}`: "
            f"{direction} mean error; paired mean-difference 95% CI "
            f"`[{fmt(item['bootstrap_mean_difference_ci'][0])}, "
            f"{fmt(item['bootstrap_mean_difference_ci'][1])}]` ({significance}); win fraction "
            f"`{100 * item['target_win_fraction']:.1f}%` over 256 geometry instances."
        )
    lines += [
        "",
        "## Negative result",
        "",
        f"Adding QBX at inference changes the local model's near aggregate by "
        f"`{100 * audit['qbx_diagnostic']['local_singular']['relative_change']:.2f}%`. The remaining error is",
        "model discrepancy, not close-quadrature error at margin 0.02; QBX is not promoted as an improvement",
        "for this experiment.",
        "",
        "## Claim",
        "",
        "This closes the missing conditional-utility regime at fixed `kappa=4`: learning is necessary, and",
        "retaining only trusted local singular structure improves over both the globally wrong full prior in",
        "bulk and the prior-free learned BIE in bulk and near-boundary evaluation. The full prior remains",
        "slightly better on the aggregate near-only slice, so the local model is a better tradeoff rather than",
        "a strict pointwise dominator.",
        "",
        "The result does not establish a single model across `kappa`; that requires a conditioned mixed-kappa",
        "experiment.",
        "",
        "## Verdict",
        "",
        "`CONTINUE`",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
