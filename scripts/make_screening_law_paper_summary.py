#!/usr/bin/env python3
"""Create the frozen radial-screening paper figures from committed results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "screening_law"


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    matched = load(RESULTS / "radial_paperization_v1" / "verdict.json")
    interval = load(RESULTS / "interval_holdout_v1" / "summary.json")
    with (
        RESULTS / "reviewer3_evidence_v1" / "tables"
        / "fresh_data_efficiency_summary.csv"
    ).open(newline="") as handle:
        efficiency_rows = list(csv.DictReader(handle))
    efficiency = {
        (row["condition"], row["method"]): float(row["mean"])
        for row in efficiency_rows
    }
    scale = load(RESULTS / "scale_equivariance_v1" / "summary.json")
    inverse = load(RESULTS / "inverse_v1" / "summary.json")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    axes = axes.ravel()

    methods = ["exact_yukawa_bie", "generic_matched", "radial_separate", "radial_normal"]
    labels = ["Exact\nBIE", "Generic\nmatched", "Radial\n(k,r)", "Normal form\nkr"]
    colors = ["#555555", "#c95f45", "#d7a13b", "#287a78"]
    values = [matched["primary_mean_rel_l2"][m] for m in methods]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean relative $L^2$")
    axes[0].set_title("A  Parameter-matched ablation", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    kappas = np.array(sorted(float(k) for k in interval["per_kappa"]))
    axes[1].axvspan(0.0, 1.5, color="#e7e7e7", alpha=0.8, label="training support")
    axes[1].axvspan(2.5, 4.0, color="#e7e7e7", alpha=0.8)
    axes[1].axvspan(1.5, 2.5, color="#f6dfac", alpha=0.65, label="withheld interval")
    for key, label, color, marker in [
        ("correct_yukawa_bie", "exact BIE", "#555555", "o"),
        ("generic_matched", "generic", "#c95f45", "s"),
        ("radial_separate", "radial $(\\kappa,r)$", "#d7a13b", "^"),
        ("radial_normal", "normal form", "#287a78", "D"),
    ]:
        vals = [interval["per_kappa"][str(k)][key] for k in kappas]
        axes[1].plot(kappas, vals, label=label, color=color, marker=marker, linewidth=1.7, markersize=4)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0.0, 4.0)
    axes[1].set_xlabel("$\\kappa$")
    axes[1].set_title("B  Withheld-interval interpolation", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)

    sample_counts = np.array([100, 300, 800, 2400])
    for key, label, color, marker in [
        ("radial_normal", "normal form", "#287a78", "D"),
        ("generic_matched", "generic", "#c95f45", "s"),
    ]:
        values = [
            efficiency[(f"sample_n{count}", key)]
            for count in sample_counts
        ]
        axes[2].plot(
            sample_counts, values, label=label, color=color, marker=marker,
            linewidth=1.7, markersize=4,
        )
    axes[2].set_yscale("log")
    axes[2].set_xscale("log")
    axes[2].set_xticks(sample_counts, labels=[str(value) for value in sample_counts])
    axes[2].set_xlabel("unique training instances")
    axes[2].set_ylabel("mean relative $L^2$")
    axes[2].set_title("C  Observation efficiency", loc="left", fontweight="bold")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    inset = axes[2].inset_axes([0.56, 0.08, 0.4, 0.42])
    noise_levels = np.array([0.0, 0.01, 0.05])
    for key, color, marker in [
        ("radial_normal", "#287a78", "D"),
        ("generic_matched", "#c95f45", "s"),
    ]:
        values = [
            efficiency[(
                f"sensor_q32_noise{str(noise).replace('.', 'p')}", key,
            )]
            for noise in noise_levels
        ]
        inset.plot(noise_levels * 100, values, color=color, marker=marker,
                   linewidth=1.2, markersize=3)
    inset.set_yscale("log")
    inset.set_xlabel("noise (%)", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)

    for key, label, color, marker in [
        ("radial", "normal form", "#287a78", "D"),
        ("full_prior", "generic prior", "#c95f45", "s"),
        ("no_prior", "generic no prior", "#d7a13b", "^"),
    ]:
        rows = scale[key]
        axes[3].plot(
            [r["scale"] for r in rows],
            [max(r["mean_equivariance_drift"], 1e-12) for r in rows],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=4,
        )
    axes[3].set_yscale("log")
    axes[3].set_xlabel("geometry scale $a$")
    axes[3].set_ylabel("equivariance drift")
    axes[3].set_title("D  Scale covariance", loc="left", fontweight="bold")
    axes[3].grid(alpha=0.25)

    output = ROOT / "figures" / "screening_law_paper_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.3, 2.8), constrained_layout=True)
    inv = inverse["summary"]
    inv_methods = ["exact_yukawa", "generic_full_prior", "generic_no_prior", "radial_normal_form"]
    inv_labels = ["Exact\nBIE", "Generic\nprior", "Generic\nno prior", "Normal\nform"]
    inv_values = [inv[m]["mean_coefficient_rmse"] for m in inv_methods]
    axis.bar(inv_labels, inv_values, color=colors)
    axis.set_yscale("log")
    axis.set_ylabel("coefficient RMSE")
    axis.set_title("Differentiable inverse recovery", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    inverse_output = ROOT / "figures" / "screening_law_inverse.png"
    fig.savefig(inverse_output, dpi=220)
    plt.close(fig)
    print(output)
    print(inverse_output)


if __name__ == "__main__":
    main()
