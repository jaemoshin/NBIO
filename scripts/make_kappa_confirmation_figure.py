#!/usr/bin/env python
"""Paper-style summary figure for the frozen-checkpoint kappa confirmation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/kappa_confirmation"
OUT = ROOT / "figures/kappa_confirmation.png"


def main():
    rows = list(csv.DictReader((RESULT / "summary.csv").open()))
    audit = json.loads((RESULT / "audit.json").read_text())
    kappas = [0.0, 0.0625, 0.2, 0.6, 1.1, 1.6, 2.2, 3.0, 3.8, 4.25, 5.25]
    methods = (
        ("correct_yukawa_bie", "Correct-kernel BIE", "black", "--"),
        ("fixed_laplace_bie", "Fixed Laplace BIE", "#999999", ":"),
        ("none_norm", "Prior-free learned BIE", "#7570b3", "-"),
        ("full_norm", "Laplace prior + correction", "#d95f02", "-"),
        ("router", "Validation-routed", "#1b9e77", "-"),
    )

    def values(method, margin, n_b):
        lookup = {
            float(row["kappa"]): float(row["mean_rel_l2"])
            for row in rows
            if (row["method"] == method
                and abs(float(row["query_margin"]) - margin) < 1e-12
                and int(row["n_b"]) == n_b)
        }
        return [lookup[kappa] for kappa in kappas]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, margin, n_b, title in (
        (axes[0], 0.12, 128, "Bulk queries"),
        (axes[1], 0.02, 256, "Near-boundary queries"),
    ):
        for method, label, color, linestyle in methods:
            ax.plot(kappas, values(method, margin, n_b), marker="o", markersize=3.3,
                    linewidth=1.6, linestyle=linestyle, color=color, label=label)
        ax.axvspan(4.0, 5.25, color="#eeeeee", zorder=-2)
        ax.axvline(4.0, color="#aaaaaa", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(r"modified-Helmholtz parameter $\kappa$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.18)
    axes[0].set_ylabel("relative L2 error")

    comparisons = audit["confirmatory_comparisons"]
    bars = [
        100 * comparisons["interpolation_bulk_full_vs_none"]["relative_improvement"],
        100 * comparisons["interpolation_near_full_vs_none"]["relative_improvement"],
        100 * comparisons["interpolation_bulk_router_vs_none"]["relative_improvement"],
        100 * comparisons["interpolation_near_router_vs_none"]["relative_improvement"],
    ]
    labels = ["Prior\nbulk", "Prior\nnear", "Router\nbulk", "Router\nnear"]
    colors = ["#d95f02", "#d95f02", "#1b9e77", "#1b9e77"]
    axes[2].bar(np.arange(4), bars, color=colors, width=0.7)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(np.arange(4), labels)
    axes[2].set_ylabel("error reduction vs prior-free (%)")
    axes[2].set_title("Frozen confirmation")
    axes[2].grid(True, axis="y", alpha=0.18)
    for index, value in enumerate(bars):
        axes[2].text(index, value + 1.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
