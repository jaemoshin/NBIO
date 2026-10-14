#!/usr/bin/env python
"""Create the consolidated reviewer-revision evidence figure."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-dir", default="results/tmlr_revision2")
    parser.add_argument("--aux-dir", default="results/tmlr_revision2")
    parser.add_argument(
        "--output", default="figures/tmlr_revision2_learning_utility.png"
    )
    args = parser.parse_args()
    transfer_root = ROOT / args.transfer_dir
    auxiliary_root = ROOT / args.aux_dir
    transfer = read_csv(transfer_root / "conditioned_transfer_summary.csv")
    inverse = read_csv(auxiliary_root / "inverse_demo_raw.csv")
    runtime = read_csv(auxiliary_root / "runtime_amortization.csv")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.9), constrained_layout=True)

    # A: in-range bulk transfer at the finest tested resolution.
    ax = axes[0, 0]
    families = ("in_dist", "highmode", "rough", "deform")
    labels = ("in-dist", "high-mode", "rough", "deformed")
    colors = {"full_prior": "#176B87", "no_prior": "#B5523B"}
    x = np.arange(len(families))
    width = 0.18
    for load_index, load in enumerate(("point_source", "fourier")):
        for method_index, method in enumerate(("full_prior", "no_prior")):
            values = []
            for family in families:
                cells = [
                    float(row["mean_rel_l2"])
                    for row in transfer
                    if row["method"] == method
                    and row["geometry_family"] == family
                    and row["load_family"] == load
                    and row["kappa_regime"] == "in_range"
                    and float(row["query_margin"]) == 0.12
                    and int(row["n_b"]) == 256
                ]
                values.append(float(np.mean(cells)))
            offset = (load_index * 2 + method_index - 1.5) * width
            hatch = "" if load == "point_source" else "///"
            ax.bar(
                x + offset, values, width, color=colors[method], hatch=hatch,
                edgecolor="white", linewidth=0.5,
                label=("full prior" if method == "full_prior" else "no prior")
                + (", point" if load == "point_source" else ", Fourier"),
            )
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("mean relative $L_2$")
    ax.set_title("A  Conditioned correction: unseen geometries and loads")
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(ncol=2, frameon=False, loc="upper left")

    # B: near-boundary resolution crossover, aggregated across in-range kappa/families.
    ax = axes[0, 1]
    for load, marker in (("point_source", "o"), ("fourier", "s")):
        for method, linestyle in (("full_prior", "-"), ("no_prior", "--")):
            values = []
            for n_b in (64, 128, 256):
                cells = [
                    float(row["mean_rel_l2"])
                    for row in transfer
                    if row["method"] == method
                    and row["load_family"] == load
                    and row["kappa_regime"] == "in_range"
                    and float(row["query_margin"]) == 0.02
                    and int(row["n_b"]) == n_b
                ]
                values.append(float(np.mean(cells)))
            ax.plot(
                (64, 128, 256), values, color=colors[method], linestyle=linestyle,
                marker=marker, linewidth=1.8, markersize=5,
                label=("full" if method == "full_prior" else "none")
                + (", point" if load == "point_source" else ", Fourier"),
            )
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks((64, 128, 256), ("64", "128", "256"))
    ax.set_xlabel("boundary nodes")
    ax.set_ylabel("near-boundary relative $L_2$")
    ax.set_title("B  Analytic singularity helps after refinement")
    ax.grid(which="both", alpha=0.2)
    ax.legend(frameon=False, ncol=2)

    # C: inverse recovery trajectories.
    ax = axes[1, 0]
    inverse_groups = defaultdict(list)
    for row in inverse:
        inverse_groups[(row["method"], int(row["step"]))].append(
            float(row["coefficient_rmse"])
        )
    inverse_style = {
        "correct_yukawa_bie": ("correct BIE", "#2F855A", "-"),
        "conditioned_full_prior": ("full prior", "#176B87", "-"),
        "conditioned_no_prior": ("no prior", "#B5523B", "--"),
    }
    for method, (label, color, linestyle) in inverse_style.items():
        steps = sorted(step for candidate, step in inverse_groups if candidate == method)
        median = [np.median(inverse_groups[(method, step)]) for step in steps]
        ax.plot(steps, median, label=label, color=color, linestyle=linestyle, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("coefficient RMSE")
    ax.set_title("C  Shape recovery through the differentiable solve")
    ax.grid(which="both", alpha=0.2)
    ax.legend(frameon=False)

    # D: explicit computational contracts.
    ax = axes[1, 1]
    chosen = [
        row for row in runtime
        if int(row["n_b"]) == 128 and int(row["n_query"]) == 256
        and int(row["n_rhs"]) in (1, 32)
    ]
    methods = ("structure_only_laplace", "conditioned_full_prior", "conditioned_no_prior")
    method_labels = ("structure", "full prior", "no prior")
    x = np.arange(len(methods))
    one_rhs = []
    cached = []
    multi_rhs = []
    for method in methods:
        row_one = next(row for row in chosen if row["method"] == method and int(row["n_rhs"]) == 1)
        row_many = next(row for row in chosen if row["method"] == method and int(row["n_rhs"]) == 32)
        one_rhs.append(float(row_one["new_geometry_total_ms_batch"]) / 8.0)
        cached.append(float(row_one["cached_geometry_cached_query_ms_per_rhs_sample"]))
        multi_rhs.append(float(row_many["cached_geometry_cached_query_ms_per_rhs_sample"]))
    ax.bar(x - 0.24, one_rhs, 0.24, label="new geometry", color="#5D6D7E")
    ax.bar(x, cached, 0.24, label="cached, 1 RHS", color="#D49A3A")
    ax.bar(x + 0.24, multi_rhs, 0.24, label="cached, 32 RHS", color="#2F855A")
    ax.set_yscale("log")
    ax.set_xticks(x, method_labels, rotation=10)
    ax.set_ylabel("ms per sample / RHS")
    ax.set_title("D  New-geometry and cached-geometry contracts")
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(frameon=False)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
