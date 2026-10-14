#!/usr/bin/env python
"""Create the paper/audit summary figure for the radial 3D tranche."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE3D = os.path.join(ROOT, "experiments", "surface3d")
if SURFACE3D not in sys.path:
    sys.path.insert(0, SURFACE3D)

from radial_surface_law import RadialMixture  # noqa: E402


METHOD_LABELS = {
    "exact_yukawa_split": "Exact radial law",
    "generic_pairwise": "Generic pairwise",
    "generic_kappa_pairwise": "Generic + parameter",
    "radial_product": "Unconstrained radial MLP",
    "radial_mixture": "Constrained radial mixture",
}


def load_mixture(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = RadialMixture()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=os.path.join(ROOT, "results", "radial_surface_law"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            ROOT, "results", "radial_surface_law", "radial_surface_summary.png",
        ),
    )
    args = parser.parse_args()

    fixed = pd.read_csv(
        os.path.join(args.result_dir, "fixed_aggregate", "summary.csv"),
    )
    family = pd.read_csv(
        os.path.join(args.result_dir, "combined_aggregate", "summary.csv"),
    )
    mesh = pd.read_csv(
        os.path.join(args.result_dir, "mesh_aggregate", "mesh_summary.csv"),
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))

    ax = axes[0, 0]
    methods = ["exact_yukawa_split", "generic_pairwise", "radial_product"]
    subset = fixed[
        (fixed["geometry"] == "sphere") & fixed["method"].isin(methods)
    ].set_index("method")
    x = np.arange(len(methods))
    values = [subset.loc[method, "mean_rel_l2"] for method in methods]
    errors = [subset.loc[method, "std_rel_l2"] for method in methods]
    ax.bar(x, values, yerr=errors, capsize=3, color=["#777777", "#cc6677", "#228833"])
    ax.set_yscale("log")
    ax.set_xticks(x, ["exact\nsplit", "generic\npairwise", "radial\nproduct"])
    ax.set_ylabel("relative $L^2$")
    ax.set_title("(a) Fixed $\\kappa=3$: torus $\\to$ sphere")

    ax = axes[0, 1]
    methods = ["exact_yukawa_split", "generic_kappa_pairwise", "radial_mixture"]
    geometries = ["torus", "sphere", "two_sphere"]
    colors = ["#777777", "#cc6677", "#228833"]
    width = 0.24
    for index, method in enumerate(methods):
        values = []
        for geometry in geometries:
            rows = family[
                (family["split"] == "interpolation")
                & (family["geometry"] == geometry)
                & (family["method"] == method)
            ]
            values.append(np.average(rows["mean_rel_l2"], weights=rows["n"]))
        ax.bar(
            np.arange(len(geometries)) + (index - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=colors[index],
        )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(geometries)), ["torus", "sphere", "two-sphere"])
    ax.set_ylabel("relative $L^2$")
    ax.set_title("(b) Held-out screening parameters")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    scaled = torch.linspace(0, 12, 500, dtype=torch.float64)
    ax.plot(
        scaled.numpy(),
        torch.exp(-scaled).numpy(),
        color="black",
        linewidth=2,
        label="withheld $e^{-s}$",
    )
    for seed in (711, 712, 713, 714):
        checkpoint = os.path.join(
            args.result_dir,
            "checkpoints",
            f"family_radial_mixture_seed{seed}.pt",
        )
        model = load_mixture(checkpoint).double()
        with torch.no_grad():
            radius = scaled
            q = model(radius, 1.0)
            multiplier = 1 + scaled * q
        ax.plot(
            scaled.numpy(),
            multiplier.numpy(),
            linewidth=1.2,
            alpha=0.8,
            label=f"seed {seed}",
        )
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$s=\\kappa r$")
    ax.set_ylabel("multiplier $m(s)$")
    ax.set_title("(c) Identified radial law")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.scatter(
        mesh["generic_gap_closed_pct"],
        mesh["mixture_gap_closed_pct"],
        c=np.where(mesh["mixture_seed_wins"] == 4, "#228833", "#cc6677"),
        s=36,
        alpha=0.9,
    )
    low = min(
        -5,
        mesh["generic_gap_closed_pct"].min(),
        mesh["mixture_gap_closed_pct"].min(),
    )
    high = max(
        100,
        mesh["generic_gap_closed_pct"].max(),
        mesh["mixture_gap_closed_pct"].max(),
    )
    ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xlim(low - 3, high + 3)
    ax.set_ylim(low - 3, high + 3)
    ax.set_xlabel("generic mesh gap closure (%)")
    ax.set_ylabel("radial mixture gap closure (%)")
    ax.set_title("(d) Frozen 21-mesh transfer")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
