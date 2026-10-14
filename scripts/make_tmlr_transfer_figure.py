#!/usr/bin/env python
"""Fixed-kappa=4 learned-correction transfer figure. Plots the SAVED audit summary CSV only
(results/tmlr_revision/learned_transfer_summary.csv) -- no recomputation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nbio_fig_style as S


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "tmlr_revision" / "learned_transfer_summary.csv"
OUT = ROOT / "figures" / "tmlr_learned_transfer.png"

METHODS = [  # (csv key, entity key, legend label)
    ("correct_yukawa_bie", "reference",    "correct Yukawa BIE (reference)"),
    ("fixed_laplace_bie",  "wrong_prior",  "fixed Laplace BIE (wrong prior)"),
    ("learned_split",      "nbio_split",   "Laplace prior + learned correction (NBIO)"),
    ("learned_nosplit",    "nbio_nosplit", "learned BIE kernel (no split)"),
]
FAMILIES = [("in_dist", "in-distribution"), ("highmode", "high-mode"),
            ("rough", "rough"), ("deform", "deformed")]
MARGINS = [("0.12", "far-field query  (margin 0.12)"), ("0.02", "near-boundary query  (margin 0.02)")]


def load():
    d = {}
    for r in csv.DictReader(DATA.open()):
        key = (r["method"], r["family"], r["query_margin"], int(float(r["n_b"])))
        d[key] = r
    return d


def main() -> None:
    D = load()
    nbs = [64, 128, 256]
    S.apply()
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), sharex=True, sharey="row",
                             constrained_layout=True)
    for ri, (mkey, mlabel) in enumerate(MARGINS):
        for ci, (fkey, ftitle) in enumerate(FAMILIES):
            ax = axes[ri, ci]
            for csvk, ent, _ in METHODS:
                y = np.array([float(D[(csvk, fkey, mkey, nb)]["mean_rel_l2"]) for nb in nbs])
                lo = np.array([float(D[(csvk, fkey, mkey, nb)]["bootstrap_mean_ci_low"]) for nb in nbs])
                hi = np.array([float(D[(csvk, fkey, mkey, nb)]["bootstrap_mean_ci_high"]) for nb in nbs])
                col = S.ENTITY[ent]
                ax.fill_between(nbs, lo, hi, color=col, alpha=0.15, linewidth=0, zorder=2)
                ax.plot(nbs, y, color=col, marker=S.MARKER[ent], linestyle=S.DASH[ent],
                        markersize=6, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            ax.set_yscale("log"); ax.set_xticks(nbs)
            ax.grid(axis="both", which="major", zorder=0)
            S.style_axes(ax)
            if ri == 0:
                ax.set_title(ftitle)
            if ci == 0:
                ax.set_ylabel("mean relative $L_2$")
                ax.text(-0.42, 0.5, mlabel, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", color=S.INK2, fontsize=10.5, weight="semibold")
            if ri == 1:
                ax.set_xlabel("boundary nodes $n_b$")
    handles = [plt.Line2D([0], [0], color=S.ENTITY[e], marker=S.MARKER[e], linestyle=S.DASH[e],
                          markeredgecolor="white", markeredgewidth=0.6, label=lab)
               for _, e, lab in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=4, columnspacing=1.6)
    fig.suptitle("Fixed-$\\kappa{=}4$ learned-correction transfer  (trained at $n_b{=}128$)",
                 fontsize=13, weight="semibold", color=S.INK)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
