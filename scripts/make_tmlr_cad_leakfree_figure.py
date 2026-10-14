#!/usr/bin/env python
"""Paper figure for the leakage-free three-seed CAD comparison (plots saved summary JSON only)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nbio_fig_style as S


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "tmlr_revision" / "cad_leakfree_summary.json"
OUT = ROOT / "figures" / "tmlr_cad_leakfree.png"


def main() -> None:
    rows = {row["dataset"]: row for row in json.loads(DATA.read_text())["rows"]}
    labels = ["ABC", "Fusion 360"]
    datasets = ["abc", "fusion"]
    structure = np.array([rows[d]["structure_median_rel_l2"] for d in datasets])
    learned = np.array([rows[d]["seed_median_mean"] for d in datasets])
    learned_std = np.array([rows[d]["seed_median_sample_std"] for d in datasets])
    gaps = learned / structure

    S.apply()
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    x = np.arange(len(labels)); width = 0.34
    ax.bar(x - width / 2, structure, width, color=S.ENTITY["nbio"], zorder=3,
           label="NBIO / exact-BIE structure (0 params)")
    ax.bar(x + width / 2, learned, width, yerr=learned_std, capsize=4, zorder=3,
           color=S.ENTITY["black_box"], error_kw=dict(ecolor=S.INK2, lw=1.1),
           label="strongest learned baseline (3 seeds)")
    for i, gap in enumerate(gaps):
        ax.text(x[i] + width / 2, learned[i] * 1.35, f"{gap:.0f}×", ha="center",
                va="bottom", weight="bold", color=S.INK, fontsize=12)
        ax.text(x[i] - width / 2, structure[i] * 1.35, f"{structure[i]:.1e}", ha="center",
                va="bottom", color=S.ENTITY["nbio"], fontsize=8.5)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.6)
    ax.set_xticks(x, labels); ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("median relative $L_2$")
    ax.set_title("Leakage-free same-protocol CAD exterior electrostatics")
    ax.grid(axis="y", which="major", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.style_axes(ax)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
