#!/usr/bin/env python
"""Build a compact figure for the next-step NBIO diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    h = pd.DataFrame(json.load(open(RESULTS / "next_steps_hmatrix_gmres.json"))["rows"])
    q = pd.DataFrame(json.load(open(RESULTS / "next_steps_qbx_gap_double.json"))["rows"])
    helm = pd.DataFrame(json.load(open(RESULTS / "next_steps_helmholtz_cfie.json"))["rows"])
    n3 = json.load(open(RESULTS / "next_steps_3d_joint.json"))["eval"]
    ds3 = json.load(open(RESULTS / "next_steps_3d_joint_deepset.json"))["eval"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))

    ax[0, 0].loglog(h["n_b"], h["compressed_matvec_ms"], "o-", label="compressed matvec")
    ax[0, 0].set_xlabel("boundary nodes")
    ax[0, 0].set_ylabel("ms / matvec")
    ax[0, 0].set_title("H-matrix + factored correction")
    ax[0, 0].grid(alpha=0.25, which="both")
    ax2 = ax[0, 0].twinx()
    ax2.plot(h["n_b"], h["gmres_iters"], "s--", color="C1", label="GMRES iters")
    ax2.set_ylabel("GMRES iterations")
    ax2.set_ylim(0, max(h["gmres_iters"]) + 3)

    best_q = q.groupby(["gap", "n_boundary"])["rel_l2_gap_region"].min().reset_index()
    for nb, g in best_q.groupby("n_boundary"):
        ax[0, 1].loglog(g["gap"], g["rel_l2_gap_region"], "o-", label=f"n={nb}")
    ax[0, 1].invert_xaxis()
    ax[0, 1].set_xlabel("gap")
    ax[0, 1].set_ylabel("best gap-region rel-L2")
    ax[0, 1].set_title("Reduced-radius QBX close-gap stress")
    ax[0, 1].grid(alpha=0.25, which="both")
    ax[0, 1].legend(fontsize=8)

    # Best GMRES per wavelength among swept variants.
    best_h = helm.groupby("wavelengths_across_diameter")["gmres_iters"].min().reset_index()
    rank_h = helm.groupby("wavelengths_across_diameter")["offdiag_rank"].min().reset_index()
    ax[1, 0].plot(best_h["wavelengths_across_diameter"], best_h["gmres_iters"], "o-", label="best swept GMRES")
    ax[1, 0].set_xlabel("wavelengths across diameter")
    ax[1, 0].set_ylabel("GMRES iterations")
    ax[1, 0].set_title("Helmholtz high-k limit remains")
    ax[1, 0].grid(alpha=0.25)
    ax3 = ax[1, 0].twinx()
    ax3.plot(rank_h["wavelengths_across_diameter"], rank_h["offdiag_rank"], "s--", color="C3", label="rank")
    ax3.set_ylabel("off-diagonal rank")

    labels = ["int", "ext", "int deform", "ext deform", "int 720", "ext 720"]
    keys = ["interior_indist", "exterior_indist", "interior_strong_deform", "exterior_strong_deform",
            "interior_res720", "exterior_res720"]
    x = np.arange(len(keys))
    nbio_vals = [n3[k]["mean_rel_l2"] for k in keys]
    ds_vals = [ds3[k]["mean_rel_l2"] for k in keys]
    ax[1, 1].bar(x - 0.18, nbio_vals, 0.36, label="3D NBIO joint")
    ax[1, 1].bar(x + 0.18, ds_vals, 0.36, label="DeepSet")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(labels, rotation=20, ha="right")
    ax[1, 1].set_ylabel("rel-L2")
    ax[1, 1].set_title("3D interior+exterior joint training")
    ax[1, 1].grid(alpha=0.25, axis="y", which="both")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES / "next_steps_acceleration_qbx_3d_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"[write] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
