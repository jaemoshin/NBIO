#!/usr/bin/env python
"""QBX close-evaluation figure from the validated runs (validate_qbx.py + qbx_transmission_diag.py)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# measured (committed run logs)
d = [0.1, 0.03, 0.01, 0.003, 0.001]
diskA = {"direct": [1.16e-15, 3.69e-11, 1.19e-5, 1.07e-3, 4.02e-3],
         "QBX": [3.25e-15, 4.63e-15, 4.97e-15, 5.09e-15, 5.16e-15]}
star = {"d": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
        "direct": [5.15e-15, 1.02e-11, 1.10e-4, 1.04e-2, 6.79e-2, 1.55e-1],
        "QBX": [6.58e-15, 6.18e-15, 2.21e-14, 3.64e-13, 8.68e-13, 1.10e-12]}
gaps = [0.10, 0.05, 0.02]
B = {"BEM128+direct": [6.25e-5, 8.07e-4, 5.68e-3],
     "BEM128+QBX": [3.66e-11, 2.52e-7, 2.45e-4],
     "BEM512+QBX": [4.61e-15, 4.39e-15, 3.36e-10]}


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].loglog(star["d"], star["direct"], "o-", c="#c0392b", label="direct quadrature")
    ax[0].loglog(star["d"], star["QBX"], "o-", c="#2980b9", label="QBX")
    ax[0].set_xlabel("distance to boundary d"); ax[0].set_ylabel("rel-L2 vs exact (log)")
    ax[0].set_title("Single-layer near-boundary eval (star)\nQBX ~1e-12 as d->0; direct blows up"); ax[0].legend(fontsize=8); ax[0].invert_xaxis()
    ax[1].loglog(d, diskA["direct"], "o-", c="#c0392b", label="direct quadrature")
    ax[1].loglog(d, diskA["QBX"], "o-", c="#2980b9", label="QBX")
    ax[1].set_xlabel("distance to interface d"); ax[1].set_ylabel("rel-L2 vs ANALYTIC (log)")
    ax[1].set_title("Transmission near-interface (concentric disk)\nQBX machine-precision to the interface"); ax[1].legend(fontsize=8); ax[1].invert_xaxis()
    x = np.arange(len(gaps)); w = 0.27
    cols = {"BEM128+direct": "#c0392b", "BEM128+QBX": "#e67e22", "BEM512+QBX": "#2980b9"}
    for i, k in enumerate(["BEM128+direct", "BEM128+QBX", "BEM512+QBX"]):
        ax[2].bar(x + (i - 1) * w, B[k], w, color=cols[k], label=k)
    ax[2].set_yscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels([f"gap {g}" for g in gaps])
    ax[2].set_ylabel("gap-region rel-L2 (log)")
    ax[2].set_title("Near-touching decomposition\nQBX fixes eval; thin gap also needs finer assembly"); ax[2].legend(fontsize=8)
    plt.tight_layout(); os.makedirs("figures", exist_ok=True); plt.savefig("figures/qbx_close_eval.png", dpi=130)
    print("wrote figures/qbx_close_eval.png")


if __name__ == "__main__":
    main()
