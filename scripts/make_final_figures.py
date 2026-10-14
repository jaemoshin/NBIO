#!/usr/bin/env python
"""Adaptive near-touching + 3D NBIO figures (measured from committed runs)."""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gaps = [0.10, 0.05, 0.02, 0.01, 0.005]
A = {"direct": [6.25e-5, 8.07e-4, 5.68e-3, 1.69e-2, 6.47e-1],
     "QBX (no oversample)": [2.00e-6, 2.18e-6, 2.45e-4, 3.33e-3, 2.18e-1],
     "QBX + oversampled assembly": [7.53e-7, 1.92e-6, 5.08e-5, 4.29e-4, 1.63e-3],
     "QBX + adaptive assembly": [7.53e-7, 1.92e-6, 5.08e-5, 4.28e-4, 1.54e-3]}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "tmlr_final_multiseed.json")) as f:
    final = json.load(f)["surface_3d_joint"]

slice_keys = ["interior_indist", "exterior_indist", "interior_strong_deform",
              "exterior_strong_deform", "interior_res720", "exterior_res720"]
labels3d = ["int.", "ext.", "int. deform", "ext. deform", "int. 2x res", "ext. 2x res"]
D3 = {
    "NBIO-3D structure (0 params)": [final["nbio3d_structure"][k]["mean"] for k in slice_keys],
    "NBIO-3D joint (1.4k, 3 seeds)": [final["nbio3d_joint"][k]["mean"] for k in slice_keys],
    "DeepSet-3D joint (117k, 3 seeds)": [final["deepset3d_joint"][k]["mean"] for k in slice_keys],
}
D3_ERR = {
    "NBIO-3D structure (0 params)": [0.0 for _ in slice_keys],
    "NBIO-3D joint (1.4k, 3 seeds)": [final["nbio3d_joint"][k]["sample_std"] for k in slice_keys],
    "DeepSet-3D joint (117k, 3 seeds)": [final["deepset3d_joint"][k]["sample_std"] for k in slice_keys],
}


def main():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    cols = {"direct": "#c0392b", "QBX (no oversample)": "#e67e22",
            "QBX + oversampled assembly": "#2980b9", "QBX + adaptive assembly": "#27ae60"}
    for k, v in A.items():
        ax[0].loglog(gaps, v, "o-", label=k, color=cols[k])
    ax[0].invert_xaxis(); ax[0].set_xlabel("inter-inclusion gap"); ax[0].set_ylabel("gap-region rel-L2 (log)")
    ax[0].set_title("Near-touching close evaluation (in-operator)\nQBX + oversampled/adaptive assembly"); ax[0].legend(fontsize=7)
    x = np.arange(len(labels3d)); w = 0.26
    c3 = {"NBIO-3D structure (0 params)": "#2980b9", "NBIO-3D joint (1.4k, 3 seeds)": "#27ae60",
          "DeepSet-3D joint (117k, 3 seeds)": "#c0392b"}
    for i, (k, v) in enumerate(D3.items()):
        ax[1].bar(x + (i - 1) * w, v, w, yerr=D3_ERR[k], capsize=2, label=k, color=c3[k])
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(labels3d, fontsize=9)
    ax[1].set_ylabel("rel-L2 (log)"); ax[1].set_title("3D NBIO on deformed spheres\njoint interior/exterior correction, 3 seeds")
    ax[1].legend(fontsize=7)
    plt.tight_layout(); os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/adaptive_3d.png", dpi=130, facecolor="white")
    print("wrote figures/adaptive_3d.png")


if __name__ == "__main__":
    main()
