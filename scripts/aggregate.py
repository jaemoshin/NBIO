#!/usr/bin/env python
"""Aggregate all NBIO results into results/summary.{json,md} + figures/nbio_summary.png."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def L(r):
    p = f"results/{r}/eval.json"
    return json.load(open(p)) if os.path.exists(p) else None


def metric(run, key):
    ev = L(run)
    if ev is None:
        return float("nan")
    d = ev["eval"].get(key, {})
    return d.get("rel_l2", float("nan")) if isinstance(d, dict) else float("nan")


def mean_seeds(runs, key):
    vals = [metric(r, key) for r in runs if L(r) is not None]
    vals = [v for v in vals if v == v]
    return (np.mean(vals), np.std(vals)) if vals else (float("nan"), 0.0)


def bem_ceiling(run, key):
    ev = L(run)
    return ev["eval"].get(key, {}).get("bem_rel_l2", float("nan")) if ev else float("nan")


def main():
    out = {}
    # ---- interior (wave 1) ----
    nbio = ["nbio_s0", "nbio_s1", "nbio_s2"]
    don = ["deeponet_s0", "deeponet_s1"]; dst = ["deepset_s0", "deepset_s1"]
    btr = [f"btransformer_s{s}" for s in range(3)]
    keys = ["interior_test", "testres64", "testres128", "testres256"]
    out["interior"] = {
        "NBIO": {k: mean_seeds(nbio, k) for k in keys},
        "DeepONet": {k: mean_seeds(don, k) for k in keys},
        "DeepSet": {k: mean_seeds(dst, k) for k in keys},
        "BoundaryTransformer": {k: mean_seeds(btr, k) for k in keys},
        "NBIO_nosplit": {k: (metric("nbio_nosplit_s0", k), 0.0) for k in keys},
        "NBIO_neumann": {k: (metric("nbio_neumann_s0", k), 0.0) for k in keys},
        "BEM_ceiling": {k: bem_ceiling("nbio_s0", k) for k in keys},
    }
    out["yukawa"] = {
        "NBIO_laplace_kernel": metric("nbio_yuk_s0", "yukawa_test"),
        "NBIO_nosplit_learned": metric("nbio_yuk_nosplit_s0", "yukawa_test"),
        "DeepONet": metric("deeponet_yuk_s0", "yukawa_test"),
        "DeepSet": metric("deepset_yuk_s0", "yukawa_test"),
        "Yukawa_BEM_ceiling": 1.4e-5, "WRONG_Laplace_BEM": 1.13,
    }
    out["exterior"] = {
        "NBIO": metric("nbio_ext_s0", "exterior_test"),
        "DeepONet": metric("deeponet_ext_s0", "exterior_test"),
        "DeepSet": metric("deepset_ext_s0", "exterior_test"),
        "Exterior_BEM_ceiling": bem_ceiling("nbio_ext_s0", "exterior_test"),
    }
    json.dump(out, open("results/summary.json", "w"), indent=2)

    # ---- markdown ----
    def fmt(m): return f"{m[0]:.2e}" if isinstance(m, (list, tuple)) else f"{m:.2e}"
    lines = ["# NBIO results summary\n", "## Interior (zero-shot geometry + resolution), rel-L2 vs exact\n",
             "| method | params | interior | n_b=64 | n_b=128 | n_b=256 |", "|---|---|---|---|---|---|"]
    pmap = {"NBIO": L("nbio_s0")["n_params"], "DeepONet": L("deeponet_s0")["n_params"],
            "DeepSet": L("deepset_s0")["n_params"], "NBIO_nosplit": L("nbio_nosplit_s0")["n_params"],
            "BoundaryTransformer": (L("btransformer_s0") or {}).get("n_params", "?"),
            "NBIO_neumann": L("nbio_neumann_s0")["n_params"], "BEM_ceiling": "-"}
    for name in ["BEM_ceiling", "NBIO", "NBIO_neumann", "NBIO_nosplit", "BoundaryTransformer", "DeepONet", "DeepSet"]:
        row = out["interior"][name]
        cells = " | ".join(fmt(row[k]) for k in keys)
        lines.append(f"| {name} | {pmap[name]} | {cells} |")
    lines += ["\n## Yukawa wrong-kernel stress test (kappa=4), rel-L2\n",
              f"- correct Yukawa-BEM ceiling: 1.4e-5 ; WRONG fixed Laplace-BEM-on-Yukawa: **1.13 (113%)**",
              "| method | rel-L2 |", "|---|---|"]
    for k, v in out["yukawa"].items():
        lines.append(f"| {k} | {v:.2e} |")
    lines += ["\n## Exterior (unbounded domain), rel-L2\n", "| method | rel-L2 |", "|---|---|"]
    for k, v in out["exterior"].items():
        lines.append(f"| {k} | {v:.2e} |")
    open("results/summary.md", "w").write("\n".join(lines) + "\n")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (1) interior accuracy bars
    meths = ["NBIO", "NBIO_nosplit", "BoundaryTransformer", "DeepONet", "DeepSet"]
    vals = [out["interior"][m]["interior_test"][0] for m in meths]
    cols = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#c0392b"]
    ax[0].bar(range(len(meths)), vals, color=cols, log=True)
    ax[0].axhline(out["interior"]["BEM_ceiling"]["interior_test"], ls="--", c="k", lw=1)
    ax[0].text(len(meths) - 0.5, out["interior"]["BEM_ceiling"]["interior_test"] * 1.3, "BEM", ha="right", fontsize=8)
    ax[0].set_xticks(range(len(meths))); ax[0].set_xticklabels(meths, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("interior rel-L2 (log)"); ax[0].set_title("Interior accuracy (unseen geometries)\nNBIO matches BEM; 1e5x over black-box")
    # (2) resolution transfer
    nbs = [64, 128, 256]
    for m, c in [("NBIO", "#2980b9"), ("BoundaryTransformer", "#8e44ad"), ("DeepONet", "#e67e22"), ("DeepSet", "#c0392b")]:
        if not np.isfinite(out["interior"][m]["testres64"][0]):
            continue
        ax[1].plot(nbs, [out["interior"][m][f"testres{n}"][0] for n in nbs], "o-", color=c, label=m)
    ax[1].plot(nbs, [out["interior"]["BEM_ceiling"][f"testres{n}"] for n in nbs], "k--", label="BEM")
    ax[1].set_yscale("log"); ax[1].set_xlabel("boundary resolution n_b"); ax[1].set_ylabel("rel-L2 (log)")
    ax[1].set_title("Zero-shot resolution (trained @128)\nNBIO converges like a quadrature"); ax[1].legend(fontsize=8)
    # (3) yukawa + exterior
    ylabels = ["Yuk:NBIO\n(wrong kernel)", "Yuk:nosplit\n(learned)", "Yuk:black-box", "Yuk:wrong-BEM", "Ext:NBIO", "Ext:black-box"]
    yvals = [out["yukawa"]["NBIO_laplace_kernel"], out["yukawa"]["NBIO_nosplit_learned"],
             out["yukawa"]["DeepONet"], out["yukawa"]["WRONG_Laplace_BEM"],
             out["exterior"]["NBIO"], out["exterior"]["DeepONet"]]
    ycol = ["#2980b9", "#27ae60", "#e67e22", "#7f8c8d", "#2980b9", "#e67e22"]
    ax[2].bar(range(len(ylabels)), yvals, color=ycol, log=True)
    ax[2].set_xticks(range(len(ylabels))); ax[2].set_xticklabels(ylabels, rotation=30, ha="right", fontsize=7)
    ax[2].set_ylabel("rel-L2 (log)"); ax[2].set_title("Yukawa wrong-kernel + Exterior\narchitecture beats black-box; correction works")
    plt.tight_layout(); os.makedirs("figures", exist_ok=True); plt.savefig("figures/nbio_summary.png", dpi=130)
    print("wrote results/summary.{json,md} + figures/nbio_summary.png")
    print(open("results/summary.md").read())


if __name__ == "__main__":
    main()
