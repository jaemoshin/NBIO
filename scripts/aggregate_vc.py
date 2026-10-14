#!/usr/bin/env python
"""Aggregate Stage-2 (variable-coefficient) results -> results/vc_summary.{json,md} + figures/vc_summary.png.
green_fno (const-coeff prior + learned correction) vs fno (black-box) vs const-coeff prior (wrong fixed
kernel). The key Stage-2 question: does the learned correction beat the fixed prior, and does the prior help
the FNO?"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [("vc_test", "in-dist"), ("vc_test_hicon", "zs contrast 6"), ("vc_testres64", "res 64"), ("vc_testres128", "res 128")]


def L(r):
    p = f"results/{r}/eval.json"
    return json.load(open(p)) if os.path.exists(p) else None


def collect(runs):
    vals = []
    prior = []
    params = None
    for r in runs:
        ev = L(r)
        if not ev:
            continue
        params = ev.get("n_params", params)
        vals.append([ev["eval"].get(k, {}).get("rel_l2", float("nan")) for k, _ in KS])
        prior.append([ev["eval"].get(k, {}).get("prior_rel_l2", float("nan")) for k, _ in KS])
    arr = np.array(vals, dtype=float) if vals else np.empty((0, len(KS)))
    parr = np.array(prior, dtype=float) if prior else np.empty((0, len(KS)))
    if arr.size == 0:
        return None, None, params
    return (np.nanmean(arr, 0), np.nanstd(arr, 0), arr.shape[0], params), (
        np.nanmean(parr, 0), np.nanstd(parr, 0), parr.shape[0], 0), params


def fmt(mu, sd, n):
    if n > 1 and sd > 0:
        return f"{mu:.2e}±{sd:.1e}"
    return f"{mu:.2e}"


def main():
    groups = {"green_fno (prior+correction)": [f"vc_green_fno_s{s}" for s in range(5)],
              "fno (black-box)": [f"vc_fno_s{s}" for s in range(5)]}
    rows = {}; prior = None
    for name, runs in groups.items():
        ms, pr, _ = collect(runs)
        if ms:
            rows[name] = ms
            if prior is None:
                prior = pr
    if prior:
        rows["const-coeff prior (wrong fixed kernel)"] = prior
    js = {}
    for name, row in rows.items():
        mu, sd, n, params = row
        js[name] = {"n_seeds": n, "n_params": params,
                    **{k: {"mean": float(mu[i]), "std": float(sd[i])} for i, (k, _) in enumerate(KS)}}
    json.dump(js, open("results/vc_summary.json", "w"), indent=2)
    lines = ["# Stage 2 (variable principal coefficient div(a grad u)=f) summary\n",
             "rel-L2 vs validated variable-coefficient FD GT. The const-coeff prior is the 'wrong fixed",
             "kernel'; green_fno = prior + learned correction; fno = black-box.\n",
             "| method | params | " + " | ".join(lab for _, lab in KS) + " |", "|---|---|" + "---|" * len(KS)]
    for name, row in rows.items():
        mu, sd, n, params = row
        cells = " | ".join(fmt(mu[i], sd[i], n) for i, _ in enumerate(KS))
        lines.append(f"| {name} (n={n}) | {params} | {cells} |")
    lines += ["\n**Stage-2 test:** if green_fno << const-coeff prior, the learned correction does real work",
              "the fixed kernel cannot (the regime no closed-form Green's function exists). Compare green_fno",
              "vs fno to see whether the structure-informed prior helps the black-box."]
    open("results/vc_summary.md", "w").write("\n".join(lines) + "\n")
    if rows:
        x = np.arange(len(KS)); names = list(rows.keys()); w = 0.8 / len(names)
        cols = {"green_fno (prior+correction)": "#27ae60", "fno (black-box)": "#e67e22",
                "const-coeff prior (wrong fixed kernel)": "#7f8c8d"}
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for i, nm in enumerate(names):
            ax.bar(x + (i - len(names) / 2) * w + w / 2, rows[nm][0], w,
                   label=nm, color=cols.get(nm, None))
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in KS], fontsize=9)
        ax.set_ylabel("rel-L2 (log)"); ax.set_title("Stage 2: variable-coefficient elliptic"); ax.legend(fontsize=8)
        plt.tight_layout(); os.makedirs("figures", exist_ok=True); plt.savefig("figures/vc_summary.png", dpi=130)
    print(open("results/vc_summary.md").read()); print("[aggregate_vc] DONE")


if __name__ == "__main__":
    main()
