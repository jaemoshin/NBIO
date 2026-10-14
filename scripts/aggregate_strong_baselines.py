#!/usr/bin/env python
"""Aggregate geometry-aware baseline runs into paper-ready artifacts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    runs = []
    for p in sorted(RESULTS.glob("strong_*_s*/eval.json")):
        d = _load(p)
        d["path"] = str(p.relative_to(ROOT))
        runs.append(d)
    if not runs:
        raise SystemExit("No strong baseline eval files found under results/strong_*_s*/eval.json")

    summary = []
    for model in sorted({r["model"] for r in runs}):
        rs = [r for r in runs if r["model"] == model]
        vals = np.array([r["test"]["interior_test"] for r in rs], dtype=float)
        summary.append({
            "model": model,
            "n_runs": len(rs),
            "mean_rel_l2": float(vals.mean()),
            "std_rel_l2": float(vals.std(ddof=0)),
            "best_rel_l2": float(vals.min()),
            "median_rel_l2": float(np.median(vals)),
            "params": int(rs[0]["n_params"]),
            "provenance": rs[0]["provenance"],
            "best_run": rs[int(vals.argmin())]["path"],
            "testres64_mean": float(np.nanmean([r["test"].get("testres64", math.nan) for r in rs])),
            "testres128_mean": float(np.nanmean([r["test"].get("testres128", math.nan) for r in rs])),
            "testres256_mean": float(np.nanmean([r["test"].get("testres256", math.nan) for r in rs])),
        })

    references = {
        "NBIO": {"rel_l2": 4.80e-7, "params": 9026, "provenance": "current paper headline"},
        "BoundaryTransformer": {"rel_l2": 2.38e-2, "params": 878000, "provenance": "current paper headline"},
        "DeepONet": {"rel_l2": 8.11e-2, "params": 625409, "provenance": "current paper headline"},
        "DeepSet": {"rel_l2": 1.66e-1, "params": 116737, "provenance": "current paper headline"},
    }
    out = {
        "task": "geometry_aware_strong_baselines_same_dataset",
        "runs": runs,
        "summary": summary,
        "references": references,
        "attempted_not_scored": [
            {
                "model": "gino",
                "provenance": "neuraloperator-library v2.0.0",
                "reason": (
                    "Canonical GINO instantiated and began training, but the library path is very slow for "
                    "per-sample changing boundary geometries on this boundary-to-query task. Full and capped "
                    "runs were cancelled before full-test evaluation; no result is reported as a scored row."
                ),
                "logs": [
                    "slurm_outputs/nbio-gino-s0-1259714.out",
                    "slurm_outputs/nbio-gino-fast-s0-1259724.out",
                    "slurm_outputs/nbio-gino-cap-s0-1259736.out",
                ],
            }
        ],
        "honest_status": "Same interior-Laplace boundary-to-query dataset; GNO/GINO use neuraloperator, GeoFNO is pure-torch fallback.",
    }
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "strong_baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    make_figure(out)
    make_report(out)
    print("[aggregate_strong_baselines] wrote results/strong_baselines.json, figures/strong_baselines.png, reports/strong_baselines.md")


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    labels = ["NBIO"] + [s["model"].upper() for s in out["summary"]] + ["BoundaryTransformer", "DeepONet", "DeepSet"]
    vals = [out["references"]["NBIO"]["rel_l2"]]
    vals += [s["mean_rel_l2"] for s in out["summary"]]
    vals += [out["references"][k]["rel_l2"] for k in ("BoundaryTransformer", "DeepONet", "DeepSet")]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(np.arange(len(labels)), vals, color=["C2"] + ["C0"] * len(out["summary"]) + ["C1", "C1", "C1"])
    ax.set_yscale("log")
    ax.set_ylabel("rel-L2 on interior_test")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Geometry-aware baselines on the same boundary-to-query task")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "strong_baselines.png", dpi=180, bbox_inches="tight")


def make_report(out):
    REPORTS.mkdir(exist_ok=True)
    lines = [
        "# Strong geometry-aware baselines",
        "",
        "Same interior-Laplace boundary-to-query dataset as NBIO. GNO/GINO use the canonical `neuraloperator` library; GeoFNO is a faithful pure-torch fallback.",
        "",
        "| method | runs | params | rel-L2 mean | rel-L2 std | best | provenance |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| NBIO | - | {out['references']['NBIO']['params']} | {out['references']['NBIO']['rel_l2']:.3e} | - | {out['references']['NBIO']['rel_l2']:.3e} | {out['references']['NBIO']['provenance']} |",
    ]
    for s in out["summary"]:
        lines.append(
            f"| {s['model']} | {s['n_runs']} | {s['params']} | {s['mean_rel_l2']:.3e} | "
            f"{s['std_rel_l2']:.1e} | {s['best_rel_l2']:.3e} | {s['provenance']} |"
        )
    for k in ("BoundaryTransformer", "DeepONet", "DeepSet"):
        r = out["references"][k]
        lines.append(f"| {k} | - | {r['params']} | {r['rel_l2']:.3e} | - | {r['rel_l2']:.3e} | {r['provenance']} |")
    lines += [
        "",
        "## Attempted but not scored",
        "",
        "- `GINO` from `neuraloperator v2.0.0` instantiated and began training, but the canonical path was too slow for per-sample changing boundary geometries here. Full/capped attempts were cancelled before full-test evaluation, so no GINO number is reported.",
        "",
        "Interpretation: these are stronger geometry-aware baselines than the original DeepONet/DeepSet rows. The claim should be updated according to the measured rows above, without hiding any case where a baseline improves materially.",
    ]
    (REPORTS / "strong_baselines.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
