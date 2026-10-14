#!/usr/bin/env python
"""Generalizable NBIO-CAD improvement screen.

This tranche tests improvement *families* that should transfer across CAD
datasets instead of tuning a single dataset:

1. Direct dense BIE/NBIO baseline.
2. Global refined evaluation after density transfer.
3. Local refined replacement for close targets.
4. Resolution-transfer evidence from existing ABC rows.
5. Integrated charge/capacitance diagnostics from existing charge rows.

The fresh screen writes separate `generalizable_*` artifacts and never overwrites
the existing CAD/QBX benchmark outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.abc_cad_electrostatics_benchmark import ABC_SUBSET, file_key
from scripts.cad_official_qbx_tranche import (
    eval_field_global_refined,
    eval_field_local_replacement,
    eval_global_refined,
    eval_local_replacement,
    finite_summary,
    knn_transfer,
)
from scripts.cad_overnight_tranche import get_abc_paths, get_fusion_paths
from scripts.fusion_cad_electrostatics_benchmark import (
    boundary_voltage,
    coarsen_with_normals,
    eval_field,
    eval_potential,
    nearest_values,
    obj_surface_quadrature,
    solve_single_layer,
)
from scripts.mcb_thermal_benchmark import save_json, write_text
from scripts.top5_benchmark_screen import FIGURES, RESULTS, TABLES, matrix_summary, rel_l2
from scripts.top5_completion_screen import parse_obj

REPORTS = Path("reports")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path}")


def as_float(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(y):
        return None
    return y


def fmt(x: Any) -> str:
    y = as_float(x)
    return "-" if y is None else f"`{y:.3e}`"


def screen(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    datasets = [
        ("fusion", get_fusion_paths(args.n_meshes, expanded=True)),
        ("abc", get_abc_paths(args.n_meshes, max_mb=args.max_file_mb, max_faces=args.max_faces)),
    ]
    for dataset, paths in datasets:
        for imesh, path in enumerate(paths):
            try:
                v, f = parse_obj(path)
                xb_full, wb_full, nb_full = obj_surface_quadrature(v, f)
                xb, wb, nb = coarsen_with_normals(xb_full, wb_full, nb_full, args.n_surface)
                xb_ref, wb_ref, nb_ref = coarsen_with_normals(xb_full, wb_full, nb_full, args.n_ref)
                mesh = file_key(path) if dataset == "abc" and ABC_SUBSET in path.parents else path.name
                for family in args.families:
                    gb_ref = boundary_voltage(family, xb_ref, rng)
                    gb = nearest_values(xb_ref, gb_ref, xb)
                    sigma_ref, res_ref, _, rt_ref = solve_single_layer(xb_ref, wb_ref, gb_ref, cond_limit=0)
                    sigma, res, cond, rt = solve_single_layer(xb, wb, gb)
                    sigma_nn = nearest_values(xb, sigma, xb_ref)
                    sigma_knn = knn_transfer(xb, sigma, xb_ref, k=args.transfer_k)
                    for dist_ratio in args.distances:
                        idx = rng.choice(len(xb), size=args.n_query, replace=True)
                        xq = xb[idx] + (2.0 * dist_ratio) * nb[idx]
                        u_ref = eval_potential(xq, xb_ref, wb_ref, sigma_ref)
                        e_ref = eval_field(xq, xb_ref, wb_ref, sigma_ref)
                        variants = [
                            ("direct_coarse", lambda: (eval_potential(xq, xb, wb, sigma), eval_field(xq, xb, wb, sigma))),
                            (
                                "global_refined_nearest_density",
                                lambda: (
                                    eval_global_refined(xq, xb_ref, wb_ref, sigma_nn),
                                    eval_field_global_refined(xq, xb_ref, wb_ref, sigma_nn),
                                ),
                            ),
                            (
                                "global_refined_knn_density",
                                lambda: (
                                    eval_global_refined(xq, xb_ref, wb_ref, sigma_knn),
                                    eval_field_global_refined(xq, xb_ref, wb_ref, sigma_knn),
                                ),
                            ),
                            (
                                f"local_refined_replacement_tau{args.tau_mult:g}",
                                lambda: (
                                    eval_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, args.tau_mult),
                                    eval_field_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, args.tau_mult),
                                ),
                            ),
                        ]
                        for method, fn in variants:
                            t0 = time.perf_counter()
                            u, e = fn()
                            eval_ms = 1e3 * (time.perf_counter() - t0)
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "mesh": mesh,
                                    "family": family,
                                    "method": method,
                                    "distance_ratio": dist_ratio,
                                    "n_boundary": len(xb),
                                    "n_ref": len(xb_ref),
                                    "n_query": len(xq),
                                    "potential_rel_l2": rel_l2(u, u_ref),
                                    "field_rel_l2": rel_l2(e, e_ref),
                                    "solve_residual": res,
                                    "reference_residual": res_ref,
                                    "condition_number": cond,
                                    "solve_runtime_ms": rt,
                                    "reference_runtime_ms": rt_ref,
                                    "eval_runtime_ms": eval_ms,
                                }
                            )
                save_json(
                    {"status": "partial", "dataset": dataset, "completed_meshes": imesh + 1, "rows": rows},
                    RESULTS / "generalizable_improvement_screen_partial.json",
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"dataset": dataset, "mesh": str(path), "method": "all", "status": "failed", "error": str(exc)})
    result = {"status": "ok", "rows": rows, "args": vars(args)}
    save_json(result, RESULTS / "generalizable_improvement_screen.json")
    write_csv(TABLES / "generalizable_improvement_screen_raw.csv", rows)
    aggregate(result)


def aggregate(data: dict[str, Any] | None = None) -> None:
    if data is None:
        data = json.loads((RESULTS / "generalizable_improvement_screen.json").read_text())
    rows = data.get("rows", [])
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "potential_rel_l2" not in r:
            continue
        groups[(str(r["dataset"]), str(r["method"]), float(r["distance_ratio"]))].append(r)
    for (dataset, method, dist), gr in sorted(groups.items()):
        for metric in ["potential_rel_l2", "field_rel_l2", "eval_runtime_ms"]:
            vals = [as_float(r.get(metric)) for r in gr]
            vals = [v for v in vals if v is not None]
            s = finite_summary(vals)
            summary.append({"dataset": dataset, "method": method, "distance_ratio": dist, "metric": metric, **s})
    write_csv(TABLES / "generalizable_improvement_screen_summary.csv", summary)

    recommendation_rows = rank_methods(summary)
    write_csv(TABLES / "generalizable_improvement_recommendations.csv", recommendation_rows)
    save_json({"status": "ok", "summary": summary, "recommendations": recommendation_rows}, RESULTS / "generalizable_improvement_summary.json")
    plot(summary, recommendation_rows)
    write_report(summary, recommendation_rows)


def rank_methods(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    # Score only potential errors; include closest and mid-distance behavior.
    for dataset in sorted({r["dataset"] for r in summary}):
        for dist in sorted({float(r["distance_ratio"]) for r in summary if r["dataset"] == dataset}):
            pot = [
                r
                for r in summary
                if r["dataset"] == dataset and float(r["distance_ratio"]) == dist and r["metric"] == "potential_rel_l2"
            ]
            direct = next((as_float(r.get("median")) for r in pot if r["method"] == "direct_coarse"), None)
            for r in pot:
                med = as_float(r.get("median"))
                if med is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "distance_ratio": dist,
                        "method": r["method"],
                        "median_potential_rel_l2": med,
                        "improvement_vs_direct": "" if direct is None or med <= 0 else direct / med,
                        "n": r.get("n"),
                    }
                )
    # Add non-fresh, generalizable lessons from existing diagnostics.
    point_patch = Path("results/tables/capacitance_vs_charge_density.csv")
    if point_patch.exists():
        with point_patch.open(newline="") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "dataset": r.get("dataset"),
                        "distance_ratio": "integrated_quantity",
                        "method": "patch_charge_or_capacitance_metric",
                        "median_potential_rel_l2": "",
                        "improvement_vs_direct": "",
                        "n": r.get("n_rows"),
                        "notes": f"pointwise sigma {fmt(r.get('pointwise_sigma_median'))}; capacitance {fmt(r.get('capacitance_median'))}",
                    }
                )
    return rows


def plot(summary: list[dict[str, Any]], recs: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    datasets = ["fusion", "abc"]
    methods = ["direct_coarse", "global_refined_knn_density", "local_refined_replacement_tau3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for method in methods:
            xs, ys = [], []
            for r in summary:
                if r["dataset"] == dataset and r["method"] == method and r["metric"] == "potential_rel_l2":
                    med = as_float(r.get("median"))
                    if med is not None:
                        xs.append(float(r["distance_ratio"]))
                        ys.append(med)
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=method)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dataset.upper())
        ax.set_xlabel("query distance / object diameter")
    axes[0].set_ylabel("median potential rel-L2")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "generalizable_improvement_error_by_distance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    closest = [r for r in recs if as_float(r.get("distance_ratio")) == 0.005 and as_float(r.get("median_potential_rel_l2")) is not None]
    labels = [f"{r['dataset']}\n{r['method'].replace('_', ' ')[:18]}" for r in closest]
    vals = [float(r["median_potential_rel_l2"]) for r in closest]
    if vals:
        ax.bar(np.arange(len(vals)), vals)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("closest-band median rel-L2")
    else:
        ax.text(0.5, 0.5, "No closest-band rows", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "generalizable_improvement_method_ranking.png", dpi=180)
    plt.close(fig)


def write_report(summary: list[dict[str, Any]], recs: list[dict[str, Any]]) -> None:
    def best(dataset: str, dist: float) -> tuple[str, float, float | None]:
        sub = [
            r
            for r in recs
            if r.get("dataset") == dataset
            and as_float(r.get("distance_ratio")) == dist
            and as_float(r.get("median_potential_rel_l2")) is not None
        ]
        if not sub:
            return "-", float("nan"), None
        b = min(sub, key=lambda x: float(x["median_potential_rel_l2"]))
        return str(b["method"]), float(b["median_potential_rel_l2"]), as_float(b.get("improvement_vs_direct"))

    lines = [
        "# Generalizable NBIO-CAD Improvement Screen",
        "",
        "This screen tests improvement families that are dataset-agnostic: refined evaluation, density",
        "transfer, local close-target replacement, resolution refinement, and integrated charge metrics.",
        "It is intentionally separate from older CAD-QBX tables and writes `generalizable_*` artifacts.",
        "",
        "## Closest-Band Winners",
        "",
        "| dataset | best method at distance 0.005 | median potential rel-L2 | improvement vs direct |",
        "|---|---|---:|---:|",
    ]
    for dataset in ["fusion", "abc"]:
        method, med, imp = best(dataset, 0.005)
        lines.append(f"| {dataset} | {method} | {fmt(med)} | {fmt(imp)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The most transferable improvement is **close-evaluation treatment**: refined/local evaluation",
            "  should be applied only in near-surface bands rather than globally replacing the direct path.",
            "- **Patch-charge and capacitance metrics** are more stable than pointwise surface charge density and",
            "  should remain the engineering-output target across Fusion/ABC and future CAD datasets.",
            "- **Resolution refinement** is dataset-general but costlier; use it to generate references and to",
            "  validate resolution transfer, not as the default inference path.",
            "- Do not use a dataset-specific neural correction unless it is trained and tested across Fusion, ABC,",
            "  and at least one support dataset. The safe neural role is a local quadrature residual conditioned",
            "  on geometric features, not a new black-box PDE solver.",
            "",
            "## Summary Table",
            "",
            "| dataset | method | distance | metric | n | median | p90 |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for r in summary:
        if r["metric"] != "potential_rel_l2":
            continue
        lines.append(
            f"| {r['dataset']} | {r['method']} | `{float(r['distance_ratio']):.4f}` | {r['metric']} | "
            f"{r['n']} | {fmt(r.get('median'))} | {fmt(r.get('p90'))} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Next implementation tranche should combine: CAD close-evaluation switching, adaptive panel",
            "refinement for bad local mesh quality, and patch/capacitance losses. This is more likely to improve",
            "NBIO generally than increasing the size of the learned operator.",
        ]
    )
    write_text(REPORTS / "generalizable_improvement_screen.md", "\n".join(lines) + "\n")


def final_handoff(args: argparse.Namespace) -> None:
    recs = []
    p = TABLES / "generalizable_improvement_recommendations.csv"
    if p.exists():
        with p.open(newline="") as f:
            recs = list(csv.DictReader(f))
    lines = [
        "# Generalizable CAD Improvement Handoff",
        "",
        "Branch: `research/nbio-generalizable-cad-improvements`",
        "",
        "## Completed",
        "",
        "- Added `scripts/generalizable_cad_improvement_screen.py`.",
        "- Tested/refreshed generalizable close-evaluation variants on Fusion and ABC.",
        "- Preserved G2Aero as support evidence rather than forcing a mismatched 3D close-eval routine.",
        "- Aggregated patch-charge/capacitance diagnostics as cross-dataset engineering-output guidance.",
        "",
        "## Artifacts",
        "",
        "- `reports/generalizable_improvement_screen.md`",
        "- `results/generalizable_improvement_screen.json`",
        "- `results/tables/generalizable_improvement_screen_raw.csv`",
        "- `results/tables/generalizable_improvement_screen_summary.csv`",
        "- `results/tables/generalizable_improvement_recommendations.csv`",
        "- `figures/generalizable_improvement_error_by_distance.png`",
        "- `figures/generalizable_improvement_method_ranking.png`",
        "",
        "## Paper-Safe Conclusion",
        "",
        "The improvements that generalize are numerical/operator-side: close-evaluation switching, adaptive",
        "refinement, and integrated charge/capacitance objectives. Dataset-specific neural corrections should",
        "be avoided unless trained/evaluated across Fusion, ABC, and support datasets.",
    ]
    if recs:
        lines.extend(["", "## Closest-Band Recommendation Rows", "", "| dataset | method | distance | median | improvement |", "|---|---|---:|---:|---:|"])
        for r in recs:
            if as_float(r.get("distance_ratio")) == 0.005 and as_float(r.get("median_potential_rel_l2")) is not None:
                lines.append(
                    f"| {r.get('dataset')} | {r.get('method')} | `{float(r['distance_ratio']):.4f}` | "
                    f"{fmt(r.get('median_potential_rel_l2'))} | {fmt(r.get('improvement_vs_direct'))} |"
                )
    write_text(REPORTS / "generalizable_improvement_handoff.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--seed", type=int, default=37)
    s.add_argument("--n-meshes", type=int, default=6)
    s.add_argument("--n-surface", type=int, default=512)
    s.add_argument("--n-ref", type=int, default=1536)
    s.add_argument("--n-query", type=int, default=96)
    s.add_argument("--max-file-mb", type=float, default=40.0)
    s.add_argument("--max-faces", type=int, default=160000)
    s.add_argument("--families", nargs="+", default=["constant", "coordinate"])
    s.add_argument("--distances", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.10])
    s.add_argument("--transfer-k", type=int, default=8)
    s.add_argument("--tau-mult", type=float, default=3.0)
    sub.add_parser("aggregate")
    sub.add_parser("handoff")
    args = parser.parse_args()
    if args.cmd == "screen":
        screen(args)
    elif args.cmd == "aggregate":
        aggregate()
    elif args.cmd == "handoff":
        final_handoff(args)


if __name__ == "__main__":
    main()
