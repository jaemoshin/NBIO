#!/usr/bin/env python
"""Validate generalizable NBIO-CAD inference policies on larger CAD samples.

This is additive and writes `generalizable_switch_validation*` artifacts.  It
does not overwrite the exploratory close-evaluation screen.
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
    green,
    nearest_values,
    obj_surface_quadrature,
    solve_single_layer,
)
from scripts.mcb_thermal_benchmark import save_json, write_text
from scripts.top5_benchmark_screen import FIGURES, RESULTS, TABLES, rel_l2
from scripts.top5_completion_screen import parse_obj

REPORTS = Path("reports")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[write] {path}")


def as_float(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def fmt(x: Any) -> str:
    y = as_float(x)
    return "-" if y is None else f"`{y:.3e}`"


def sample_voltage_spec(kind: str, xb_ref: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    if kind == "constant":
        return {"kind": kind}
    if kind == "coordinate":
        d = rng.normal(size=3)
        d /= max(np.linalg.norm(d), 1e-12)
        y_ref = xb_ref @ d
        y_ref += 0.35 * ((xb_ref[:, 0] ** 2 - xb_ref[:, 1] ** 2) + 0.5 * xb_ref[:, 2])
        return {"kind": kind, "direction": d.tolist(), "scale": max(float(np.std(y_ref)), 1e-12)}
    if kind == "patch":
        direction = np.array([0.8, -0.4, 0.25], dtype=np.float64)
        score = xb_ref @ direction
        lo, hi = np.quantile(score, [0.25, 0.75])
        return {"kind": kind, "direction": direction.tolist(), "lo": float(lo), "hi": float(hi)}
    if kind in {"smooth_patch", "smooth_patch_wide"}:
        direction = np.array([0.8, -0.4, 0.25], dtype=np.float64)
        score = xb_ref @ direction
        lo, hi = np.quantile(score, [0.25, 0.75])
        width_frac = 0.075 if kind == "smooth_patch_wide" else 0.025
        eps = width_frac * max(float(np.max(score) - np.min(score)), 1e-12)
        return {"kind": kind, "direction": direction.tolist(), "lo": float(lo), "hi": float(hi), "eps": float(eps)}
    if kind == "random_smooth":
        dirs = []
        phases = []
        amps = []
        y_ref = np.zeros(len(xb_ref), dtype=np.float64)
        for _ in range(4):
            d = rng.normal(size=3)
            d /= max(np.linalg.norm(d), 1e-12)
            phase = float(rng.uniform(0, 2 * np.pi))
            amp = float(rng.normal())
            dirs.append(d.tolist())
            phases.append(phase)
            amps.append(amp)
            y_ref += amp * np.sin(np.pi * (xb_ref @ d) + phase)
        return {"kind": kind, "dirs": dirs, "phases": phases, "amps": amps, "scale": max(float(np.std(y_ref)), 1e-12)}
    if kind == "manufactured_source":
        coeffs = rng.normal(size=3)
        coeffs /= max(np.linalg.norm(coeffs), 1e-12)
        return {"kind": kind, "coeffs": coeffs.tolist()}
    raise ValueError(kind)


def eval_voltage_spec(spec: dict[str, Any], xb: np.ndarray) -> np.ndarray:
    kind = spec["kind"]
    if kind == "constant":
        return np.ones(len(xb), dtype=np.float64)
    if kind == "coordinate":
        d = np.asarray(spec["direction"], dtype=np.float64)
        y = xb @ d
        y += 0.35 * ((xb[:, 0] ** 2 - xb[:, 1] ** 2) + 0.5 * xb[:, 2])
        return y / max(float(spec["scale"]), 1e-12)
    if kind == "patch":
        d = np.asarray(spec["direction"], dtype=np.float64)
        score = xb @ d
        y = np.zeros(len(xb), dtype=np.float64)
        y[score >= float(spec["hi"])] = 1.0
        y[score <= float(spec["lo"])] = -1.0
        return y
    if kind in {"smooth_patch", "smooth_patch_wide"}:
        d = np.asarray(spec["direction"], dtype=np.float64)
        score = xb @ d
        eps = max(float(spec["eps"]), 1e-12)
        pos = 0.5 * (1.0 + np.tanh((score - float(spec["hi"])) / eps))
        neg = 0.5 * (1.0 + np.tanh((float(spec["lo"]) - score) / eps))
        return pos - neg
    if kind == "random_smooth":
        y = np.zeros(len(xb), dtype=np.float64)
        for d, phase, amp in zip(spec["dirs"], spec["phases"], spec["amps"]):
            y += float(amp) * np.sin(np.pi * (xb @ np.asarray(d, dtype=np.float64)) + float(phase))
        return y / max(float(spec["scale"]), 1e-12)
    if kind == "manufactured_source":
        centers = np.array([[0.0, 0.0, 0.0], [0.10, -0.04, 0.03], [-0.06, 0.08, -0.02]], dtype=np.float64)
        return green(xb, centers) @ np.asarray(spec["coeffs"], dtype=np.float64)
    raise ValueError(kind)


def patch_edge_coarsen_with_normals(
    x: np.ndarray,
    w: np.ndarray,
    nrm: np.ndarray,
    n: int,
    voltage_spec: dict[str, Any],
    edge_budget_frac: float,
    edge_band_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if voltage_spec.get("kind") != "patch" or len(x) <= n:
        return coarsen_with_normals(x, w, nrm, n)
    d = np.asarray(voltage_spec["direction"], dtype=np.float64)
    score = x @ d
    span = max(float(np.max(score) - np.min(score)), 1e-12)
    band = max(edge_band_frac * span, 1e-12)
    lo = float(voltage_spec["lo"])
    hi = float(voltage_spec["hi"])
    edge = (np.abs(score - lo) <= band) | (np.abs(score - hi) <= band)
    if int(np.sum(edge)) < 16 or int(np.sum(~edge)) < 16:
        return coarsen_with_normals(x, w, nrm, n)
    n_edge = int(round(n * edge_budget_frac))
    n_edge = min(max(8, n_edge), int(np.sum(edge)))
    n_rest = min(n - n_edge, int(np.sum(~edge)))
    if n_rest <= 0:
        return coarsen_with_normals(x, w, nrm, n)
    xe, we, ne = coarsen_with_normals(x[edge], w[edge], nrm[edge], n_edge)
    xr, wr, nr = coarsen_with_normals(x[~edge], w[~edge], nrm[~edge], n_rest)
    return np.concatenate([xe, xr], axis=0), np.concatenate([we, wr], axis=0), np.concatenate([ne, nr], axis=0)


def evaluate_policy(
    policy: str,
    dist_ratio: float,
    xq: np.ndarray,
    xb: np.ndarray,
    wb: np.ndarray,
    sigma: np.ndarray,
    xb_ref: np.ndarray,
    wb_ref: np.ndarray,
    sigma_knn: np.ndarray,
    tau_mult: float,
    boundary_scale: float = 1.0,
    guard_factor: float = 25.0,
    piecewise_constant_boundary: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, float, int]:
    guard_replaced = 0
    if policy == "direct_all":
        method = "direct_coarse"
        t0 = time.perf_counter()
        u = eval_potential(xq, xb, wb, sigma)
        e = eval_field(xq, xb, wb, sigma)
    elif policy == "piecewise_constant_direct_guard":
        if piecewise_constant_boundary:
            method = "piecewise_constant_direct_coarse"
            t0 = time.perf_counter()
            u = eval_potential(xq, xb, wb, sigma)
            e = eval_field(xq, xb, wb, sigma)
        elif dist_ratio <= 0.005:
            method = "global_refined_knn_density"
            t0 = time.perf_counter()
            u = eval_global_refined(xq, xb_ref, wb_ref, sigma_knn)
            e = eval_field_global_refined(xq, xb_ref, wb_ref, sigma_knn)
        elif dist_ratio <= 0.010:
            method = f"local_refined_replacement_tau{tau_mult:g}"
            t0 = time.perf_counter()
            u = eval_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, tau_mult)
            e = eval_field_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, tau_mult)
        else:
            method = "direct_coarse"
            t0 = time.perf_counter()
            u = eval_potential(xq, xb, wb, sigma)
            e = eval_field(xq, xb, wb, sigma)
    elif policy == "robust_refined_all":
        method = "global_refined_knn_density"
        t0 = time.perf_counter()
        u = eval_global_refined(xq, xb_ref, wb_ref, sigma_knn)
        e = eval_field_global_refined(xq, xb_ref, wb_ref, sigma_knn)
    elif policy == "fast_switch_le_0p01":
        if dist_ratio <= 0.005:
            method = "global_refined_knn_density"
            t0 = time.perf_counter()
            u = eval_global_refined(xq, xb_ref, wb_ref, sigma_knn)
            e = eval_field_global_refined(xq, xb_ref, wb_ref, sigma_knn)
        elif dist_ratio <= 0.010:
            method = f"local_refined_replacement_tau{tau_mult:g}"
            t0 = time.perf_counter()
            u = eval_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, tau_mult)
            e = eval_field_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, tau_mult)
        else:
            method = "direct_coarse"
            t0 = time.perf_counter()
            u = eval_potential(xq, xb, wb, sigma)
            e = eval_field(xq, xb, wb, sigma)
    elif policy == "safe_switch_guard":
        if dist_ratio <= 0.005:
            method = "global_refined_knn_density"
            t0 = time.perf_counter()
            u = eval_global_refined(xq, xb_ref, wb_ref, sigma_knn)
            e = eval_field_global_refined(xq, xb_ref, wb_ref, sigma_knn)
        elif dist_ratio <= 0.010:
            method = f"local_refined_replacement_tau{tau_mult:g}"
            t0 = time.perf_counter()
            u = eval_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, tau_mult)
            e = eval_field_local_replacement(xq, xb, wb, sigma, xb_ref, wb_ref, sigma_knn, 2.0 * dist_ratio, tau_mult)
        else:
            method = "direct_coarse_with_refined_guard"
            t0 = time.perf_counter()
            u = eval_potential(xq, xb, wb, sigma)
            e = eval_field(xq, xb, wb, sigma)
            guard = np.abs(u) > guard_factor * max(boundary_scale, 1e-12)
            if np.any(guard):
                ur = eval_global_refined(xq, xb_ref, wb_ref, sigma_knn)
                er = eval_field_global_refined(xq, xb_ref, wb_ref, sigma_knn)
                u = u.copy()
                e = e.copy()
                u[guard] = ur[guard]
                e[guard] = er[guard]
                guard_replaced = int(np.sum(guard))
    else:
        raise ValueError(f"Unknown policy {policy}")
    return u, e, method, 1e3 * (time.perf_counter() - t0), guard_replaced


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    dataset_builders = {
        "fusion": lambda: get_fusion_paths(args.n_meshes, expanded=True),
        "abc": lambda: get_abc_paths(args.n_meshes, max_mb=args.max_file_mb, max_faces=args.max_faces),
    }
    datasets = [(name, dataset_builders[name]()) for name in args.datasets]
    policies = args.policies
    for dataset, paths in datasets:
        for imesh, path in enumerate(paths):
            try:
                v, f = parse_obj(path)
                xb_full, wb_full, nb_full = obj_surface_quadrature(v, f)
                xb_default, wb_default, nb_default = coarsen_with_normals(xb_full, wb_full, nb_full, args.n_surface)
                xb_ref, wb_ref, _ = coarsen_with_normals(xb_full, wb_full, nb_full, args.n_ref)
                mesh = file_key(path) if dataset == "abc" and ABC_SUBSET in path.parents else path.name
                for family in args.families:
                    if args.voltage_mode == "analytic_shared":
                        voltage_spec = sample_voltage_spec(family, xb_ref, rng)
                        gb_ref = eval_voltage_spec(voltage_spec, xb_ref)
                    else:
                        voltage_spec = None
                        gb_ref = boundary_voltage(family, xb_ref, rng)
                    if (
                        args.coarsen_mode == "patch_edge"
                        and args.voltage_mode == "analytic_shared"
                        and family == "patch"
                        and voltage_spec is not None
                    ):
                        xb, wb, nb = patch_edge_coarsen_with_normals(
                            xb_full,
                            wb_full,
                            nb_full,
                            args.n_surface,
                            voltage_spec,
                            args.edge_budget_frac,
                            args.edge_band_frac,
                        )
                    else:
                        xb, wb, nb = xb_default, wb_default, nb_default
                    if args.voltage_mode == "analytic_shared":
                        gb = eval_voltage_spec(voltage_spec, xb)
                    else:
                        gb = nearest_values(xb_ref, gb_ref, xb)
                    unique_levels = int(len(np.unique(np.round(gb, decimals=8))))
                    piecewise_constant_boundary = unique_levels <= args.piecewise_constant_levels
                    boundary_scale = float(max(np.percentile(np.abs(gb), 95), np.std(gb), np.mean(np.abs(gb)), 1e-12))
                    sigma_ref, res_ref, _, rt_ref = solve_single_layer(xb_ref, wb_ref, gb_ref, cond_limit=0)
                    sigma, res, cond, rt = solve_single_layer(xb, wb, gb)
                    sigma_knn = knn_transfer(xb, sigma, xb_ref, k=args.transfer_k)
                    for dist_ratio in args.distances:
                        idx = rng.choice(len(xb), size=args.n_query, replace=True)
                        xq = xb[idx] + (2.0 * dist_ratio) * nb[idx]
                        u_ref = eval_potential(xq, xb_ref, wb_ref, sigma_ref)
                        e_ref = eval_field(xq, xb_ref, wb_ref, sigma_ref)
                        for policy in policies:
                            u, e, selected, eval_ms, guard_replaced = evaluate_policy(
                                policy,
                                dist_ratio,
                                xq,
                                xb,
                                wb,
                                sigma,
                                xb_ref,
                                wb_ref,
                                sigma_knn,
                                args.tau_mult,
                                boundary_scale,
                                args.guard_factor,
                                piecewise_constant_boundary,
                            )
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "mesh": mesh,
                                    "family": family,
                                    "policy": policy,
                                    "selected_method": selected,
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
                                    "guard_replaced": guard_replaced,
                                    "boundary_scale": boundary_scale,
                                    "boundary_unique_levels": unique_levels,
                                    "piecewise_constant_boundary": piecewise_constant_boundary,
                                    "voltage_mode": args.voltage_mode,
                                    "voltage_spec": json.dumps(voltage_spec) if voltage_spec is not None else "",
                                    "coarsen_mode": args.coarsen_mode,
                                    "edge_budget_frac": args.edge_budget_frac if args.coarsen_mode == "patch_edge" else "",
                                    "edge_band_frac": args.edge_band_frac if args.coarsen_mode == "patch_edge" else "",
                                }
                            )
                save_json(
                    {
                        "status": "partial",
                        "dataset": dataset,
                        "completed_meshes": imesh + 1,
                        "rows": rows,
                        "failures": failures,
                        "args": vars(args),
                    },
                    RESULTS / f"{args.tag}_partial.json",
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"dataset": dataset, "mesh": str(path), "error": repr(exc)})
    result = {"status": "ok", "rows": rows, "failures": failures, "args": vars(args)}
    save_json(result, RESULTS / f"{args.tag}.json")
    write_csv(TABLES / f"{args.tag}_raw.csv", rows)
    summarize(args.tag, rows, failures)


def summarize(tag: str, rows: list[dict[str, Any]] | None = None, failures: list[dict[str, Any]] | None = None) -> None:
    if rows is None:
        data = json.loads((RESULTS / f"{tag}.json").read_text())
        rows = data.get("rows", [])
        failures = data.get("failures", [])
    failures = failures or []
    summary: list[dict[str, Any]] = []
    for dataset in ["all"] + sorted({str(r["dataset"]) for r in rows}):
        sub0 = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
        for policy in sorted({str(r["policy"]) for r in sub0}):
            sub = [r for r in sub0 if r["policy"] == policy]
            for metric in ["potential_rel_l2", "field_rel_l2", "eval_runtime_ms"]:
                vals = [as_float(r.get(metric)) for r in sub]
                vals = [v for v in vals if v is not None]
                summary.append({"dataset": dataset, "policy": policy, "metric": metric, **finite_summary(vals)})
    write_csv(TABLES / f"{tag}_summary.csv", summary)
    plot(tag, summary)
    write_report(tag, summary, failures)


def plot(tag: str, summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = [r for r in summary if r["dataset"] == "all" and r["metric"] == "potential_rel_l2"]
    order = [
        "direct_all",
        "fast_switch_le_0p01",
        "piecewise_constant_direct_guard",
        "safe_switch_guard",
        "robust_refined_all",
    ]
    rows = sorted(rows, key=lambda r: order.index(r["policy"]) if r["policy"] in order else 99)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [float(r["median"]) for r in rows], width=0.36, label="median")
    ax.bar(x + 0.18, [float(r["p90"]) for r in rows], width=0.36, label="p90")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["policy"].replace("_", "\n") for r in rows], fontsize=7)
    ax.set_ylabel("potential rel-L2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{tag}_potential_policy.png", dpi=180)
    plt.close(fig)


def write_report(tag: str, summary: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    def get(dataset: str, policy: str, metric: str) -> dict[str, Any] | None:
        for r in summary:
            if r["dataset"] == dataset and r["policy"] == policy and r["metric"] == metric:
                return r
        return None

    lines = [
        "# Generalizable Switch-Policy Validation",
        "",
        f"Tag: `{tag}`",
        "",
        "This validates fixed, dataset-general NBIO inference policies on a larger Fusion/ABC sample.",
        "",
        "Policies:",
        "",
        "- `direct_all`: current direct coarse evaluation everywhere.",
        "- `fast_switch_le_0p01`: close-evaluation only for query distances up to 1% of object scale.",
        "- `piecewise_constant_direct_guard`: direct evaluation for piecewise-constant electrode data, fast switch otherwise.",
        "- `safe_switch_guard`: fast switch plus a prediction-scale guard that falls back to refined evaluation.",
        "- `robust_refined_all`: refined KNN-density evaluation everywhere, for accuracy mode.",
        "",
        "## Pooled Results",
        "",
        "| policy | median potential | p90 potential | median field | median eval ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ["direct_all", "fast_switch_le_0p01", "piecewise_constant_direct_guard", "safe_switch_guard", "robust_refined_all"]:
        pot = get("all", policy, "potential_rel_l2")
        fld = get("all", policy, "field_rel_l2")
        rt = get("all", policy, "eval_runtime_ms")
        if pot and fld and rt:
            lines.append(f"| {policy} | {fmt(pot['median'])} | {fmt(pot['p90'])} | {fmt(fld['median'])} | {fmt(rt['median'])} |")
    lines.extend(["", "## Dataset Breakdown", "", "| dataset | policy | median potential | p90 potential |", "|---|---|---:|---:|"])
    for dataset in ["fusion", "abc"]:
        for policy in ["direct_all", "fast_switch_le_0p01", "piecewise_constant_direct_guard", "safe_switch_guard", "robust_refined_all"]:
            pot = get(dataset, policy, "potential_rel_l2")
            if pot:
                lines.append(f"| {dataset} | {policy} | {fmt(pot['median'])} | {fmt(pot['p90'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use `fast_switch_le_0p01` as the default inference policy when runtime matters. Use",
            "`robust_refined_all` as the accuracy/reference mode when outlier suppression and field accuracy",
            "matter more than raw evaluation speed.",
            "",
            f"Failures: `{len(failures)}`.",
        ]
    )
    write_text(REPORTS / f"{tag}.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="generalizable_switch_validation")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--n-meshes", type=int, default=24)
    parser.add_argument("--n-surface", type=int, default=512)
    parser.add_argument("--n-ref", type=int, default=1536)
    parser.add_argument("--n-query", type=int, default=128)
    parser.add_argument("--max-file-mb", type=float, default=40.0)
    parser.add_argument("--max-faces", type=int, default=160000)
    parser.add_argument("--families", nargs="+", default=["constant", "coordinate"])
    parser.add_argument("--datasets", nargs="+", choices=["fusion", "abc"], default=["fusion", "abc"])
    parser.add_argument("--voltage-mode", choices=["nearest_ref", "analytic_shared"], default="nearest_ref")
    parser.add_argument("--coarsen-mode", choices=["default", "patch_edge"], default="default")
    parser.add_argument("--edge-budget-frac", type=float, default=0.45)
    parser.add_argument("--edge-band-frac", type=float, default=0.035)
    parser.add_argument("--piecewise-constant-levels", type=int, default=5)
    parser.add_argument("--distances", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--transfer-k", type=int, default=8)
    parser.add_argument("--tau-mult", type=float, default=3.0)
    parser.add_argument("--guard-factor", type=float, default=25.0)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "direct_all",
            "fast_switch_le_0p01",
            "piecewise_constant_direct_guard",
            "safe_switch_guard",
            "robust_refined_all",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
