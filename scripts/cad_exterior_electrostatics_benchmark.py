#!/usr/bin/env python
"""CAD exterior electrostatics benchmark.

This focused tranche turns the CAD boundary-physics screen into a benchmark that
looks like a real operator-learning task:

    surface geometry + surface voltage + exterior query points
      -> potential, electric field, charge density, capacitance.

Geometry comes from the cleaned/tetrahedralized MCB CAD subset. The exterior
potential/field task uses manufactured interior charges for exact labels. The
constant-voltage capacitance task uses a high-panel single-layer BEM solve as a
reference and evaluates coarser NBIO-style BIE discretizations against it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.mcb_thermal_benchmark import (
    FIGURES,
    REPORTS,
    RESULTS,
    boundary_panel_quadrature,
    coarsen_boundary_quadrature,
    load_msh_arrays,
    save_json,
    tet_volumes,
    write_text,
)

TARGET_CATEGORIES = ["screws_and_bolts", "nut", "motor", "fitting", "gear"]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def green(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = x[:, None, :] - y[None, :, :]
    d = np.linalg.norm(r, axis=-1)
    return 1.0 / (4.0 * math.pi * np.maximum(d, 1e-8))


def grad_green_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of 1/(4*pi*|x-y|) with respect to x."""
    r = x[:, None, :] - y[None, :, :]
    d = np.linalg.norm(r, axis=-1)
    return -r / (4.0 * math.pi * np.maximum(d, 1e-8)[..., None] ** 3)


def source_potential(x: np.ndarray, centers: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return green(x, centers) @ coeffs


def source_field(x: np.ndarray, centers: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    # E = -grad u.
    return -np.einsum("qsc,s->qc", grad_green_x(x, centers), coeffs)


def solve_single_layer(
    xb: np.ndarray,
    wb: np.ndarray,
    gb: np.ndarray,
    reg: float = 1e-7,
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    A = green(xb, xb) * wb[None, :]
    # Constant-panel self term for a disk with same panel area.
    diag = np.sqrt(np.maximum(wb, 1e-16) / math.pi) / 2.0
    np.fill_diagonal(A, diag + reg)
    try:
        sigma = np.linalg.solve(A, gb)
    except np.linalg.LinAlgError:
        sigma = np.linalg.lstsq(A + reg * np.eye(len(A)), gb, rcond=1e-6)[0]
    residual = float(np.linalg.norm(A @ sigma - gb) / max(np.linalg.norm(gb), 1e-30))
    return sigma, residual, 1e3 * (time.perf_counter() - t0)


def eval_single_layer_potential(xq: np.ndarray, xb: np.ndarray, wb: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return green(xq, xb) @ (wb * sigma)


def eval_single_layer_field(xq: np.ndarray, xb: np.ndarray, wb: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # E = -grad S[sigma].
    return -np.einsum("qbc,b->qc", grad_green_x(xq, xb), wb * sigma)


def total_charge(wb: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum(wb * sigma))


def choose_inside_sources(cent: np.ndarray, vols: np.ndarray, rng: np.random.Generator, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    p = vols / np.sum(vols) if np.sum(vols) > 0 else None
    idx = rng.choice(len(cent), size=n, replace=True, p=p)
    coeffs = rng.normal(size=n)
    coeffs = coeffs / max(np.linalg.norm(coeffs), 1e-12)
    return cent[idx], coeffs


def exterior_queries(rng: np.random.Generator, n: int, r_min: float = 1.15, r_max: float = 3.0) -> np.ndarray:
    dirs = rng.normal(size=(n, 3))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    rad = rng.uniform(r_min ** 3, r_max ** 3, size=(n, 1)) ** (1.0 / 3.0)
    return dirs * rad


def near_surface_queries(xb: np.ndarray, nb: np.ndarray, rng: np.random.Generator, n: int, offset: float = 0.08) -> np.ndarray:
    idx = rng.choice(len(xb), size=n, replace=True)
    jitter = rng.uniform(0.5, 1.5, size=(n, 1)) * offset
    return xb[idx] + jitter * nb[idx]


def nearest_boundary_potential(xb: np.ndarray, gb: np.ndarray, xq: np.ndarray) -> np.ndarray:
    out = []
    for start in range(0, len(xq), 512):
        q = xq[start:start + 512]
        d2 = np.sum((q[:, None, :] - xb[None, :, :]) ** 2, axis=-1)
        out.append(gb[np.argmin(d2, axis=1)])
    return np.concatenate(out, axis=0)


def decay_mean_potential(xb: np.ndarray, gb: np.ndarray, xq: np.ndarray) -> np.ndarray:
    center = np.mean(xb, axis=0)
    rb = np.linalg.norm(xb - center[None, :], axis=1)
    rq = np.linalg.norm(xq - center[None, :], axis=1)
    return np.mean(gb) * np.mean(rb) / np.maximum(rq, 1e-8)


def zero_field(xq: np.ndarray) -> np.ndarray:
    return np.zeros((len(xq), 3), dtype=np.float64)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    valid = [r for r in rows if isinstance(r.get("value"), (int, float))]
    for metric in sorted(set(r["metric"] for r in valid)):
        out[metric] = {}
        mr = [r for r in valid if r["metric"] == metric]
        for method in sorted(set(r["method"] for r in mr)):
            rr = [r for r in mr if r["method"] == method]
            vals = np.asarray([r["value"] for r in rr], dtype=float)
            rt = [r["runtime_ms"] for r in rr if isinstance(r.get("runtime_ms"), (int, float))]
            out[metric][method] = {
                "n": len(rr),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p90": float(np.quantile(vals, 0.9)),
                "max": float(np.max(vals)),
                "runtime_ms_mean": float(np.mean(rt)) if rt else None,
                "by_category": {},
            }
            for cat in sorted(set(r["category"] for r in rr)):
                cv = [r["value"] for r in rr if r["category"] == cat]
                out[metric][method]["by_category"][cat] = float(np.mean(cv))
    return out


def write_tables(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    table_dir = RESULTS / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    raw_path = table_dir / "cad_es_raw.csv"
    fields = [
        "metric", "method", "category", "mesh_stem", "case_id", "value",
        "runtime_ms", "n_boundary", "n_query", "solve_residual", "cap_ref",
        "cap_pred", "boundary_rule",
    ]
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    sum_path = table_dir / "cad_es_summary.csv"
    with sum_path.open("w", newline="") as f:
        fields = ["metric", "method", "n", "mean", "median", "p90", "max", "runtime_ms_mean"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for metric, methods in summary.items():
            for method, s in methods.items():
                w.writerow({"metric": metric, "method": method, **{k: s.get(k) for k in fields[2:]}})


def plot_summary(summary: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["potential_rel_l2", "field_rel_l2", "near_surface_potential_rel_l2", "capacitance_rel_error"]
    methods = ["nbio_single_layer", "nearest_boundary", "decay_mean", "zero_field"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.6), sharey=False)
    for ax, metric in zip(axes, metrics):
        labels = [m for m in methods if metric in summary and m in summary[metric]]
        vals = [summary[metric][m]["median"] for m in labels]
        ax.bar(range(len(labels)), vals)
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([m.replace("_", "\n") for m in labels], fontsize=7)
        ax.set_title(metric.replace("_", " "), fontsize=8)
        ax.set_ylabel("median relative error")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_capacitance(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rr = [r for r in rows if r.get("metric") == "capacitance_rel_error" and r.get("method") == "nbio_single_layer"]
    if not rr:
        return
    ref = np.asarray([r["cap_ref"] for r in rr], dtype=float)
    pred = np.asarray([r["cap_pred"] for r in rr], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(ref, pred, s=18, alpha=0.75)
    lo = min(float(ref.min()), float(pred.min()))
    hi = max(float(ref.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("reference capacitance / charge")
    ax.set_ylabel("coarse NBIO capacitance / charge")
    ax.set_title("CAD electrostatic capacitance")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(obj: dict[str, Any]) -> None:
    s = obj["summary"]
    lines = [
        "# CAD Exterior Electrostatics Benchmark",
        "",
        "This benchmark targets the paper setup recommended after the NGF/MCB comparison: real CAD geometry with",
        "boundary-only exterior physics. Each sample maps surface geometry and voltage data to exterior potential,",
        "electric field, surface charge density, and capacitance-like total charge.",
        "",
        "Geometry comes from the cleaned/tetrahedralized MCB subset. The potential/field labels are manufactured",
        "from interior point charges, so they are exact exterior harmonic fields. The constant-voltage capacitance",
        "task uses a higher-panel BEM solve as the reference and compares a coarser NBIO-style BIE solve.",
        "",
        f"Meshes: `{obj['mesh_count']}`; cases/mesh: `{obj['cases_per_mesh']}`; coarse panels: `{obj['coarse_boundary']}`;",
        f"reference panels: `{obj['ref_boundary']}`; exterior queries/case: `{obj['n_query']}`.",
        "",
        "## Results",
        "",
        "| metric | method | n | mean | median | p90 | max | ms/sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, methods in s.items():
        for method, row in sorted(methods.items()):
            rt = row.get("runtime_ms_mean")
            rt_s = f"`{rt:.2f}`" if isinstance(rt, (int, float)) else "-"
            lines.append(
                f"| {metric} | {method} | `{row['n']}` | `{row['mean']:.3e}` | `{row['median']:.3e}` | "
                f"`{row['p90']:.3e}` | `{row['max']:.3e}` | {rt_s} |"
            )
    nbio_pot = s.get("potential_rel_l2", {}).get("nbio_single_layer", {})
    nbio_field = s.get("field_rel_l2", {}).get("nbio_single_layer", {})
    nbio_cap = s.get("capacitance_rel_error", {}).get("nbio_single_layer", {})
    lines += [
        "",
        "## Interpretation",
        "",
        f"NBIO/coarse BIE reaches median potential rel-L2 `{nbio_pot.get('median', float('nan')):.3e}` and",
        f"field rel-L2 `{nbio_field.get('median', float('nan')):.3e}` on real CAD exterior queries.",
        f"Constant-voltage capacitance has median relative error `{nbio_cap.get('median', float('nan')):.3e}`",
        "against the high-panel BEM reference.",
        "",
        "This is a much better main benchmark candidate than raw MCB volume-source thermal Poisson: it keeps the",
        "real mechanical-part geometry aesthetic, but the physics is boundary-only and exterior/unbounded, which",
        "is exactly where NBIO's representation is structurally meaningful.",
        "",
        "Caveats: this is still a structured-solver screen, not a final neural benchmark. The next required step is",
        "to train the learned smooth correction and add strong learned geometry baselines on the exact same CAD-ES",
        "data. The high-panel BEM reference is also still first-order panel BEM, so capacitance should be described",
        "as a BEM-reference task rather than an analytic ground truth task.",
        "",
        "Artifacts:",
        "",
        "- `results/cad_es_results.json`",
        "- `results/tables/cad_es_raw.csv`",
        "- `results/tables/cad_es_summary.csv`",
        "- `figures/cad_es_summary.png`",
        "- `figures/cad_es_capacitance_scatter.png`",
    ]
    write_text(REPORTS / "cad_es_nbio_comparison.md", "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    tet_path = RESULTS / "mcb_tet_quality.json"
    if not tet_path.exists():
        raise FileNotFoundError(f"run/keep MCB tetra pass first: {tet_path}")
    tet = json.loads(tet_path.read_text())
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in tet["rows"]:
        if r.get("status") == "ok" and Path(r.get("msh", "")).exists() and r.get("category") in TARGET_CATEGORIES:
            by_cat[r["category"]].append(r)
    meshes = []
    for cat in TARGET_CATEGORIES:
        meshes.extend(by_cat.get(cat, [])[: args.meshes_per_category])

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for mi, row in enumerate(meshes):
        pts, tets, tris = load_msh_arrays(Path(row["msh"]))
        if len(tets) == 0 or len(tris) == 0:
            continue
        vols = tet_volumes(pts, tets)
        cent = pts[tets].mean(axis=1)
        xb_full, wb_full, nb_full = boundary_panel_quadrature(pts, tris, tets)
        xb_ref, wb_ref, nb_ref, ref_rule = coarsen_boundary_quadrature(
            xb_full, wb_full, nb_full, args.ref_boundary, rng
        )
        xb, wb, nb, coarse_rule = coarsen_boundary_quadrature(
            xb_full, wb_full, nb_full, args.coarse_boundary, rng
        )
        cat = row["category"]
        stem = row["stem"]

        # Capacitance-like constant-voltage task: high-panel BEM reference.
        g_ref = np.ones(len(xb_ref), dtype=np.float64)
        sigma_ref, res_ref, ms_ref = solve_single_layer(xb_ref, wb_ref, g_ref)
        cap_ref = total_charge(wb_ref, sigma_ref)
        g_coarse = np.ones(len(xb), dtype=np.float64)
        sigma_c, res_c, ms_c = solve_single_layer(xb, wb, g_coarse)
        cap_pred = total_charge(wb, sigma_c)
        rows.append({
            "metric": "capacitance_rel_error",
            "method": "nbio_single_layer",
            "category": cat,
            "mesh_stem": stem,
            "case_id": "constant_voltage",
            "value": abs(cap_pred - cap_ref) / max(abs(cap_ref), 1e-30),
            "runtime_ms": ms_c,
            "n_boundary": len(xb),
            "n_query": 0,
            "solve_residual": res_c,
            "cap_ref": cap_ref,
            "cap_pred": cap_pred,
            "boundary_rule": f"coarse={coarse_rule};ref={ref_rule};ref_res={res_ref:.2e};ref_ms={ms_ref:.2f}",
        })

        for case_id in range(args.cases_per_mesh):
            crng = np.random.default_rng(args.seed + 10007 * mi + case_id)
            src, coeff = choose_inside_sources(cent, vols, crng, n=args.n_sources)
            xq_far = exterior_queries(crng, args.n_query)
            xq_near = near_surface_queries(xb, nb, crng, args.n_query, offset=args.near_offset)
            gb = source_potential(xb, src, coeff)
            uq_far = source_potential(xq_far, src, coeff)
            uq_near = source_potential(xq_near, src, coeff)
            eq_far = source_field(xq_far, src, coeff)

            sigma, res, ms_solve = solve_single_layer(xb, wb, gb)
            t0 = time.perf_counter()
            pred_u = eval_single_layer_potential(xq_far, xb, wb, sigma)
            pred_near = eval_single_layer_potential(xq_near, xb, wb, sigma)
            pred_e = eval_single_layer_field(xq_far, xb, wb, sigma)
            ms_eval = 1e3 * (time.perf_counter() - t0)
            runtime = ms_solve + ms_eval

            rows.extend([
                {
                    "metric": "potential_rel_l2", "method": "nbio_single_layer",
                    "category": cat, "mesh_stem": stem, "case_id": case_id,
                    "value": rel_l2(pred_u, uq_far), "runtime_ms": runtime,
                    "n_boundary": len(xb), "n_query": len(xq_far), "solve_residual": res,
                    "boundary_rule": coarse_rule,
                },
                {
                    "metric": "near_surface_potential_rel_l2", "method": "nbio_single_layer",
                    "category": cat, "mesh_stem": stem, "case_id": case_id,
                    "value": rel_l2(pred_near, uq_near), "runtime_ms": runtime,
                    "n_boundary": len(xb), "n_query": len(xq_near), "solve_residual": res,
                    "boundary_rule": coarse_rule,
                },
                {
                    "metric": "field_rel_l2", "method": "nbio_single_layer",
                    "category": cat, "mesh_stem": stem, "case_id": case_id,
                    "value": rel_l2(pred_e, eq_far), "runtime_ms": runtime,
                    "n_boundary": len(xb), "n_query": len(xq_far), "solve_residual": res,
                    "boundary_rule": coarse_rule,
                },
            ])

            for method, upred in [
                ("nearest_boundary", nearest_boundary_potential(xb, gb, xq_far)),
                ("decay_mean", decay_mean_potential(xb, gb, xq_far)),
            ]:
                rows.append({
                    "metric": "potential_rel_l2", "method": method,
                    "category": cat, "mesh_stem": stem, "case_id": case_id,
                    "value": rel_l2(upred, uq_far), "runtime_ms": None,
                    "n_boundary": len(xb), "n_query": len(xq_far), "solve_residual": "",
                    "boundary_rule": coarse_rule,
                })
            rows.append({
                "metric": "field_rel_l2", "method": "zero_field",
                "category": cat, "mesh_stem": stem, "case_id": case_id,
                "value": rel_l2(zero_field(xq_far), eq_far), "runtime_ms": None,
                "n_boundary": len(xb), "n_query": len(xq_far), "solve_residual": "",
                "boundary_rule": coarse_rule,
            })
        print(f"[cad-es] {mi + 1}/{len(meshes)} {stem} done", flush=True)

    summary = summarize(rows)
    obj = {
        "task": "cad_exterior_electrostatics",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mesh_count": len(meshes),
        "cases_per_mesh": args.cases_per_mesh,
        "coarse_boundary": args.coarse_boundary,
        "ref_boundary": args.ref_boundary,
        "n_query": args.n_query,
        "near_offset": args.near_offset,
        "n_sources": args.n_sources,
        "rows": rows,
        "summary": summary,
    }
    save_json(obj, RESULTS / "cad_es_results.json")
    write_tables(rows, summary)
    plot_summary(summary, FIGURES / "cad_es_summary.png")
    plot_capacitance(rows, FIGURES / "cad_es_capacitance_scatter.png")
    write_report(obj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meshes-per-category", type=int, default=6)
    p.add_argument("--cases-per-mesh", type=int, default=4)
    p.add_argument("--coarse-boundary", type=int, default=900)
    p.add_argument("--ref-boundary", type=int, default=2000)
    p.add_argument("--n-query", type=int, default=512)
    p.add_argument("--n-sources", type=int, default=4)
    p.add_argument("--near-offset", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=23)
    run(p.parse_args())


if __name__ == "__main__":
    main()
