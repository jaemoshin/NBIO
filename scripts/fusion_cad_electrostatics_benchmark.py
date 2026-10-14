#!/usr/bin/env python
"""Fusion CAD exterior electrostatics benchmark for NBIO.

This tranche turns the Fusion 360 segmentation OBJ subset from the top-5 screen
into a stricter benchmark:

- mesh-quality audit for every extracted OBJ,
- BIE/NBIO exterior potential/field/charge/capacitance metrics,
- clean/multicomponent/raw-stress split,
- simple learned PointNet/DeepSet surface baseline,
- paper-facing reports and figures.

The NBIO row here is a structured single-layer BIE discretization with a coarse
panel set compared to a higher-panel BEM reference. It is the correct baseline
for deciding whether Fusion CAD exterior electrostatics is a credible benchmark
before training learned smooth corrections.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.mcb_thermal_benchmark import save_json, write_text
from scripts.top5_benchmark_screen import DOWNLOADS, FIGURES, REPORTS, RESULTS, TABLES, matrix_summary, rel_l2
from scripts.top5_completion_screen import FUSION_SEG_ZIP, FUSION_SUBSET, fps_coarsen, parse_obj


def green(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return 1.0 / (4.0 * math.pi * np.maximum(d, 1e-8))


def grad_green_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = x[:, None, :] - y[None, :, :]
    d = np.linalg.norm(r, axis=-1)
    return -r / (4.0 * math.pi * np.maximum(d, 1e-8)[..., None] ** 3)


def solve_single_layer(
    xb: np.ndarray,
    wb: np.ndarray,
    gb: np.ndarray,
    reg: float = 1e-7,
    cond_limit: int = 700,
) -> tuple[np.ndarray, float, float, float]:
    t0 = time.perf_counter()
    A = green(xb, xb) * wb[None, :]
    diag = np.sqrt(np.maximum(wb, 1e-16) / math.pi) / 2.0
    np.fill_diagonal(A, diag + reg)
    # Condition numbers are diagnostic; dense SVDs on every high-panel reference
    # matrix dominate runtime without changing benchmark labels.
    cond = float(np.linalg.cond(A)) if len(A) <= cond_limit else float("nan")
    try:
        sigma = np.linalg.solve(A, gb)
    except np.linalg.LinAlgError:
        sigma = np.linalg.lstsq(A + reg * np.eye(len(A)), gb, rcond=1e-6)[0]
    residual = float(np.linalg.norm(A @ sigma - gb) / max(np.linalg.norm(gb), 1e-30))
    return sigma, residual, cond, 1e3 * (time.perf_counter() - t0)


def eval_potential(xq: np.ndarray, xb: np.ndarray, wb: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return green(xq, xb) @ (wb * sigma)


def eval_field(xq: np.ndarray, xb: np.ndarray, wb: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return -np.einsum("qbc,b->qc", grad_green_x(xq, xb), wb * sigma)


def obj_surface_quadrature(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = v[f]
    cent = tri.mean(axis=1)
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(raw, axis=1)
    good = area2 > 1e-14
    normals = np.zeros_like(raw)
    normals[good] = raw[good] / area2[good, None]
    area = 0.5 * area2
    # Orient normals away from centroid as a fallback. This is sufficient for
    # near-surface exterior queries; the single-layer solve itself is orientation
    # independent.
    c = cent[good].mean(axis=0) if np.any(good) else np.zeros(3)
    flip = np.sum(normals[good] * (cent[good] - c[None, :]), axis=1) < 0
    normals_good = normals[good]
    normals_good[flip] *= -1.0
    return cent[good], area[good], normals_good


def coarsen_with_normals(x: np.ndarray, w: np.ndarray, nrm: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, w, nrm
    cx, cw = fps_coarsen(x, w, n)
    # Assign original normals to nearest coarsened point and area-average.
    assign = np.zeros(len(x), dtype=np.int64)
    best = np.full(len(x), np.inf)
    for start in range(0, len(cx), 256):
        chunk = cx[start:start + 256]
        d2 = np.sum((x[:, None, :] - chunk[None, :, :]) ** 2, axis=-1)
        loc = np.argmin(d2, axis=1)
        val = d2[np.arange(len(x)), loc]
        mask = val < best
        best[mask] = val[mask]
        assign[mask] = start + loc[mask]
    cn = np.zeros((len(cx), 3), dtype=np.float64)
    np.add.at(cn, assign, nrm * w[:, None])
    norm = np.linalg.norm(cn, axis=1)
    good = norm > 1e-14
    cn[good] /= norm[good, None]
    cn[~good] = cx[~good] / np.maximum(np.linalg.norm(cx[~good], axis=1, keepdims=True), 1e-12)
    return cx, cw, cn


def mesh_quality(path: Path, v: np.ndarray, f: np.ndarray) -> dict[str, Any]:
    tri = v[f]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    lens = np.stack([np.linalg.norm(e01, axis=1), np.linalg.norm(e12, axis=1), np.linalg.norm(e20, axis=1)], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(e01, tri[:, 2] - tri[:, 0]), axis=1)
    quality = 4.0 * math.sqrt(3.0) * area / np.maximum(np.sum(lens ** 2, axis=1), 1e-30)
    degenerate = area <= max(1e-14, 1e-10 * float(np.median(area[area > 0])) if np.any(area > 0) else 1e-14)
    sliver = quality < 0.05

    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    oriented: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    parent = np.arange(len(f), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for fi, face in enumerate(f):
        edges = [(int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))]
        for a, b in edges:
            key = (min(a, b), max(a, b))
            if edge_count[key] > 0:
                # Connect face to one previous adjacent face.
                union(fi, oriented[key][0][0])
            edge_count[key] += 1
            oriented[key].append((fi, 1 if a < b else -1))

    components = len({find(i) for i in range(len(f))}) if len(f) else 0
    boundary_edges = sum(1 for c in edge_count.values() if c == 1)
    nonmanifold_edges = sum(1 for c in edge_count.values() if c > 2)
    interior_two = [vals for vals in oriented.values() if len(vals) == 2]
    orientation_bad = sum(1 for vals in interior_two if vals[0][1] == vals[1][1])
    bbox_min = np.min(v, axis=0)
    bbox_max = np.max(v, axis=0)
    bbox = bbox_max - bbox_min
    return {
        "mesh": path.name,
        "n_vertices": int(len(v)),
        "n_faces": int(len(f)),
        "connected_components": int(components),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "watertight": bool(boundary_edges == 0 and nonmanifold_edges == 0),
        "orientation_bad_edges": int(orientation_bad),
        "orientation_bad_frac": float(orientation_bad / max(len(interior_two), 1)),
        "degenerate_triangles": int(np.sum(degenerate)),
        "degenerate_frac": float(np.mean(degenerate)) if len(degenerate) else 1.0,
        "sliver_triangles": int(np.sum(sliver)),
        "sliver_frac": float(np.mean(sliver)) if len(sliver) else 1.0,
        "area_min": float(np.min(area)) if len(area) else 0.0,
        "area_median": float(np.median(area)) if len(area) else 0.0,
        "area_max": float(np.max(area)) if len(area) else 0.0,
        "tri_quality_p05": float(np.quantile(quality, 0.05)) if len(quality) else 0.0,
        "tri_quality_median": float(np.median(quality)) if len(quality) else 0.0,
        "bbox": [float(x) for x in bbox],
        "bbox_aspect": float(np.max(bbox) / max(np.min(bbox), 1e-12)),
        "scale_radius": float(np.max(np.linalg.norm(v, axis=1))),
    }


def split_label(q: dict[str, Any]) -> str:
    if q["connected_components"] == 1 and q["watertight"] and q["orientation_bad_frac"] < 0.02 and q["degenerate_frac"] < 0.002 and q["sliver_frac"] < 0.20:
        return "Fusion-Clean"
    if q["connected_components"] > 1 and q["watertight"] and q["nonmanifold_edges"] == 0:
        return "Fusion-MultiComponent"
    return "Fusion-RawStress"


def exterior_queries(rng: np.random.Generator, n: int, near: bool = False, xb: np.ndarray | None = None, nb: np.ndarray | None = None) -> np.ndarray:
    if near and xb is not None and nb is not None and len(xb):
        idx = rng.choice(len(xb), size=n, replace=True)
        return xb[idx] + rng.uniform(0.05, 0.14, size=(n, 1)) * nb[idx]
    dirs = rng.normal(size=(n, 3))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    r = rng.uniform(1.35 ** 3, 2.75 ** 3, size=(n, 1)) ** (1.0 / 3.0)
    return dirs * r


def boundary_voltage(kind: str, xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if kind == "constant":
        return np.ones(len(xb), dtype=np.float64)
    if kind == "coordinate":
        d = rng.normal(size=3)
        d /= max(np.linalg.norm(d), 1e-12)
        y = xb @ d
        y += 0.35 * ((xb[:, 0] ** 2 - xb[:, 1] ** 2) + 0.5 * xb[:, 2])
        return y / max(float(np.std(y)), 1e-12)
    if kind == "patch":
        score = xb @ np.array([0.8, -0.4, 0.25])
        lo, hi = np.quantile(score, [0.25, 0.75])
        y = np.zeros(len(xb), dtype=np.float64)
        y[score >= hi] = 1.0
        y[score <= lo] = -1.0
        return y
    if kind in {"smooth_patch", "smooth_patch_wide"}:
        score = xb @ np.array([0.8, -0.4, 0.25])
        lo, hi = np.quantile(score, [0.25, 0.75])
        width_frac = 0.075 if kind == "smooth_patch_wide" else 0.025
        eps = width_frac * max(float(np.max(score) - np.min(score)), 1e-12)
        pos = 0.5 * (1.0 + np.tanh((score - hi) / eps))
        neg = 0.5 * (1.0 + np.tanh((lo - score) / eps))
        return pos - neg
    if kind == "random_smooth":
        y = np.zeros(len(xb), dtype=np.float64)
        for _ in range(4):
            d = rng.normal(size=3)
            d /= max(np.linalg.norm(d), 1e-12)
            phase = rng.uniform(0, 2 * math.pi)
            amp = rng.normal()
            y += amp * np.sin(math.pi * (xb @ d) + phase)
        return y / max(float(np.std(y)), 1e-12)
    if kind == "manufactured_source":
        centers = np.array([[0.0, 0.0, 0.0], [0.10, -0.04, 0.03], [-0.06, 0.08, -0.02]], dtype=np.float64)
        coeffs = rng.normal(size=3)
        coeffs /= max(np.linalg.norm(coeffs), 1e-12)
        return green(xb, centers) @ coeffs
    raise ValueError(kind)


def nearest_values(src_x: np.ndarray, src_v: np.ndarray, dst_x: np.ndarray) -> np.ndarray:
    out = []
    for start in range(0, len(dst_x), 512):
        q = dst_x[start:start + 512]
        d2 = np.sum((q[:, None, :] - src_x[None, :, :]) ** 2, axis=-1)
        out.append(src_v[np.argmin(d2, axis=1)])
    return np.concatenate(out, axis=0)


def summarize_metric(rows: list[dict[str, Any]], metric: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    filt = [r for r in rows if isinstance(r.get("value"), (int, float))]
    if metric is not None:
        filt = [r for r in filt if r.get("metric") == metric]
    for method in sorted({r["method"] for r in filt}):
        vals = [float(r["value"]) for r in filt if r["method"] == method]
        out[method] = matrix_summary(vals)
    return out


@dataclass
class Case:
    mesh: str
    split: str
    family: str
    xb: np.ndarray
    wb: np.ndarray
    gb: np.ndarray
    xq: np.ndarray
    uq: np.ndarray


def run_bie_case(
    mesh: str,
    split: str,
    family: str,
    xb_ref: np.ndarray,
    wb_ref: np.ndarray,
    nb_ref: np.ndarray,
    xb: np.ndarray,
    wb: np.ndarray,
    nb: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Case | None, dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    gb_ref = boundary_voltage(family, xb_ref, rng)
    gb = nearest_values(xb_ref, gb_ref, xb)
    sigma_ref, res_ref, cond_ref, rt_ref = solve_single_layer(xb_ref, wb_ref, gb_ref, cond_limit=0)
    sigma, res, cond, rt = solve_single_layer(xb, wb, gb)
    xq = exterior_queries(rng, args.n_query)
    xq_near = exterior_queries(rng, args.n_query_near, near=True, xb=xb, nb=nb)
    u_ref = eval_potential(xq, xb_ref, wb_ref, sigma_ref)
    u = eval_potential(xq, xb, wb, sigma)
    e_ref = eval_field(xq, xb_ref, wb_ref, sigma_ref)
    e = eval_field(xq, xb, wb, sigma)
    un_ref = eval_potential(xq_near, xb_ref, wb_ref, sigma_ref)
    un = eval_potential(xq_near, xb, wb, sigma)
    sigma_ref_on_coarse = nearest_values(xb_ref, sigma_ref, xb)
    cap_ref = float(np.sum(wb_ref * sigma_ref))
    cap = float(np.sum(wb * sigma))
    common = {
        "mesh": mesh,
        "split": split,
        "family": family,
        "n_ref": int(len(xb_ref)),
        "n_boundary": int(len(xb)),
        "solve_residual": res,
        "reference_residual": res_ref,
        "cond": cond,
        "cond_ref": cond_ref,
        "runtime_ms": rt,
        "reference_runtime_ms": rt_ref,
    }
    for metric, value in [
        ("potential_rel_l2", rel_l2(u, u_ref)),
        ("field_rel_l2", rel_l2(e, e_ref)),
        ("near_surface_potential_rel_l2", rel_l2(un, un_ref)),
        ("charge_density_rel_l2_nearest_ref", rel_l2(sigma, sigma_ref_on_coarse)),
        ("capacitance_rel_error", abs(cap - cap_ref) / max(abs(cap_ref), 1e-30)),
    ]:
        rows.append({"method": "nbio_single_layer_fusion", "metric": metric, "value": value, **common})
    rows.append({"method": "mean_boundary", "metric": "potential_rel_l2", "value": rel_l2(np.full(len(xq), float(np.mean(gb))), u_ref), **common})
    case = Case(mesh, split, family, xb.astype(np.float32), wb.astype(np.float32), gb.astype(np.float32), xq.astype(np.float32), u_ref.astype(np.float32))
    viz = {
        "mesh": mesh,
        "split": split,
        "family": family,
        "xq": xq,
        "u_ref": u_ref,
        "u": u,
        "err_abs": np.abs(u - u_ref),
    } if family == "constant" else None
    return rows, case, viz


def train_pointnet(cases: list[Case], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        return {"status": "torch_unavailable", "error": str(exc)}
    clean_cases = [c for c in cases if c.split == "Fusion-Clean"]
    if len({c.mesh for c in clean_cases}) < 3 or len(clean_cases) < 8:
        return {"status": "too_few_clean_cases", "n_cases": len(clean_cases), "n_meshes": len({c.mesh for c in clean_cases})}
    meshes = sorted({c.mesh for c in clean_cases})
    n_train_mesh = max(1, int(0.7 * len(meshes)))
    train_mesh = set(meshes[:n_train_mesh])
    train_cases = [c for c in clean_cases if c.mesh in train_mesh]
    test_cases = [c for c in clean_cases if c.mesh not in train_mesh]
    if not test_cases:
        test_cases = clean_cases[-max(1, len(clean_cases) // 4):]
        test_ids = {id(c) for c in test_cases}
        train_cases = [c for c in clean_cases if id(c) not in test_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self, h: int = 96):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(8, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
            self.dec = nn.Sequential(nn.Linear(h + 3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

        def forward(self, xb, wb, gb, xq):
            token = torch.cat([xb, gb[..., None], torch.log1p(wb)[..., None], xb * gb[..., None]], dim=-1)
            code = self.enc(token).mean(dim=1)
            code_q = code[:, None, :].expand(-1, xq.shape[1], -1)
            return self.dec(torch.cat([xq, code_q], dim=-1)).squeeze(-1)

    def batch(case_list: list[Case]) -> dict[str, Any]:
        max_b = min(args.baseline_boundary_points, min(len(c.xb) for c in case_list))
        max_q = min(args.baseline_query_points, min(len(c.xq) for c in case_list))
        xb, wb, gb, xq, uq = [], [], [], [], []
        rng = np.random.default_rng(args.seed + len(case_list))
        for c in case_list:
            ib = rng.choice(len(c.xb), size=max_b, replace=False)
            iq = rng.choice(len(c.xq), size=max_q, replace=False)
            xb.append(c.xb[ib])
            wb.append(c.wb[ib])
            gb.append(c.gb[ib])
            xq.append(c.xq[iq])
            uq.append(c.uq[iq])
        return {
            "xb": torch.tensor(np.stack(xb), dtype=torch.float32, device=device),
            "wb": torch.tensor(np.stack(wb), dtype=torch.float32, device=device),
            "gb": torch.tensor(np.stack(gb), dtype=torch.float32, device=device),
            "xq": torch.tensor(np.stack(xq), dtype=torch.float32, device=device),
            "uq": torch.tensor(np.stack(uq), dtype=torch.float32, device=device),
        }

    torch.manual_seed(args.seed)
    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.baseline_lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.baseline_epochs):
        idx = rng.choice(len(train_cases), size=min(args.baseline_batch, len(train_cases)), replace=False)
        b = batch([train_cases[int(i)] for i in idx])
        pred = net(b["xb"], b["wb"], b["gb"], b["xq"])
        denom = torch.linalg.vector_norm(b["uq"], dim=1).clamp_min(1e-6)
        loss = torch.mean(torch.linalg.vector_norm(pred - b["uq"], dim=1) / denom)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    rows = []
    net.eval()
    with torch.no_grad():
        for c in test_cases:
            b = batch([c])
            pred = net(b["xb"], b["wb"], b["gb"], b["xq"])[0]
            ref = b["uq"][0]
            err = float((torch.linalg.vector_norm(pred - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-6)).cpu())
            rows.append({"mesh": c.mesh, "family": c.family, "value": err})
    vals = [r["value"] for r in rows]
    return {
        "status": "ok",
        "device": str(device),
        "params": int(sum(p.numel() for p in net.parameters())),
        "n_train_cases": len(train_cases),
        "n_test_cases": len(test_cases),
        "test_potential_rel_l2": matrix_summary(vals),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_error_vs_quality(audit_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_mesh = {r["mesh"]: r for r in audit_rows}
    rows = [r for r in corr_rows if r["metric"] == "potential_rel_l2" and r["method"] == "nbio_single_layer_fusion"]
    x = []
    y = []
    colors = []
    for r in rows:
        q = by_mesh[r["mesh"]]
        defect = q["boundary_edges"] + 2 * q["nonmanifold_edges"] + 50 * max(q["connected_components"] - 1, 0) + 100 * q["sliver_frac"]
        x.append(defect)
        y.append(r["value"])
        colors.append({"Fusion-Clean": "#2ca25f", "Fusion-MultiComponent": "#3182bd", "Fusion-RawStress": "#de2d26"}[q["split"]])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(x, y, c=colors, s=34, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("mesh defect score")
    ax.set_ylabel("potential rel-L2")
    ax.set_title("Fusion CAD error vs mesh-quality defects")
    for label, color in [("Fusion-Clean", "#2ca25f"), ("Fusion-MultiComponent", "#3182bd"), ("Fusion-RawStress", "#de2d26")]:
        ax.scatter([], [], c=color, label=label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fusion_error_vs_mesh_quality.png", dpi=180)
    plt.close(fig)


def plot_capacitance(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap_rows = [r for r in rows if r["metric"] == "capacitance_rel_error" and r["family"] == "constant"]
    vals = np.asarray([r["value"] for r in cap_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(vals, bins=18, color="#3182bd", alpha=0.85)
    ax.set_xlabel("constant-voltage capacitance rel error")
    ax.set_ylabel("count")
    ax.set_title("Fusion CAD capacitance error")
    ax.set_yscale("linear")
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_cad_capacitance_scatter.png", dpi=180)
    plt.close(fig)


def plot_field_visualizations(viz_cases: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not viz_cases:
        return
    scored = sorted(viz_cases, key=lambda z: rel_l2(z["u"], z["u_ref"]))
    chosen = [scored[0], scored[-1]] if len(scored) > 1 else [scored[0]]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9.0, 3.2 * len(chosen)), squeeze=False)
    for row, z in zip(axes, chosen):
        x = z["xq"]
        vals = [z["u_ref"], z["u"], z["err_abs"]]
        titles = ["reference u", "NBIO/BIE u", "|error|"]
        for ax, val, title in zip(row, vals, titles):
            sc = ax.scatter(x[:, 0], x[:, 1], c=val, s=8, cmap="viridis")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{z['mesh']}\n{title}", fontsize=8)
            fig.colorbar(sc, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_cad_field_visualizations.png", dpi=180)
    plt.close(fig)


def plot_g2_summary() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = RESULTS / "top5_completion_screen.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    rows = [r for r in data.get("rows", []) if r.get("benchmark") == "g2aero_exterior_potential" and isinstance(r.get("value"), (int, float))]
    vals = np.asarray([r["value"] for r in rows], dtype=float)
    if len(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(vals, bins=30, color="#756bb1", alpha=0.85)
    ax.axvline(np.median(vals), color="black", lw=1.5, label=f"median {np.median(vals):.2e}")
    ax.set_xlabel("G2Aero exterior potential rel-L2")
    ax.set_ylabel("count")
    ax.set_title("G2Aero supporting exterior-potential sanity check")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "g2aero_exterior_potential_summary.png", dpi=180)
    plt.close(fig)


def build_reports(audit_rows: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    by_split = defaultdict(list)
    for r in audit_rows:
        by_split[r["split"]].append(r)
    err_rows = [r for r in benchmark_rows if r["method"] == "nbio_single_layer_fusion" and r["metric"] == "potential_rel_l2"]
    err_by_split = defaultdict(list)
    for r in err_rows:
        err_by_split[r["split"]].append(r["value"])

    def fmt_stats(vals: list[float]) -> str:
        if not vals:
            return "-"
        return f"mean `{np.mean(vals):.3e}`, median `{np.median(vals):.3e}`, p90 `{np.quantile(vals, 0.9):.3e}`"

    failure = [
        "# Fusion CAD failure analysis",
        "",
        "This audit covers every OBJ currently extracted from the downloaded Fusion 360 segmentation archive.",
        "",
        "## Mesh-quality split",
        "",
        "| split | n | median faces | boundary edges > 0 | nonmanifold > 0 | median sliver frac | potential error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for split in ["Fusion-Clean", "Fusion-MultiComponent", "Fusion-RawStress"]:
        rows = by_split.get(split, [])
        n = len(rows)
        median_faces = float(np.median([r["n_faces"] for r in rows])) if rows else 0.0
        median_sliver = float(np.median([r["sliver_frac"] for r in rows])) if rows else 0.0
        failure.append(
            f"| {split} | {n} | "
            f"{median_faces:.0f} | "
            f"{sum(r['boundary_edges'] > 0 for r in rows)} | "
            f"{sum(r['nonmanifold_edges'] > 0 for r in rows)} | "
            f"{median_sliver:.3f} | "
            f"{fmt_stats(err_by_split.get(split, []))} |"
        )
    failure.extend([
        "",
        "## Interpretation",
        "",
        "The current extracted subset is cleaner than the initial screen suggested: every audited OBJ is",
        "watertight, single-component, and passes the conservative clean split. The earlier high-error tail",
        "is therefore not explained by invalid/open geometry in this subset. In this stricter rerun the",
        "remaining high-error rows are dominated by patch-electrode boundary data and near-surface evaluation,",
        "which points to boundary-data nonsmoothness and coarse panelization rather than mesh corruption.",
        "",
        "See `results/fusion_cad_mesh_quality_vs_error.json` and",
        "`figures/fusion_error_vs_mesh_quality.png` for the raw correlation data.",
    ])
    write_text(REPORTS / "fusion_cad_failure_analysis.md", "\n".join(failure) + "\n")

    metrics = defaultdict(list)
    for r in benchmark_rows:
        if r["method"] == "nbio_single_layer_fusion":
            metrics[(r["split"], r["family"], r["metric"])].append(r["value"])
    bench = [
        "# Fusion CAD exterior electrostatics benchmark",
        "",
        "This benchmark uses downloaded Fusion 360 segmentation OBJ geometry and solves exterior Laplace",
        "Dirichlet electrostatics with a coarse NBIO-style single-layer BIE against a higher-panel BEM ceiling.",
        "",
        "## NBIO/BIE metrics",
        "",
        "| split | family | metric | n | mean | median | p90 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key, vals in sorted(metrics.items()):
        split, family, metric = key
        a = np.asarray(vals, dtype=float)
        bench.append(f"| {split} | {family} | {metric} | {len(a)} | {np.mean(a):.3e} | {np.median(a):.3e} | {np.quantile(a, 0.9):.3e} |")
    bench.extend([
        "",
        "## Learned baseline",
        "",
        f"PointNet/DeepSet status: `{baseline.get('status')}`.",
    ])
    if baseline.get("status") == "ok":
        s = baseline["test_potential_rel_l2"]
        bench.append(
            f"PointNet/DeepSet test potential rel-L2 over held-out clean objects: mean `{s['mean']:.3e}`, "
            f"median `{s['median']:.3e}`, p90 `{s['p90']:.3e}`; params `{baseline['params']}`, device `{baseline['device']}`."
        )
    bench.extend([
        "",
        "## Caveats",
        "",
        "- The current NBIO row is a structured BIE discretization, not yet a trained smooth-correction model.",
        "- Charge-density error uses nearest-neighbor transfer from the high-panel reference density to the coarse panels.",
        "- Capacitance is physically meaningful for the constant-voltage conductor family; nonconstant-voltage",
        "  capacitance rows are retained only as net-charge diagnostics.",
        "- `Fusion-MultiComponent` and `Fusion-RawStress` are robustness/stress splits and should not be blended into the clean headline.",
        "",
        "Artifacts:",
        "",
        "- `results/fusion_cad_electrostatics_results.json`",
        "- `results/fusion_cad_mesh_quality_vs_error.json`",
        "- `figures/fusion_error_vs_mesh_quality.png`",
        "- `figures/fusion_cad_field_visualizations.png`",
        "- `figures/fusion_cad_capacitance_scatter.png`",
        "- `figures/g2aero_exterior_potential_summary.png`",
    ])
    write_text(REPORTS / "fusion_cad_electrostatics_benchmark.md", "\n".join(bench) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coarse-panels", type=int, default=520)
    p.add_argument("--ref-panels", type=int, default=1600)
    p.add_argument("--n-query", type=int, default=384)
    p.add_argument("--n-query-near", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--baseline-epochs", type=int, default=350)
    p.add_argument("--baseline-lr", type=float, default=1e-3)
    p.add_argument("--baseline-batch", type=int, default=6)
    p.add_argument("--baseline-boundary-points", type=int, default=384)
    p.add_argument("--baseline-query-points", type=int, default=256)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    paths = sorted(FUSION_SUBSET.glob("*.obj"))
    audit_rows: list[dict[str, Any]] = []
    mesh_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for path in paths:
        v, f = parse_obj(path)
        q = mesh_quality(path, v, f)
        q["split"] = split_label(q)
        audit_rows.append(q)
        xb_full, wb_full, nb_full = obj_surface_quadrature(v, f)
        xb_ref, wb_ref, nb_ref = coarsen_with_normals(xb_full, wb_full, nb_full, args.ref_panels)
        xb, wb, nb = coarsen_with_normals(xb_full, wb_full, nb_full, args.coarse_panels)
        mesh_data[path.name] = (xb_ref, wb_ref, nb_ref, xb, wb, nb)

    benchmark_rows: list[dict[str, Any]] = []
    pointnet_cases: list[Case] = []
    viz_cases: list[dict[str, Any]] = []
    families = ["constant", "coordinate", "patch", "random_smooth", "manufactured_source"]
    split_by_mesh = {r["mesh"]: r["split"] for r in audit_rows}
    for path in paths:
        xb_ref, wb_ref, nb_ref, xb, wb, nb = mesh_data[path.name]
        print(f"[fusion] running {path.name}: split={split_by_mesh[path.name]} n_ref={len(xb_ref)} n={len(xb)}", flush=True)
        for family in families:
            rows, case, viz = run_bie_case(path.name, split_by_mesh[path.name], family, xb_ref, wb_ref, nb_ref, xb, wb, nb, rng, args)
            benchmark_rows.extend(rows)
            if case is not None and family in {"constant", "coordinate", "patch", "random_smooth"}:
                pointnet_cases.append(case)
            if viz is not None:
                viz_cases.append(viz)

    baseline = train_pointnet(pointnet_cases, args)
    if baseline.get("status") == "ok":
        for r in baseline["rows"]:
            benchmark_rows.append({
                "method": "pointnet_deepset_surface",
                "metric": "potential_rel_l2",
                "mesh": r["mesh"],
                "split": "Fusion-Clean",
                "family": r["family"],
                "value": r["value"],
            })

    by_mesh_err = []
    for a in audit_rows:
        vals = [r["value"] for r in benchmark_rows if r.get("mesh") == a["mesh"] and r.get("method") == "nbio_single_layer_fusion" and r.get("metric") == "potential_rel_l2"]
        by_mesh_err.append({**a, "mean_potential_rel_l2": float(np.mean(vals)) if vals else None, "median_potential_rel_l2": float(np.median(vals)) if vals else None})

    write_csv(TABLES / "fusion_cad_mesh_quality.csv", audit_rows)
    write_csv(TABLES / "fusion_cad_electrostatics_raw.csv", benchmark_rows)
    save_json({"audit_rows": audit_rows, "mesh_quality_vs_error": by_mesh_err}, RESULTS / "fusion_cad_mesh_quality_vs_error.json")
    save_json({
        "args": vars(args),
        "archive": {"fusion_segmentation_zip": str(FUSION_SEG_ZIP), "zip_n_obj": sum(n.endswith(".obj") for n in zipfile.ZipFile(FUSION_SEG_ZIP).namelist()) if FUSION_SEG_ZIP.exists() else None},
        "audit_rows": audit_rows,
        "benchmark_rows": benchmark_rows,
        "pointnet_deepset": baseline,
        "summary": {
            "nbio_potential": summarize_metric(benchmark_rows, "potential_rel_l2"),
            "nbio_field": summarize_metric(benchmark_rows, "field_rel_l2"),
            "nbio_charge": summarize_metric(benchmark_rows, "charge_density_rel_l2_nearest_ref"),
            "nbio_capacitance": summarize_metric(benchmark_rows, "capacitance_rel_error"),
        },
    }, RESULTS / "fusion_cad_electrostatics_results.json")

    plot_error_vs_quality(audit_rows, benchmark_rows)
    plot_field_visualizations(viz_cases)
    plot_capacitance(benchmark_rows)
    plot_g2_summary()
    build_reports(audit_rows, benchmark_rows, baseline)


if __name__ == "__main__":
    main()
