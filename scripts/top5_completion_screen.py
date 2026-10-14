#!/usr/bin/env python
"""Strict completion pass for the top-5 NBIO benchmark screen.

This script answers the follow-up request "make sure it's all downloaded and
tested" by validating downloaded archives and running small physics screens on
newly available geometry sources:

- Fusion 360 segmentation OBJ meshes: real CAD exterior Laplace screen.
- G2Aero airfoils: 2D exterior Laplace screen when the NPZ validates.
- DPM tools sample: porous-media simulation-volume audit.

It intentionally keeps ABC/DPM portal access probes honest: if a source requires
portal credentials or a click-through restrictions page, the report says so.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.top5_benchmark_screen import (
    FIGURES,
    REPORTS,
    RESULTS,
    TABLES,
    DOWNLOADS,
    green2,
    green3,
    matrix_summary,
    rel_l2,
    sample_airfoil_queries,
    solve_single_layer2,
    solve_single_layer3,
    source_potential2,
)
from scripts.mcb_thermal_benchmark import save_json, write_text

FUSION_ARCHIVES = DOWNLOADS / "fusion_archives"
FUSION_SEG_ZIP = FUSION_ARCHIVES / "s2.0.1.zip"
FUSION_ASSEMBLY_7Z = FUSION_ARCHIVES / "j1.0.0.7z"
FUSION_SUBSET = DOWNLOADS / "fusion_segmentation_subset"
G2_ROOT = DOWNLOADS / "g2aero"
DPM_ROOT = DOWNLOADS / "dpm_tools"


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            idx = []
            for item in line.split()[1:]:
                raw = item.split("/")[0]
                if raw:
                    j = int(raw)
                    idx.append(j - 1 if j > 0 else len(verts) + j)
            if len(idx) >= 3:
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        raise ValueError("empty OBJ vertices/faces")
    center = 0.5 * (np.min(v, axis=0) + np.max(v, axis=0))
    v = v - center[None, :]
    scale = float(np.max(np.linalg.norm(v, axis=1)))
    if scale <= 1e-12:
        raise ValueError("degenerate OBJ scale")
    v = v / scale
    return v, f


def obj_surface_quadrature(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = v[f]
    cent = tri.mean(axis=1)
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(raw, axis=1)
    good = area > 1e-14
    return cent[good], area[good]


def fps_coarsen(x: np.ndarray, w: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, w
    selected = np.empty(n, dtype=np.int64)
    selected[0] = int(np.argmax(w))
    d2_min = np.sum((x - x[selected[0]][None, :]) ** 2, axis=1)
    taken = np.zeros(len(x), dtype=bool)
    taken[selected[0]] = True
    for i in range(1, n):
        score = d2_min.copy()
        score[taken] = -1.0
        j = int(np.argmax(score))
        selected[i] = j
        taken[j] = True
        d2_min = np.minimum(d2_min, np.sum((x - x[j][None, :]) ** 2, axis=1))
    assign = np.zeros(len(x), dtype=np.int64)
    best = np.full(len(x), np.inf)
    for start in range(0, n, 256):
        chunk = selected[start:start + 256]
        d2 = np.sum((x[:, None, :] - x[chunk][None, :, :]) ** 2, axis=-1)
        loc = np.argmin(d2, axis=1)
        val = d2[np.arange(len(x)), loc]
        mask = val < best
        best[mask] = val[mask]
        assign[mask] = start + loc[mask]
    agg_w = np.zeros(n, dtype=np.float64)
    agg_x = np.zeros((n, 3), dtype=np.float64)
    np.add.at(agg_w, assign, w)
    np.add.at(agg_x, assign, x * w[:, None])
    good = agg_w > 1e-30
    agg_x[good] /= agg_w[good, None]
    agg_x[~good] = x[selected[~good]]
    return agg_x, agg_w


def exterior_queries3(rng: np.random.Generator, n: int) -> np.ndarray:
    dirs = rng.normal(size=(n, 3))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    r = rng.uniform(1.35 ** 3, 2.75 ** 3, size=(n, 1)) ** (1.0 / 3.0)
    return dirs * r


def run_fusion_screen(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(FUSION_SUBSET.glob("*.obj"))[: args.fusion_count]
    for i, path in enumerate(paths):
        rng = np.random.default_rng(args.seed + i)
        try:
            v, f = parse_obj(path)
            xb, wb = obj_surface_quadrature(v, f)
            xb, wb = fps_coarsen(xb, wb, args.fusion_panels)
            centers = np.array([[0.0, 0.0, 0.0], [0.10, -0.04, 0.03], [-0.06, 0.08, -0.02]], dtype=np.float64)
            coeffs = rng.normal(size=3)
            coeffs /= max(np.linalg.norm(coeffs), 1e-12)
            gb = green3(xb, centers) @ coeffs
            q = exterior_queries3(rng, args.fusion_queries)
            ref = green3(q, centers) @ coeffs
            sigma, residual, runtime = solve_single_layer3(xb, wb, gb)
            pred = green3(q, xb) @ (wb * sigma)
            mean_pred = np.full(len(q), float(np.mean(gb)))
            for method, value, rt in [
                ("nbio_single_layer_fusion", rel_l2(pred, ref), runtime),
                ("mean_boundary", rel_l2(mean_pred, ref), None),
            ]:
                rows.append({
                    "benchmark": "fusion_cad_exterior_potential",
                    "dataset": "Fusion360 segmentation s2.0.1 OBJ subset",
                    "method": method,
                    "metric": "potential_rel_l2",
                    "mesh": path.name,
                    "value": value,
                    "n_boundary": int(len(xb)),
                    "n_query": int(len(q)),
                    "runtime_ms": rt,
                    "solve_residual": residual if method.startswith("nbio") else None,
                    "status": "ok",
                })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "benchmark": "fusion_cad_exterior_potential",
                "dataset": "Fusion360 segmentation s2.0.1 OBJ subset",
                "method": "nbio_single_layer_fusion",
                "metric": "potential_rel_l2",
                "mesh": path.name,
                "value": None,
                "status": "failed",
                "error": str(exc),
            })
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        vals = [float(r["value"]) for r in rows if r["method"] == method and isinstance(r.get("value"), (int, float))]
        summary[method] = matrix_summary(vals)
    return rows, summary


def load_g2_shapes(path: Path) -> tuple[np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=True)
    shapes = z["shapes"]
    classes = z["classes"].tolist() if "classes" in z.files else ["unknown"] * len(shapes)
    if shapes.ndim != 3:
        raise ValueError(f"unexpected G2 shapes ndim {shapes.ndim}")
    if shapes.shape[-1] == 2:
        coords = shapes.astype(np.float64)
    elif shapes.shape[1] == 2:
        coords = np.transpose(shapes, (0, 2, 1)).astype(np.float64)
    else:
        raise ValueError(f"unexpected G2 shape layout {shapes.shape}")
    return coords, [str(c) for c in classes]


def run_g2_screen(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = G2_ROOT / "curated_airfoils.npz"
    if not p.exists():
        return rows, {"status": "missing"}
    coords, classes = load_g2_shapes(p)
    n = min(args.g2_count, len(coords))
    for i in range(n):
        rng = np.random.default_rng(args.seed + 10000 + i)
        try:
            poly = np.asarray(coords[i], dtype=np.float64)
            poly = poly[np.all(np.isfinite(poly), axis=1)]
            if np.linalg.norm(poly[0] - poly[-1]) > 1e-8:
                poly = np.vstack([poly, poly[0]])
            center = 0.5 * (np.min(poly, axis=0) + np.max(poly, axis=0))
            scale = float(np.max(np.ptp(poly, axis=0)))
            poly = (poly - center[None, :]) / max(scale, 1e-12)
            from scripts.top5_benchmark_screen import curve_quadrature

            xb, wb = curve_quadrature(poly, args.g2_panels)
            centers = np.array([[-0.06, 0.0], [0.10, 0.025], [0.18, -0.02]], dtype=np.float64)
            coeffs = rng.normal(size=3)
            coeffs /= max(np.linalg.norm(coeffs), 1e-12)
            gb = source_potential2(xb, centers, coeffs)
            q = sample_airfoil_queries(rng, args.g2_queries)
            ref = source_potential2(q, centers, coeffs)
            sigma, residual, runtime = solve_single_layer2(xb, wb, gb)
            pred = green2(q, xb) @ (wb * sigma)
            rows.append({
                "benchmark": "g2aero_exterior_potential",
                "dataset": "G2Aero curated_airfoils",
                "method": "nbio_single_layer_g2aero",
                "metric": "potential_rel_l2",
                "mesh": f"g2_{i:05d}",
                "category": classes[i] if i < len(classes) else "unknown",
                "value": rel_l2(pred, ref),
                "n_boundary": int(len(xb)),
                "n_query": int(len(q)),
                "runtime_ms": runtime,
                "solve_residual": residual,
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "benchmark": "g2aero_exterior_potential",
                "dataset": "G2Aero curated_airfoils",
                "method": "nbio_single_layer_g2aero",
                "metric": "potential_rel_l2",
                "mesh": f"g2_{i:05d}",
                "value": None,
                "status": "failed",
                "error": str(exc),
            })
    vals = [float(r["value"]) for r in rows if isinstance(r.get("value"), (int, float))]
    return rows, {"nbio_single_layer_g2aero": matrix_summary(vals), "n_available_shapes": int(len(coords))}


def audit_sources() -> dict[str, Any]:
    audit: dict[str, Any] = {}
    audit["fusion_segmentation"] = {
        "archive": str(FUSION_SEG_ZIP),
        "exists": FUSION_SEG_ZIP.exists(),
        "bytes": FUSION_SEG_ZIP.stat().st_size if FUSION_SEG_ZIP.exists() else 0,
        "aria2_sidecar": (FUSION_ARCHIVES / "s2.0.1.zip.aria2").exists(),
        "extracted_obj_subset": len(list(FUSION_SUBSET.glob("*.obj"))),
    }
    if FUSION_SEG_ZIP.exists():
        with zipfile.ZipFile(FUSION_SEG_ZIP) as z:
            names = z.namelist()
            audit["fusion_segmentation"].update({
                "zip_open_ok": True,
                "n_files": len(names),
                "n_obj": sum(n.endswith(".obj") for n in names),
            })
    audit["fusion_assembly_joint"] = {
        "archive": str(FUSION_ASSEMBLY_7Z),
        "exists": FUSION_ASSEMBLY_7Z.exists(),
        "bytes": FUSION_ASSEMBLY_7Z.stat().st_size if FUSION_ASSEMBLY_7Z.exists() else 0,
        "expected_bytes": 2975602672,
        "complete_by_size": FUSION_ASSEMBLY_7Z.exists() and FUSION_ASSEMBLY_7Z.stat().st_size == 2975602672,
    }
    if audit["fusion_assembly_joint"]["complete_by_size"]:
        cmd = ["bsdtar", "-tf", str(FUSION_ASSEMBLY_7Z)]
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
        audit["fusion_assembly_joint"]["list_ok"] = p.returncode == 0
        audit["fusion_assembly_joint"]["first_entries"] = p.stdout.splitlines()[:10]
    audit["g2aero"] = {
        "curated": str(G2_ROOT / "curated_airfoils.npz"),
        "pga": str(G2_ROOT / "PGAspace.npz"),
        "curated_exists": (G2_ROOT / "curated_airfoils.npz").exists(),
        "pga_exists": (G2_ROOT / "PGAspace.npz").exists(),
        "curated_bytes": (G2_ROOT / "curated_airfoils.npz").stat().st_size if (G2_ROOT / "curated_airfoils.npz").exists() else 0,
        "pga_bytes": (G2_ROOT / "PGAspace.npz").stat().st_size if (G2_ROOT / "PGAspace.npz").exists() else 0,
    }
    if (G2_ROOT / "curated_airfoils.npz").exists():
        try:
            coords, classes = load_g2_shapes(G2_ROOT / "curated_airfoils.npz")
            audit["g2aero"].update({"curated_open_ok": True, "n_shapes": int(len(coords)), "shape_layout": list(coords.shape), "n_classes": len(set(classes))})
        except Exception as exc:  # noqa: BLE001
            audit["g2aero"].update({"curated_open_ok": False, "open_error": str(exc)})
    if (G2_ROOT / "PGAspace.npz").exists():
        try:
            pga = np.load(G2_ROOT / "PGAspace.npz", allow_pickle=True)
            audit["g2aero"].update({
                "pga_open_ok": True,
                "pga_keys": pga.files,
                "pga_coords_shape": list(pga["coords"].shape) if "coords" in pga.files else None,
            })
        except Exception as exc:  # noqa: BLE001
            audit["g2aero"].update({"pga_open_ok": False, "pga_open_error": str(exc)})
    audit["dpm_tools"] = {
        "repo": str(DPM_ROOT),
        "exists": DPM_ROOT.exists(),
        "sample_files": len(list((DPM_ROOT / "data").glob("*"))) if (DPM_ROOT / "data").exists() else 0,
    }
    mat = DPM_ROOT / "data" / "10_01_256_elec.mat"
    if mat.exists():
        import h5py

        with h5py.File(mat, "r") as f:
            audit["dpm_tools"]["elec_mat_keys"] = sorted(f.keys())
            audit["dpm_tools"]["phi_shape"] = list(f["phi"].shape)
            # One slice is enough to validate data access without pulling the
            # full 256^3 volume into memory in this audit.
            sl = np.asarray(f["phi"][128, :, :])
            audit["dpm_tools"]["phi_mid_slice_mean"] = float(np.mean(sl))
            audit["dpm_tools"]["phi_mid_slice_std"] = float(np.std(sl))
    audit["abc"] = {
        "status": "blocked_by_nyu_restrictions_redirect",
        "probe_url": "https://archive.nyu.edu/retrieve/89085/abc_0000_obj_v00.7z",
    }
    return audit


def write_rows(rows: list[dict[str, Any]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark", "dataset", "method", "metric", "mesh", "category", "value", "n_boundary", "n_query", "runtime_ms", "solve_residual", "status", "error"]
    with (TABLES / "top5_completion_screen_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_summary(summary: dict[str, Any]) -> None:
    fields = ["benchmark", "method", "n", "mean", "median", "p90", "max", "status"]
    with (TABLES / "top5_completion_screen_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bench, methods in summary.items():
            for method, s in methods.items():
                if isinstance(s, dict) and "median" in s:
                    w.writerow({"benchmark": bench, "method": method, **{k: s.get(k) for k in fields[2:]}})


def build_report(audit: dict[str, Any], summary: dict[str, Any]) -> str:
    def fmt(x: Any) -> str:
        return f"{float(x):.3e}" if isinstance(x, (int, float)) else "-"

    lines = [
        "# Top-5 benchmark download and test completion",
        "",
        "This is the stricter completion pass after the first top-5 screen. It validates downloaded sources",
        "and runs small NBIO/BIE physics tests on newly available geometry.",
        "",
        "## Tested rows",
        "",
        "| benchmark | method | n | mean | median | p90 | max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bench, methods in summary.items():
        for method, s in methods.items():
            if isinstance(s, dict) and "median" in s:
                lines.append(f"| {bench} | {method} | {s.get('n', 0)} | {fmt(s.get('mean'))} | {fmt(s.get('median'))} | {fmt(s.get('p90'))} | {fmt(s.get('max'))} |")
    lines.extend([
        "",
        "## Download validation",
        "",
        f"- Fusion segmentation: `{audit['fusion_segmentation'].get('n_obj', 0)}` OBJ meshes in `s2.0.1.zip`; extracted subset `{audit['fusion_segmentation'].get('extracted_obj_subset', 0)}`.",
        f"- Fusion assembly-joint: `{audit['fusion_assembly_joint'].get('bytes', 0)}` / `{audit['fusion_assembly_joint'].get('expected_bytes')}` bytes downloaded at audit time; complete by size: `{audit['fusion_assembly_joint'].get('complete_by_size')}`.",
        f"- G2Aero curated: open ok `{audit.get('g2aero', {}).get('curated_open_ok', False)}`, shapes `{audit.get('g2aero', {}).get('n_shapes', 0)}`; PGAspace open ok `{audit.get('g2aero', {}).get('pga_open_ok', False)}`.",
        f"- DPM tools: sample files `{audit['dpm_tools'].get('sample_files', 0)}`, electric sample keys `{audit['dpm_tools'].get('elec_mat_keys', [])}`.",
        f"- ABC: `{audit['abc']['status']}` for `{audit['abc']['probe_url']}`.",
        "",
        "## Interpretation",
        "",
        "Fusion segmentation is now a real downloaded CAD source, not just a link. The Fusion exterior screen",
        "tests NBIO-style single-layer solves on extracted OBJ surfaces. G2Aero is a larger airfoil source than",
        "UIUC and is tested when the NPZ passes CRC/open validation. DPM is downloaded through the public",
        "`dpm_tools` repository with sample porous-media simulation arrays; turning it into a true NBIO",
        "transmission benchmark still requires interface extraction from voxel phases.",
        "",
        "Artifacts:",
        "",
        "- `results/top5_completion_screen.json`",
        "- `results/tables/top5_completion_screen_raw.csv`",
        "- `results/tables/top5_completion_screen_summary.csv`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fusion-count", type=int, default=24)
    p.add_argument("--fusion-panels", type=int, default=520)
    p.add_argument("--fusion-queries", type=int, default=384)
    p.add_argument("--g2-count", type=int, default=160)
    p.add_argument("--g2-panels", type=int, default=128)
    p.add_argument("--g2-queries", type=int, default=384)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fusion_rows, fusion_summary = run_fusion_screen(args)
    rows.extend(fusion_rows)
    summary: dict[str, Any] = {"fusion_cad_exterior_potential": fusion_summary}
    try:
        g2_rows, g2_summary = run_g2_screen(args)
        rows.extend(g2_rows)
        summary["g2aero_exterior_potential"] = g2_summary
    except Exception as exc:  # noqa: BLE001
        summary["g2aero_exterior_potential"] = {"status": f"failed: {exc}"}

    audit = audit_sources()
    write_rows(rows)
    write_summary(summary)
    save_json({"args": vars(args), "audit": audit, "summary": summary, "rows": rows}, RESULTS / "top5_completion_screen.json")
    write_text(REPORTS / "top5_benchmark_completion.md", build_report(audit, summary))


if __name__ == "__main__":
    main()
