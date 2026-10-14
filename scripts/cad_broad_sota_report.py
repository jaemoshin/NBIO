#!/usr/bin/env python
"""Summarize broad CAD NBIO validation with clean/raw-stress mesh splits."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.abc_cad_electrostatics_benchmark import file_key as abc_file_key
from scripts.abc_cad_electrostatics_benchmark import split_label as abc_split_label
from scripts.cad_overnight_tranche import get_abc_paths, get_fusion_paths
from scripts.fusion_cad_electrostatics_benchmark import mesh_quality, split_label as fusion_split_label
from scripts.mcb_thermal_benchmark import save_json, write_text
from scripts.top5_benchmark_screen import FIGURES, RESULTS, TABLES
from scripts.top5_completion_screen import parse_obj

REPORTS = Path("reports")


def as_float(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def finite_summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def fmt(x: Any) -> str:
    y = as_float(x)
    return "-" if y is None else f"`{y:.3e}`"


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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[write] {path}")


def load_result(tag: str) -> tuple[dict[str, Any], str]:
    final = RESULTS / f"{tag}.json"
    partial = RESULTS / f"{tag}_partial.json"
    if final.exists():
        return json.loads(final.read_text()), "final"
    if partial.exists():
        return json.loads(partial.read_text()), "partial"
    return {"rows": [], "failures": [{"tag": tag, "error": "missing"}], "args": {}}, "missing"


def quality_map(dataset: str, args: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n_meshes = int(args.get("n_meshes", 0) or 0)
    max_file_mb = float(args.get("max_file_mb", 40.0))
    max_faces = int(args.get("max_faces", 160000))
    if dataset == "abc":
        paths = get_abc_paths(n_meshes, max_mb=max_file_mb, max_faces=max_faces)
        splitter = abc_split_label
        key_fn = abc_file_key
    elif dataset == "fusion":
        paths = get_fusion_paths(n_meshes, expanded=True)
        splitter = fusion_split_label
        key_fn = lambda p: p.name
    else:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            vertices, faces = parse_obj(path)
            q = mesh_quality(path, vertices, faces)
            q["split"] = splitter(q)
            out[key_fn(path)] = q
        except Exception as exc:  # noqa: BLE001
            out[key_fn(path)] = {"split": f"{dataset.upper()}-AuditFailed", "error": repr(exc)}
    return out


def read_best_adapters() -> dict[str, dict[str, Any]]:
    path = TABLES / "official_sota_headtohead.csv"
    if not path.exists():
        return {}
    best: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            dataset = str(row.get("dataset", "")).lower()
            med = as_float(row.get("median_rel_l2"))
            if med is None:
                continue
            if dataset not in best or med < float(best[dataset]["median_rel_l2"]):
                best[dataset] = {
                    "model": row.get("model", ""),
                    "median_rel_l2": med,
                    "p90_rel_l2": as_float(row.get("p90_rel_l2")),
                    "provenance": row.get("provenance", ""),
                }
    return best


def summarize(tags: list[str]) -> None:
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    qcache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for tag in tags:
        data, status = load_result(tag)
        args = data.get("args", {})
        rows = data.get("rows", [])
        failures.extend([{**f, "tag": tag, "status": status} for f in data.get("failures", [])])
        status_rows.append({"tag": tag, "status": status, "rows": len(rows), "failures": len(data.get("failures", []))})
        datasets = sorted({str(row.get("dataset", "")) for row in rows})
        for dataset in datasets:
            qcache[(tag, dataset)] = quality_map(dataset, args)
        for row in rows:
            dataset = str(row.get("dataset", ""))
            mesh = str(row.get("mesh", ""))
            q = qcache.get((tag, dataset), {}).get(mesh, {})
            split = q.get("split", f"{dataset.upper()}-Unknown")
            enriched.append(
                {
                    **row,
                    "validation_tag": tag,
                    "validation_status": status,
                    "mesh_split": split,
                    "watertight": q.get("watertight"),
                    "connected_components": q.get("connected_components"),
                    "boundary_edges": q.get("boundary_edges"),
                    "nonmanifold_edges": q.get("nonmanifold_edges"),
                    "tri_quality_p05": q.get("tri_quality_p05"),
                    "sliver_frac": q.get("sliver_frac"),
                }
            )

    summary: list[dict[str, Any]] = []
    group_keys = [
        ["validation_tag", "dataset", "policy"],
        ["validation_tag", "dataset", "mesh_split", "policy"],
        ["validation_tag", "dataset", "family", "policy"],
        ["validation_tag", "dataset", "distance_ratio", "policy"],
    ]
    for keys in group_keys:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in enriched:
            buckets[tuple(row.get(k) for k in keys)].append(row)
        for key, rows in buckets.items():
            vals = [as_float(row.get("potential_rel_l2")) for row in rows]
            vals = [v for v in vals if v is not None]
            out = {k: v for k, v in zip(keys, key)}
            meshes = {row.get("mesh") for row in rows}
            summary.append(
                {
                    "group": "+".join(keys),
                    **out,
                    "n_rows": len(rows),
                    "n_meshes": len(meshes),
                    **finite_summary(vals),
                }
            )

    best_adapters = read_best_adapters()
    comparison: list[dict[str, Any]] = []
    for row in summary:
        if row["group"] != "validation_tag+dataset+policy":
            continue
        dataset = str(row.get("dataset", "")).lower()
        adapter = best_adapters.get(dataset)
        med = as_float(row.get("median"))
        if adapter and med is not None:
            comparison.append(
                {
                    "validation_tag": row["validation_tag"],
                    "dataset": dataset,
                    "policy": row["policy"],
                    "nbio_median": med,
                    "nbio_p90": row.get("p90"),
                    "best_adapter": adapter["model"],
                    "adapter_median": adapter["median_rel_l2"],
                    "adapter_p90": adapter.get("p90_rel_l2"),
                    "median_gap": float(adapter["median_rel_l2"]) / max(med, 1e-30),
                }
            )

    write_csv(TABLES / "cad_broad_sota_raw.csv", enriched)
    write_csv(TABLES / "cad_broad_sota_summary.csv", summary)
    write_csv(TABLES / "cad_broad_sota_comparison.csv", comparison)
    write_csv(TABLES / "cad_broad_sota_status.csv", status_rows)
    save_json(
        {"status": "ok", "run_status": status_rows, "summary": summary, "comparison": comparison, "failures": failures},
        RESULTS / "cad_broad_sota_report.json",
    )
    plot(summary)
    write_report(status_rows, summary, comparison, failures)


def plot(summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in summary if r["group"] == "validation_tag+dataset+mesh_split+policy" and r.get("policy") == "direct_all"]
    if not rows:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = []
    for row in rows:
        split = str(row["mesh_split"]).replace("-", "-\n")
        labels.append(f"{row['dataset']}\n{split}")
    med = [as_float(r.get("median")) or np.nan for r in rows]
    p90 = [as_float(r.get("p90")) or np.nan for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(rows)), 4))
    ax.bar(x - 0.18, med, width=0.36, label="median")
    ax.bar(x + 0.18, p90, width=0.36, label="p90")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("potential rel-L2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "cad_broad_clean_raw_sota.png", dpi=180)
    plt.close(fig)


def best_policy_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in summary if r["group"] == "validation_tag+dataset+policy"]
    best = []
    for tag in sorted({r["validation_tag"] for r in rows}):
        sub = [r for r in rows if r["validation_tag"] == tag]
        if sub:
            best.append(min(sub, key=lambda r: as_float(r.get("median")) or float("inf")))
    return best


def write_report(
    status_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    lines = [
        "# CAD Broad SOTA Validation",
        "",
        "This report extends the CAD exterior-electrostatics claim from the high-resolution 32-object ABC row",
        "to broader Fusion/ABC samples and explicitly separates clean geometry from raw-stress meshes.",
        "",
        "## Run Status",
        "",
        "| tag | status | rows | failures |",
        "|---|---|---:|---:|",
    ]
    for row in status_rows:
        lines.append(f"| {row['tag']} | {row['status']} | {row['rows']} | {row['failures']} |")

    lines.extend(["", "## Broad Headline", "", "| tag | dataset | best policy | rows | meshes | median | p90 | max |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in best_policy_rows(summary):
        lines.append(
            f"| {row['validation_tag']} | {row['dataset']} | {row['policy']} | {row['n_rows']} | {row['n_meshes']} | {fmt(row['median'])} | {fmt(row['p90'])} | {fmt(row['max'])} |"
        )

    split_rows = [r for r in summary if r["group"] == "validation_tag+dataset+mesh_split+policy" and r.get("policy") == "direct_all"]
    lines.extend(["", "## Clean/Raw-Stress Split", "", "| tag | dataset | split | meshes | median | p90 | max |", "|---|---|---|---:|---:|---:|---:|"])
    for row in sorted(split_rows, key=lambda r: (str(r.get("validation_tag")), str(r.get("mesh_split")))):
        lines.append(
            f"| {row['validation_tag']} | {row['dataset']} | {row['mesh_split']} | {row['n_meshes']} | {fmt(row['median'])} | {fmt(row['p90'])} | {fmt(row['max'])} |"
        )

    lines.extend(["", "## Same-Protocol Learned-Adapter Gap", "", "| tag | dataset | policy | NBIO median | best adapter | adapter median | gap |", "|---|---|---|---:|---|---:|---:|"])
    for row in sorted(comparison, key=lambda r: (str(r["validation_tag"]), str(r["dataset"]), float(r["median_gap"]))):
        lines.append(
            f"| {row['validation_tag']} | {row['dataset']} | {row['policy']} | {fmt(row['nbio_median'])} | {row['best_adapter']} | {fmt(row['adapter_median'])} | `{row['median_gap']:.1f}x` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this report to decide whether the CAD SOTA claim is only a 32-object high-resolution result or",
            "holds on broader samples. Clean rows are appropriate for the main benchmark; raw-stress rows should",
            "be reported separately as CAD robustness/geometry-validity stress tests.",
            "",
            f"Total recorded failures: `{len(failures)}`.",
        ]
    )
    write_text(REPORTS / "cad_broad_sota_validation.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()
    summarize(args.tags)


if __name__ == "__main__":
    main()
