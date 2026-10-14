#!/usr/bin/env python
"""Summarize broad CAD learned-baseline reruns against the broad NBIO rows.

The official SOTA head-to-head table contains older 24-mesh adapter runs plus
newer broad reruns. This report keeps the broad seed90 checks isolated so the
paper claim is not mixing protocols silently.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
OUT = RESULTS / "official_sota_headtohead"

NBIO_ROWS = {
    "abc": {
        "method": "NBIO direct BIE, broad ABC 3072x96",
        "median": 2.0798790758951797e-03,
        "p90": 1.4605747775314044e-02,
        "n_cases": 2400,
        "source": "results/generalizable_switch_validation_abc_broad_allfam_3072_s90.json",
        "protocol_note": "96 meshes, five voltage families; learned baselines use the four families supported by the shared tensor generator.",
    },
    "fusion": {
        "method": "NBIO fast-switch BIE, broad Fusion 1024x70",
        "median": 5.56257419296818e-03,
        "p90": 1.9986440420328e-02,
        "n_cases": 2000,
        "source": "results/generalizable_switch_validation_fusion_broad_allfam_1024_s88.json",
        "protocol_note": "70 meshes, five voltage families; learned baselines use the four families supported by the shared tensor generator.",
    },
}


def as_float(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def fmt(x: Any) -> str:
    y = as_float(x)
    return "-" if y is None else f"{y:.3e}"


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
        writer.writerows(rows)
    print(f"[write] {path}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


CHECKS = [
    {
        "seed": 90,
        "label": "broad_four_family",
        "families": "constant coordinate random_smooth manufactured_source",
        "note": "Adapter generator before smooth finite-width electrode support.",
    },
    {
        "seed": 91,
        "label": "broad_five_family",
        "families": "constant coordinate random_smooth manufactured_source smooth_patch_wide",
        "note": "Exact family match to the broad NBIO CAD row.",
    },
]


def broad_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    expected = [(dataset, model) for dataset in ["abc", "fusion"] for model in ["ngf", "transolver", "spatial_bias_sdpa"]]
    for check in CHECKS:
      for dataset, model in expected:
        seed = int(check["seed"])
        path = OUT / f"{dataset}_{model}_seed{seed}.json"
        data = load_json(path)
        if not data:
            status = "missing"
            if seed == 91 and dataset == "fusion" and model in {"transolver", "spatial_bias_sdpa"}:
                status = "cancelled_after_hung_no_result"
            pending.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "check": check["label"],
                "status": status,
                "path": str(path.relative_to(ROOT)),
            })
            continue
        if data.get("status") != "ok":
            pending.append({
                "dataset": dataset,
                "model": model,
                "seed": data.get("seed", seed),
                "check": check["label"],
                "status": data.get("status", "failed"),
                "path": str(path.relative_to(ROOT)),
                "error": data.get("error", ""),
            })
            continue
        summary = data.get("test", {}).get("summary", {})
        nbio = NBIO_ROWS[str(dataset)]
        med = as_float(summary.get("median"))
        p90 = as_float(summary.get("p90"))
        rows.append({
            "dataset": dataset,
            "model": model,
            "seed": data.get("seed", seed),
            "check": check["label"],
            "check_note": check["note"],
            "status": "ok",
            "median_rel_l2": med,
            "mean_rel_l2": summary.get("mean"),
            "p90_rel_l2": p90,
            "max_rel_l2": summary.get("max"),
            "n_train_cases": data.get("n_train_cases"),
            "n_test_cases": data.get("n_test_cases"),
            "families": " ".join(data.get("families", [])),
            "params": data.get("params"),
            "wall_s": data.get("wall_s"),
            "provenance": data.get("provenance"),
            "raw_path": str(path.relative_to(ROOT)),
            "nbio_median_rel_l2": nbio["median"],
            "nbio_p90_rel_l2": nbio["p90"],
            "median_gap_vs_nbio": None if med is None else med / max(float(nbio["median"]), 1e-30),
            "p90_gap_vs_nbio": None if p90 is None else p90 / max(float(nbio["p90"]), 1e-30),
        })
    return rows, pending


def best_rows(rows: list[dict[str, Any]], check: str | None = None) -> list[dict[str, Any]]:
    if check is not None:
        rows = [r for r in rows if r.get("check") == check]
    out = []
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        sub = [r for r in rows if r["dataset"] == dataset and as_float(r.get("median_rel_l2")) is not None]
        if sub:
            out.append(min(sub, key=lambda r: float(r["median_rel_l2"])))
    return out


def plot(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_rows = []
    for dataset, nbio in NBIO_ROWS.items():
        plot_rows.append({"label": f"{dataset}\nNBIO", "median": nbio["median"]})
        for row in sorted([r for r in rows if r["dataset"] == dataset and r.get("check") == "broad_five_family"], key=lambda r: str(r["model"])):
            plot_rows.append({"label": f"{dataset}\n{row['model']}", "median": row["median_rel_l2"]})
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(plot_rows)), 4))
    ax.bar(np.arange(len(plot_rows)), [float(r["median"]) for r in plot_rows])
    ax.set_yscale("log")
    ax.set_ylabel("median rel-L2")
    ax.set_xticks(np.arange(len(plot_rows)))
    ax.set_xticklabels([r["label"] for r in plot_rows], rotation=35, ha="right", fontsize=8)
    ax.set_title("Broad CAD NBIO vs learned adapters")
    fig.tight_layout()
    fig.savefig(FIGURES / "cad_broad_baseline_recheck.png", dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], pending: list[dict[str, Any]]) -> None:
    lines = [
        "# CAD Broad Baseline Recheck",
        "",
        "This report isolates the broad `seed90` learned-adapter reruns from the older 24-mesh SOTA table.",
        "The goal is to test whether the NBIO advantage survives broader Fusion/ABC CAD splits.",
        "",
        "## Five-Family Headline",
        "",
        "| dataset | NBIO median | best broad adapter | adapter median | median gap | adapter p90 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in best_rows(rows, "broad_five_family"):
        nbio = NBIO_ROWS[str(row["dataset"])]
        lines.append(
            f"| {row['dataset']} | `{fmt(nbio['median'])}` | {row['model']} | "
            f"`{fmt(row['median_rel_l2'])}` | `{fmt(row['median_gap_vs_nbio'])}x` | `{fmt(row['p90_rel_l2'])}` |"
        )
    lines.extend([
        "",
        "## Four-Family Headline",
        "",
        "| dataset | NBIO median | best broad adapter | adapter median | median gap | adapter p90 |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for row in best_rows(rows, "broad_four_family"):
        nbio = NBIO_ROWS[str(row["dataset"])]
        lines.append(
            f"| {row['dataset']} | `{fmt(nbio['median'])}` | {row['model']} | "
            f"`{fmt(row['median_rel_l2'])}` | `{fmt(row['median_gap_vs_nbio'])}x` | `{fmt(row['p90_rel_l2'])}` |"
        )
    lines.extend([
        "",
        "## Completed Broad Adapter Rows",
        "",
        "| check | dataset | model | median | p90 | n_test | families | params | gap vs NBIO |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ])
    for row in sorted(rows, key=lambda r: (str(r["check"]), str(r["dataset"]), str(r["model"]))):
        lines.append(
            f"| {row['check']} | {row['dataset']} | {row['model']} | `{fmt(row['median_rel_l2'])}` | `{fmt(row['p90_rel_l2'])}` | "
            f"{row['n_test_cases']} | {row['families']} | {row['params']} | `{fmt(row['median_gap_vs_nbio'])}x` |"
        )
    lines.extend(["", "## Pending / Failed Rows", "", "| dataset | model | status | path |", "|---|---|---|---|"])
    for row in pending:
        lines.append(f"| {row['dataset']} | {row['model']} ({row.get('check')}) | {row['status']} | `{row['path']}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The five-family rows match the broad NBIO CAD voltage-family contract, including smooth finite-width",
        "electrodes. Fusion Transolver and spatial-bias SDPA initially hung in an unbounded run, then completed",
        "under a bounded replacement rerun; those completed seed91 rows are the five-family comparison rows.",
    ])
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "cad_broad_baseline_recheck.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def main() -> None:
    rows, pending = broad_rows()
    write_csv(TABLES / "cad_broad_baseline_recheck.csv", rows)
    (RESULTS / "cad_broad_baseline_recheck.json").write_text(json.dumps({"status": "ok", "rows": rows, "pending": pending}, indent=2))
    print(f"[write] {RESULTS / 'cad_broad_baseline_recheck.json'}")
    plot(rows)
    write_report(rows, pending)


if __name__ == "__main__":
    main()
