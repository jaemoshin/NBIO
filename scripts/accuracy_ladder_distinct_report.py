#!/usr/bin/env python
"""Report high-resolution CAD accuracy ladder with distinct-reference filtering."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.cad_official_qbx_tranche import finite_summary
from scripts.mcb_thermal_benchmark import save_json, write_text
from scripts.top5_benchmark_screen import FIGURES, RESULTS, TABLES

REPORTS = Path("reports")


def as_float(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[write] {path}")


def load_tag(tag: str) -> tuple[list[dict[str, Any]], str]:
    final = RESULTS / f"{tag}.json"
    partial = RESULTS / f"{tag}_partial.json"
    if final.exists():
        data = json.loads(final.read_text())
        status = "final"
    elif partial.exists():
        data = json.loads(partial.read_text())
        status = "partial"
    else:
        return [], "missing"
    rows = data.get("rows", [])
    for row in rows:
        row["tag"] = tag
        row["tag_status"] = status
        row["distinct_reference"] = int(row.get("n_ref", 0)) > int(row.get("n_boundary", 0))
    return rows, status


def run(args: argparse.Namespace) -> None:
    raw: list[dict[str, Any]] = []
    status_rows = []
    for tag in args.tags:
        rows, status = load_tag(tag)
        raw.extend(rows)
        status_rows.append({"tag": tag, "status": status, "rows": len(rows)})

    summary: list[dict[str, Any]] = []
    for tag in sorted({r["tag"] for r in raw}):
        tag_rows = [r for r in raw if r["tag"] == tag]
        for ref_filter, rows in [
            ("distinct_ref_only", [r for r in tag_rows if r["distinct_reference"]]),
            ("all_rows", tag_rows),
            ("same_ref_or_invalid", [r for r in tag_rows if not r["distinct_reference"]]),
        ]:
            for dataset in ["all"] + sorted({str(r["dataset"]) for r in rows}):
                drows = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
                if not drows:
                    continue
                for policy in sorted({str(r["policy"]) for r in drows}):
                    prows = [r for r in drows if r["policy"] == policy]
                    vals = [as_float(r.get("potential_rel_l2")) for r in prows]
                    vals = [v for v in vals if v is not None]
                    summary.append(
                        {
                            "tag": tag,
                            "tag_status": prows[0].get("tag_status", "") if prows else "",
                            "reference_filter": ref_filter,
                            "dataset": dataset,
                            "policy": policy,
                            "n_rows": len(prows),
                            **finite_summary(vals),
                        }
                    )

    write_csv(TABLES / "accuracy_ladder_distinct_raw.csv", raw)
    write_csv(TABLES / "accuracy_ladder_distinct_summary.csv", summary)
    write_csv(TABLES / "accuracy_ladder_distinct_status.csv", status_rows)
    save_json({"status": "ok", "run_status": status_rows, "summary": summary}, RESULTS / "accuracy_ladder_distinct.json")
    plot(summary)
    write_report(status_rows, summary)


def plot(summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in summary
        if r["reference_filter"] == "distinct_ref_only"
        and r["dataset"] == "all"
        and r["policy"] in {"direct_all", "fast_switch_le_0p01", "safe_switch_guard", "robust_refined_all"}
        and r["n_rows"]
    ]
    tags = sorted({r["tag"] for r in rows})
    policies = ["direct_all", "fast_switch_le_0p01", "safe_switch_guard", "robust_refined_all"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(1, len(tags)), figsize=(4 * max(1, len(tags)), 4), sharey=True)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        by = {r["policy"]: r for r in rows if r["tag"] == tag}
        x = np.arange(len(policies))
        med = [as_float(by[p].get("median")) if p in by else np.nan for p in policies]
        p90 = [as_float(by[p].get("p90")) if p in by else np.nan for p in policies]
        ax.bar(x - 0.18, med, width=0.36, label="median")
        ax.bar(x + 0.18, p90, width=0.36, label="p90")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=7)
        ax.set_title(tag.replace("generalizable_switch_validation_acc_ladder_", ""))
    axes[0].set_ylabel("potential rel-L2")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_ladder_distinct.png", dpi=180)
    plt.close(fig)


def best_row(summary: list[dict[str, Any]], tag: str, dataset: str = "all") -> dict[str, Any] | None:
    rows = [
        r
        for r in summary
        if r["tag"] == tag
        and r["reference_filter"] == "distinct_ref_only"
        and r["dataset"] == dataset
        and r["policy"] in {"direct_all", "fast_switch_le_0p01", "safe_switch_guard", "robust_refined_all"}
        and r["n_rows"]
    ]
    if not rows:
        return None
    return min(rows, key=lambda r: (as_float(r.get("median")) or float("inf"), as_float(r.get("p90")) or float("inf")))


def write_report(status_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# CAD Accuracy Ladder With Distinct References",
        "",
        "This report filters headline rows to cases where `n_ref > n_boundary`. Rows where the coarse",
        "and reference discretizations collapse to the same surface are retained in the raw table but not",
        "used for headline accuracy claims.",
        "",
        "## Run Status",
        "",
        "| tag | status | rows |",
        "|---|---|---:|",
    ]
    for r in status_rows:
        lines.append(f"| {r['tag']} | {r['status']} | {r['rows']} |")
    lines.extend(
        [
            "",
            "## Distinct-Reference Headline",
            "",
            "| tag | best policy | n rows | median | p90 | max |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for tag in [r["tag"] for r in status_rows]:
        b = best_row(summary, tag)
        if b:
            lines.append(f"| {tag} | {b['policy']} | {b['n_rows']} | {fmt(b['median'])} | {fmt(b['p90'])} | {fmt(b['max'])} |")
    lines.extend(
        [
            "",
            "## Dataset Breakdown",
            "",
            "| tag | dataset | best policy | median | p90 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for tag in [r["tag"] for r in status_rows]:
        for dataset in ["fusion", "abc"]:
            b = best_row(summary, tag, dataset)
            if b:
                lines.append(f"| {tag} | {dataset} | {b['policy']} | {fmt(b['median'])} | {fmt(b['p90'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The high-resolution ladder tests whether the remaining NBIO CAD error is mostly quadrature/",
            "discretization. The headline should use only `distinct_ref_only` rows. If `2048` has many",
            "same-reference rows on Fusion, use `1536` or `1792` as the cleaner main setting.",
        ]
    )
    write_text(REPORTS / "accuracy_ladder_distinct.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tags",
        nargs="+",
        default=[
            "generalizable_switch_validation_acc_ladder_1536",
            "generalizable_switch_validation_acc_ladder_1792",
            "generalizable_switch_validation_acc_ladder_2048",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
