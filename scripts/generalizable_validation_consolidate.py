#!/usr/bin/env python
"""Consolidate generalizable switch-policy validation runs.

Reads final JSONs when available and falls back to partial JSONs for in-flight
or timed-out jobs.  Partial rows are labeled explicitly in the output.
"""
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


def load_tag(tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    final = RESULTS / f"{tag}.json"
    partial = RESULTS / f"{tag}_partial.json"
    if final.exists():
        data = json.loads(final.read_text())
        status = "final"
    elif partial.exists():
        data = json.loads(partial.read_text())
        status = "partial"
    else:
        return [], [{"tag": tag, "error": "missing"}], "missing"
    rows = data.get("rows", [])
    failures = data.get("failures", [])
    for r in rows:
        r["validation_tag"] = tag
        r["validation_status"] = status
    for f in failures:
        f["validation_tag"] = tag
        f["validation_status"] = status
    return rows, failures, status


def consolidate(tags: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for tag in tags:
        r, f, status = load_tag(tag)
        rows.extend(r)
        failures.extend(f)
        status_rows.append({"tag": tag, "status": status, "n_rows": len(r), "n_failures": len(f)})

    summary: list[dict[str, Any]] = []
    for tag in sorted({str(r.get("validation_tag")) for r in rows}):
        tag_rows = [r for r in rows if r.get("validation_tag") == tag]
        for dataset in ["all"] + sorted({str(r["dataset"]) for r in tag_rows}):
            drows = tag_rows if dataset == "all" else [r for r in tag_rows if r["dataset"] == dataset]
            for policy in sorted({str(r["policy"]) for r in drows}):
                prows = [r for r in drows if r["policy"] == policy]
                for metric in ["potential_rel_l2", "field_rel_l2", "eval_runtime_ms"]:
                    vals = [as_float(r.get(metric)) for r in prows]
                    vals = [v for v in vals if v is not None]
                    summary.append(
                        {
                            "validation_tag": tag,
                            "validation_status": prows[0].get("validation_status") if prows else "",
                            "dataset": dataset,
                            "policy": policy,
                            "metric": metric,
                            **finite_summary(vals),
                        }
                    )

    write_csv(TABLES / "generalizable_validation_consolidated_raw.csv", rows)
    write_csv(TABLES / "generalizable_validation_consolidated_summary.csv", summary)
    write_csv(TABLES / "generalizable_validation_consolidated_status.csv", status_rows)
    save_json(
        {"status": "ok", "run_status": status_rows, "summary": summary, "failures": failures},
        RESULTS / "generalizable_validation_consolidated.json",
    )
    plot(summary)
    write_report(status_rows, summary, failures)


def plot(summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    policy_order = ["direct_all", "fast_switch_le_0p01", "safe_switch_guard", "robust_refined_all"]
    rows = [
        r
        for r in summary
        if r["dataset"] == "all" and r["metric"] == "potential_rel_l2" and r["policy"] in set(policy_order)
    ]
    tags = sorted({r["validation_tag"] for r in rows})
    policies = [p for p in policy_order if any(r["policy"] == p for r in rows)]
    fig, axes = plt.subplots(1, max(1, len(tags)), figsize=(4 * max(1, len(tags)), 4), sharey=True)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        tr = {r["policy"]: r for r in rows if r["validation_tag"] == tag}
        vals = [as_float(tr[p].get("median")) if p in tr else np.nan for p in policies]
        p90 = [as_float(tr[p].get("p90")) if p in tr else np.nan for p in policies]
        x = np.arange(len(policies))
        ax.bar(x - 0.18, vals, width=0.36, label="median")
        ax.bar(x + 0.18, p90, width=0.36, label="p90")
        ax.set_title(tag.replace("generalizable_switch_validation_", ""))
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=7)
        ax.set_yscale("log")
    axes[0].set_ylabel("potential rel-L2")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "generalizable_validation_consolidated.png", dpi=180)
    plt.close(fig)


def write_report(status_rows: list[dict[str, Any]], summary: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Generalizable CAD Validation Consolidated",
        "",
        "This report consolidates final validation runs when present and explicitly labels partial",
        "in-flight/timed-out rows when only partial JSONs exist.",
        "",
        "## Run Status",
        "",
        "| tag | status | rows | failures |",
        "|---|---|---:|---:|",
    ]
    for r in status_rows:
        lines.append(f"| {r['tag']} | {r['status']} | {r['n_rows']} | {r['n_failures']} |")
    lines.extend(
        [
            "",
            "## Pooled Potential Results",
            "",
            "| tag | status | policy | median | p90 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for r in summary:
        if r["dataset"] == "all" and r["metric"] == "potential_rel_l2":
            lines.append(
                f"| {r['validation_tag']} | {r['validation_status']} | {r['policy']} | {fmt(r.get('median'))} | {fmt(r.get('p90'))} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The default performance target is the best validated low-overhead policy. `safe_switch_guard` is",
            "designed to keep the fast-switch median behavior while suppressing rare pathological direct-eval",
            "outliers. `robust_refined_all` is the accuracy mode when field accuracy matters more than raw speed.",
            "",
            f"Total failures recorded: `{len(failures)}`.",
        ]
    )
    write_text(REPORTS / "generalizable_validation_consolidated.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tags",
        nargs="+",
        default=[
            "generalizable_switch_validation_wide",
            "generalizable_switch_validation_hires",
            "generalizable_switch_validation_oodbc",
        ],
    )
    args = parser.parse_args()
    consolidate(args.tags)


if __name__ == "__main__":
    main()
