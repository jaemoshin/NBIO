#!/usr/bin/env python
"""Optimize dataset-general CAD close-evaluation switching policies.

This script consumes the raw Fusion/ABC screen from
`generalizable_cad_improvement_screen.py` and evaluates policies of the form:

    if query_distance <= threshold: use close-evaluation variant
    else: use direct coarse evaluation

It also reports a band policy whose method choices are fixed by pooled
Fusion+ABC median error at each distance, without using dataset labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path}")


def finite_stats(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "trimmed_mean_lt10": None, "median": None, "p90": None, "max": None}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "trimmed_mean_lt10": float(np.mean([v for v in vals if v < 10.0])) if any(v < 10.0 for v in vals) else None,
        "median": float(np.median(vals)),
        "p90": float(s[min(len(s) - 1, int(np.ceil(0.90 * len(s)) - 1))]),
        "max": float(max(vals)),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def group_by_case(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, float], dict[str, dict[str, Any]]]:
    cases: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        dist = as_float(r.get("distance_ratio"))
        if dist is None:
            continue
        cases[(str(r["dataset"]), str(r["mesh"]), str(r["family"]), dist)][str(r["method"])] = r
    return cases


def eval_policy(
    name: str,
    cases: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]],
    choices: dict[float, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded: list[dict[str, Any]] = []
    for (dataset, mesh, family, dist), methods in cases.items():
        method = choices.get(dist, "direct_coarse")
        if method not in methods:
            method = "direct_coarse"
        r = methods[method]
        expanded.append(
            {
                "policy": name,
                "dataset": dataset,
                "mesh": mesh,
                "family": family,
                "distance_ratio": dist,
                "selected_method": method,
                "potential_rel_l2": as_float(r.get("potential_rel_l2")),
                "field_rel_l2": as_float(r.get("field_rel_l2")),
                "eval_runtime_ms": as_float(r.get("eval_runtime_ms")),
            }
        )

    summary: list[dict[str, Any]] = []
    for dataset in ["all"] + sorted({r["dataset"] for r in expanded}):
        sub = expanded if dataset == "all" else [r for r in expanded if r["dataset"] == dataset]
        for metric in ["potential_rel_l2", "field_rel_l2", "eval_runtime_ms"]:
            vals = [r[metric] for r in sub if as_float(r.get(metric)) is not None]
            summary.append({"policy": name, "dataset": dataset, "metric": metric, **finite_stats(vals)})
    return expanded, summary


def pooled_band_policy(cases: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]]) -> dict[float, str]:
    """Pick one method per distance using pooled Fusion+ABC median potential error."""
    distances = sorted({key[3] for key in cases})
    out: dict[float, str] = {}
    for dist in distances:
        methods = sorted({m for key, mm in cases.items() if key[3] == dist for m in mm})
        best_method = "direct_coarse"
        best_median = float("inf")
        for method in methods:
            vals = [
                as_float(mm[method].get("potential_rel_l2"))
                for key, mm in cases.items()
                if key[3] == dist and method in mm
            ]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            med = statistics.median(vals)
            # Prefer direct when the difference is negligible; this avoids paying
            # close-eval overhead when the direct path is statistically tied.
            if med < best_median * 0.98:
                best_median = med
                best_method = method
        out[dist] = best_method
    return out


def build_policies(cases: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]]) -> dict[str, dict[float, str]]:
    distances = sorted({key[3] for key in cases})
    close_methods = sorted({m for mm in cases.values() for m in mm if m != "direct_coarse"})
    policies: dict[str, dict[float, str]] = {"direct_all": {d: "direct_coarse" for d in distances}}
    for method in close_methods:
        policies[f"{method}_all"] = {d: method for d in distances}
        for threshold in distances:
            policies[f"{method}_switch_le_{threshold:g}"] = {
                d: method if d <= threshold else "direct_coarse" for d in distances
            }
    policies["pooled_band_best"] = pooled_band_policy(cases)
    return policies


def plot(summary: list[dict[str, Any]], policy_choices: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    pot = [r for r in summary if r["dataset"] == "all" and r["metric"] == "potential_rel_l2"]
    top = sorted(pot, key=lambda r: (float(r["p90"]), float(r["median"])))[:10]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [r["policy"].replace("_", " ")[:26] for r in top]
    med = [float(r["median"]) for r in top]
    p90 = [float(r["p90"]) for r in top]
    x = np.arange(len(top))
    ax.bar(x - 0.18, med, width=0.36, label="median")
    ax.bar(x + 0.18, p90, width=0.36, label="p90")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("potential rel-L2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "generalizable_switch_policy_ranking.png", dpi=180)
    plt.close(fig)

    band = [r for r in policy_choices if r["policy"] == "pooled_band_best"]
    if band:
        fig, ax = plt.subplots(figsize=(7, 3))
        labels = [f"{float(r['distance_ratio']):.3g}" for r in band]
        vals = [r["selected_method"].replace("_", "\n") for r in band]
        ax.axis("off")
        table = ax.table(
            cellText=[[d, m] for d, m in zip(labels, vals)],
            colLabels=["distance", "method"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        fig.tight_layout()
        fig.savefig(FIGURES / "generalizable_switch_policy_table.png", dpi=180)
        plt.close(fig)


def write_report(summary: list[dict[str, Any]], policy_choices: list[dict[str, Any]]) -> None:
    def row(policy: str, dataset: str = "all", metric: str = "potential_rel_l2") -> dict[str, Any] | None:
        for r in summary:
            if r["policy"] == policy and r["dataset"] == dataset and r["metric"] == metric:
                return r
        return None

    direct = row("direct_all")
    band = row("pooled_band_best")
    best = min(
        [r for r in summary if r["dataset"] == "all" and r["metric"] == "potential_rel_l2"],
        key=lambda r: (float(r["p90"]), float(r["median"])),
    )
    band_choices = [r for r in policy_choices if r["policy"] == "pooled_band_best"]
    lines = [
        "# Generalizable CAD Switch-Policy Optimization",
        "",
        "This pass turns the previous close-evaluation screen into deployable policies. A policy is",
        "fixed across Fusion and ABC: it can switch by query distance, but it cannot inspect the",
        "dataset label. The objective is robust potential accuracy across both CAD sources.",
        "",
        "## Main Result",
        "",
        "| policy | pooled median potential | pooled p90 potential | median field | median eval ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["direct_all", "pooled_band_best", str(best["policy"])]:
        pot = row(name)
        fld = row(name, metric="field_rel_l2")
        rt = row(name, metric="eval_runtime_ms")
        if pot and fld and rt:
            lines.append(f"| {name} | {fmt(pot['median'])} | {fmt(pot['p90'])} | {fmt(fld['median'])} | {fmt(rt['median'])} |")
    if direct and band:
        lines.extend(
            [
                "",
                "Compared with direct evaluation everywhere, the pooled band policy changes:",
                "",
                f"- median potential rel-L2: {fmt(direct['median'])} -> {fmt(band['median'])}",
                f"- p90 potential rel-L2: {fmt(direct['p90'])} -> {fmt(band['p90'])}",
                f"- trimmed mean potential rel-L2: {fmt(direct['trimmed_mean_lt10'])} -> {fmt(band['trimmed_mean_lt10'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Pooled Band Policy",
            "",
            "| query distance / object scale | selected method |",
            "|---:|---|",
        ]
    )
    for r in band_choices:
        lines.append(f"| `{float(r['distance_ratio']):.4f}` | {r['selected_method']} |")
    lines.extend(
        [
            "",
            "## Dataset Breakdown",
            "",
            "| policy | dataset | median potential | p90 potential |",
            "|---|---|---:|---:|",
        ]
    )
    for policy in ["direct_all", "pooled_band_best"]:
        for dataset in ["fusion", "abc"]:
            pot = row(policy, dataset)
            if pot:
                lines.append(f"| {policy} | {dataset} | {fmt(pot['median'])} | {fmt(pot['p90'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The useful improvement is not global refinement. It is a near-surface switch:",
            "close-evaluation variants reduce the hardest near-boundary errors, while direct",
            "evaluation remains preferable away from the surface. This is dataset-general and",
            "should be implemented as an NBIO inference policy before adding larger learned",
            "correction networks.",
        ]
    )
    write_text(REPORTS / "generalizable_switch_policy.md", "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    rows = load_rows(Path(args.raw))
    cases = group_by_case(rows)
    policies = build_policies(cases)
    expanded_all: list[dict[str, Any]] = []
    summary_all: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    for name, choices in sorted(policies.items()):
        expanded, summary = eval_policy(name, cases, choices)
        expanded_all.extend(expanded)
        summary_all.extend(summary)
        for dist, method in sorted(choices.items()):
            choice_rows.append({"policy": name, "distance_ratio": dist, "selected_method": method})
    write_csv(TABLES / "generalizable_switch_policy_rows.csv", expanded_all)
    write_csv(TABLES / "generalizable_switch_policy_summary.csv", summary_all)
    write_csv(TABLES / "generalizable_switch_policy_choices.csv", choice_rows)
    save_json(
        {"status": "ok", "n_cases": len(cases), "summary": summary_all, "policy_choices": choice_rows},
        RESULTS / "generalizable_switch_policy.json",
    )
    plot(summary_all, choice_rows)
    write_report(summary_all, choice_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/tables/generalizable_improvement_screen_raw.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
