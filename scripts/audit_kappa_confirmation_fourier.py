#!/usr/bin/env python
"""Aggregate and report the frozen-checkpoint Fourier boundary-load audit."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "kappa_confirmation_fourier"
REPORT = ROOT / "reports" / "kappa_confirmation_fourier_results.md"


def main():
    rows = list(csv.DictReader((RESULT / "raw.csv").open()))
    model_rows = [row for row in rows if row["method"] in {"full_norm", "none_norm"}]
    clustered = defaultdict(list)
    for row in model_rows:
        clustered[(row["method"], int(row["sample_id"]))].append(float(row["rel_l2"]))
    full = np.asarray([np.mean(clustered[("full_norm", i)]) for i in range(96)])
    none = np.asarray([np.mean(clustered[("none_norm", i)]) for i in range(96)])
    difference = full - none
    rng = np.random.default_rng(42)
    boot = np.asarray([
        difference[rng.integers(0, len(difference), len(difference))].mean()
        for _ in range(10_000)
    ])
    paired = {
        "n_geometry_clusters": len(difference),
        "mean_difference": float(difference.mean()),
        "bootstrap_mean_difference_ci95": [
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        ],
        "full_win_fraction": float((difference < 0).mean()),
    }

    per_kappa = []
    kappas = sorted({float(row["kappa"]) for row in rows})
    for kappa in kappas:
        item = {"kappa": kappa}
        for method in ("correct_bie_nb128", "fixed_laplace_bie", "full_norm", "none_norm"):
            values = [
                float(row["rel_l2"]) for row in rows
                if row["method"] == method and float(row["kappa"]) == kappa
            ]
            item[method] = float(np.mean(values))
        item["full_vs_none_improvement"] = (
            1.0 - item["full_norm"] / item["none_norm"]
        )
        per_kappa.append(item)

    audit_path = RESULT / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["paired_geometry_audit"] = paired
    audit["per_kappa"] = per_kappa
    audit["verdict"] = "NARROW"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    lines = [
        "# Frozen continuous-kappa models under Fourier boundary-load shift",
        "",
        "## Protocol",
        "",
        "The six continuous-kappa checkpoints and confirmation geometries were frozen before this audit. "
        "The manufactured point-source boundary traces were replaced by six-mode Fourier traces that are "
        "independent of `kappa`. Models use 128 boundary nodes; the reference is a correct-kernel 256-node "
        "Nystrom BEM on the same geometry and queries. No retraining or model selection was performed.",
        "",
        "## Results",
        "",
        "| kappa | BEM 128 vs 256 | fixed Laplace BIE | full prior | no prior | full improvement |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in per_kappa:
        lines.append(
            f"| `{item['kappa']:g}` | `{item['correct_bie_nb128']:.3e}` | "
            f"`{item['fixed_laplace_bie']:.3e}` | `{item['full_norm']:.3e}` | "
            f"`{item['none_norm']:.3e}` | `{100 * item['full_vs_none_improvement']:.1f}%` |"
        )
    agg = audit["aggregate"]
    lines.extend([
        "",
        "Across the eight in-range parameters, the full prior has mean error "
        f"`{agg['full_norm']['mean_rel_l2']:.3e}` versus `{agg['none_norm']['mean_rel_l2']:.3e}` "
        f"for no-prior, a `{100 * audit['full_vs_none_relative_improvement']:.1f}%` reduction. "
        f"The paired geometry-cluster mean-difference 95% bootstrap interval is "
        f"`[{paired['bootstrap_mean_difference_ci95'][0]:.3e}, "
        f"{paired['bootstrap_mean_difference_ci95'][1]:.3e}]`, and the full prior wins on "
        f"`{100 * paired['full_win_fraction']:.1f}%` of geometry clusters.",
        "",
        "The full prior loses to no-prior at `kappa=3.8`, so this remains a mismatch-dependent result rather "
        "than uniform dominance. Relative to the original point-source-load confirmation, absolute error "
        "increases for both learned models; the Fourier test therefore confirms a boundary-load OOD gap even "
        "though the prior comparison remains favorable.",
        "",
        "## Verdict",
        "",
        "`NARROW`: the frozen prior result survives an independent boundary-load family, but the correction "
        "does not reach the BEM ceiling and the prior should be routed away at sufficiently high mismatch.",
    ])
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"paired": paired, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
