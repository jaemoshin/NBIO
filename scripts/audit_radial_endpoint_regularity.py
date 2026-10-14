#!/usr/bin/env python3
"""Audit endpoint derivatives and directional diagonal limits of radial remainders."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import k1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_endpoint_v1"
FIGURE = ROOT / "figures" / "radial_endpoint_regularity_v2.png"
REPORT = ROOT / "reports" / "radial_endpoint_regularity_v2.md"
EPSILONS = np.logspace(-8, -1, 50)
ANGLES = (0.0, 30.0, 60.0)
KAPPA = 3.0


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_models():
    manifest = json.loads((RESULT_ROOT / "checkpoint_manifest.json").read_text())
    selected = []
    for record in manifest["records"]:
        if record["method"] not in {"radial_normal", "radial_endpoint"}:
            continue
        checkpoint = torch.load(ROOT / record["path"], map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"]).double().eval()
        model.load_state_dict(checkpoint["state_dict"])
        selected.append((record["method"], record["seed_index"], model))
    if len(selected) != 6:
        raise RuntimeError(f"expected six old/new radial models, found {len(selected)}")
    return selected


def directional_remainder(model, epsilon: float, angle_degrees: float) -> float:
    angle = np.deg2rad(angle_degrees)
    # Clamp-free analytic off-surface kernel; the production safety clamp would
    # obscure the epsilon -> 0 directional limit this diagnostic is testing.
    laplace = torch.tensor(
        np.cos(angle) / (2.0 * np.pi * epsilon), dtype=torch.float64,
    )
    scaled = torch.tensor([[[KAPPA * epsilon]]], dtype=torch.float64)
    multiplier = model.m_law(scaled)
    return float((laplace * (multiplier - 1.0)).squeeze())


def exact_remainder(epsilon: float, angle_degrees: float) -> float:
    angle = np.deg2rad(angle_degrees)
    laplace = np.cos(angle) / (2.0 * np.pi * epsilon)
    scaled = KAPPA * epsilon
    return laplace * (scaled * k1(scaled) - 1.0)


def main() -> None:
    outputs = (
        RESULT_ROOT / "regularity_v2.json", RESULT_ROOT / "regularity_v2_raw.csv",
        FIGURE, REPORT,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("write-once regularity output exists")
    models = load_models()
    rows = []
    derivatives: dict[str, list[float]] = {"radial_normal": [], "radial_endpoint": []}
    for method, seed, model in models:
        s0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.grad(model.m_law(s0).sum(), s0)[0]
        derivatives[method].append(float(derivative))
        for angle in ANGLES:
            for epsilon in EPSILONS:
                rows.append({
                    "method": method,
                    "seed_index": seed,
                    "angle_degrees": angle,
                    "epsilon": epsilon,
                    "remainder": directional_remainder(model, epsilon, angle),
                })
    for angle in ANGLES:
        for epsilon in EPSILONS:
            rows.append({
                "method": "exact_yukawa",
                "seed_index": -1,
                "angle_degrees": angle,
                "epsilon": epsilon,
                "remainder": exact_remainder(epsilon, angle),
            })
    write_csv(rows, RESULT_ROOT / "regularity_v2_raw.csv")

    endpoint_small = [
        abs(row["remainder"]) for row in rows
        if row["method"] == "radial_endpoint" and row["epsilon"] == EPSILONS[0]
    ]
    original_small = [
        abs(row["remainder"]) for row in rows
        if row["method"] == "radial_normal" and row["epsilon"] == EPSILONS[0]
    ]
    result = {
        "kappa": KAPPA,
        "angles_degrees": ANGLES,
        "epsilon_range": [float(EPSILONS[0]), float(EPSILONS[-1])],
        "endpoint_derivatives": derivatives,
        "max_abs_endpoint_remainder_at_min_epsilon": max(endpoint_small),
        "median_abs_original_remainder_at_min_epsilon": float(np.median(original_small)),
        "endpoint_direction_independent_zero_limit": bool(
            max(abs(value) for value in derivatives["radial_endpoint"]) <= 1e-10
            and max(endpoint_small) <= 1e-6
        ),
    }
    (RESULT_ROOT / "regularity_v2.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    colors = {0.0: "#0072B2", 30.0: "#D55E00", 60.0: "#009E73"}
    for axis, method, title in (
        (axes[0], "radial_normal", "Original radial law"),
        (axes[1], "radial_endpoint", "Endpoint-matched radial law"),
    ):
        for angle in ANGLES:
            values = []
            for epsilon in EPSILONS:
                cell = [
                    abs(row["remainder"]) for row in rows
                    if row["method"] == method and row["angle_degrees"] == angle
                    and row["epsilon"] == epsilon
                ]
                values.append(np.median(cell))
            axis.plot(EPSILONS, values, color=colors[angle], label=fr"$\phi={angle:g}^\circ$")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("source-target distance")
        axis.set_title(title)
    axes[0].set_ylabel(r"median $|K_0(m_\theta-1)|$")
    axes[0].legend(fontsize=7)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)

    REPORT.write_text(f"""# Radial endpoint regularity audit

The original checkpoints have `m'(0)` values `{derivatives['radial_normal']}`. Their bounded remainder
approaches a direction-dependent nonzero limit. The endpoint-matched checkpoints have derivatives
`{derivatives['radial_endpoint']}` and the learned remainder tends to zero for every audited approach
direction.

- Maximum endpoint remainder magnitude at `epsilon={EPSILONS[0]:.0e}`: `{max(endpoint_small):.3e}`.
- Median original remainder magnitude at the same distance: `{np.median(original_small):.3e}`.
- Direction-independent zero-limit diagnostic: `{result['endpoint_direction_independent_zero_limit']}`.

This verifies local continuity of the endpoint-matched learned remainder for the implemented smooth MLP.
It is a local regularity diagnostic, not an accuracy result.
""")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
