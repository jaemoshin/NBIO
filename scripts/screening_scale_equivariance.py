#!/usr/bin/env python
"""Post-confirmation scale-equivariance test for the radial screening normal form."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from screening_query_resolution_phase import fourier_solution  # noqa: E402
from src.data.bie import boundary, bem_solve, eval_u, radius_at, sample_charges, sample_queries  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402

RESULT = ROOT / "results" / "screening_law" / "scale_equivariance_v1"
REPORT = ROOT / "reports" / "screening_law_scale_equivariance_v1.md"
FIGURE = ROOT / "figures" / "screening_law_scale_equivariance_v1.png"
SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
BASE_KAPPA = 2.0
N_BOUNDARY = 128
N_GEOMETRIES = 16
FAMILY_SEEDS = {"in_dist": 98001, "rough": 98002}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(device: torch.device):
    manifest = json.loads(
        (ROOT / "results" / "screening_law" / "confirmatory_v1" / "checkpoint_manifest.json").read_text()
    )
    output = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        output.append((record["method"], record["seed_index"], model.to(device).eval()))
    return output


def geometries(family: str) -> list[dict]:
    rng = np.random.default_rng(FAMILY_SEEDS[family])
    settings = FAMILIES[family]
    fine = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    output = []
    while len(output) < N_GEOMETRIES:
        coefficients = ood_coeffs(rng, settings)
        if radius_at(coefficients, fine).min() <= 0.2:
            continue
        curve = boundary(coefficients, N_BOUNDARY)
        output.append({
            "curve": curve,
            "query": sample_queries(rng, coefficients, "interior", 256, margin=0.12),
            "charges": sample_charges(rng, "interior", coefficients, n_charges=6),
            "fourier": rng.normal(size=(6, 2)) / np.arange(1, 7)[:, None],
        })
    return output


def transformed_curve(curve: dict, scale: float) -> dict:
    return {
        "X": scale * curve["X"],
        "normal": curve["normal"],
        "w": scale * curve["w"],
        "curv": curve["curv"] / scale,
        "theta": curve["theta"],
    }


def rel_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=-1) / np.maximum(
        np.linalg.norm(target, axis=-1), 1e-12,
    )


@torch.no_grad()
def main() -> None:
    raw_path = RESULT / "raw.csv"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    models = load_models(device)
    rows = []
    for family in FAMILY_SEEDS:
        items = geometries(family)
        for load in ("point_source", "fourier"):
            base_target = np.asarray([
                eval_u(item["query"], item["charges"], BASE_KAPPA)
                if load == "point_source" else
                fourier_solution(item["query"], BASE_KAPPA, item["fourier"])
                for item in items
            ])
            base_trace = np.asarray([
                eval_u(item["curve"]["X"], item["charges"], BASE_KAPPA)
                if load == "point_source" else
                fourier_solution(item["curve"]["X"], BASE_KAPPA, item["fourier"])
                for item in items
            ])
            predictions = defaultdict(dict)
            for scale in SCALES:
                kappa = BASE_KAPPA / scale
                curves = [transformed_curve(item["curve"], scale) for item in items]
                query = np.asarray([scale * item["query"] for item in items])
                exact = []
                laplace = []
                for curve, trace, query_points in zip(curves, base_trace, query):
                    _, exact_eval = bem_solve(curve, trace, "interior", kappa=kappa)
                    _, laplace_eval = bem_solve(curve, trace, "interior", kappa=0.0)
                    exact.append(exact_eval(query_points))
                    laplace.append(laplace_eval(query_points))
                predictions[("correct_yukawa_bie", -1)][scale] = np.asarray(exact)
                predictions[("fixed_laplace_bie", -1)][scale] = np.asarray(laplace)

                batch = {
                    "Xb": torch.tensor(np.asarray([curve["X"] for curve in curves]), dtype=torch.float32,
                                       device=device),
                    "Nb": torch.tensor(np.asarray([curve["normal"] for curve in curves]), dtype=torch.float32,
                                       device=device),
                    "wb": torch.tensor(np.asarray([curve["w"] for curve in curves]), dtype=torch.float32,
                                       device=device),
                    "curv": torch.tensor(np.asarray([curve["curv"] for curve in curves]), dtype=torch.float32,
                                         device=device),
                    "g": torch.tensor(base_trace, dtype=torch.float32, device=device),
                    "Xq": torch.tensor(query, dtype=torch.float32, device=device),
                    "kappa": torch.full((N_GEOMETRIES,), kappa, device=device),
                }
                for method, seed, model in models:
                    predictions[(method, seed)][scale] = model(batch).cpu().numpy()
                print(f"[{family}] load={load} scale={scale:g} kappa={kappa:g}", flush=True)

            for (method, seed), by_scale in predictions.items():
                reference = by_scale[1.0]
                for scale, prediction in by_scale.items():
                    errors = rel_error(prediction, base_target)
                    drifts = rel_error(prediction, reference)
                    for sample_id, (error, drift) in enumerate(zip(errors, drifts)):
                        rows.append({
                            "method": method, "seed_index": seed, "family": family,
                            "load_family": load, "scale": scale, "kappa": BASE_KAPPA / scale,
                            "sample_id": sample_id, "rel_l2": float(error),
                            "equivariance_drift": float(drift),
                            "finite": bool(np.isfinite(error) and np.isfinite(drift)),
                        })
            predictions.clear()

    expected = 2 * 2 * 5 * 16 * 14
    if len(rows) != expected or not all(row["finite"] for row in rows):
        raise RuntimeError(f"coverage/finite failure: {len(rows)} != {expected}")
    RESULT.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    drift_grouped = defaultdict(list)
    for row in rows:
        key = (row["method"], row["scale"])
        grouped[key].append(row["rel_l2"])
        drift_grouped[key].append(row["equivariance_drift"])
    summary = []
    for key in sorted(grouped):
        summary.append({
            "method": key[0], "scale": key[1], "kappa": BASE_KAPPA / key[1],
            "mean_rel_l2": float(np.mean(grouped[key])),
            "median_rel_l2": float(np.median(grouped[key])),
            "mean_equivariance_drift": float(np.mean(drift_grouped[key])),
            "max_equivariance_drift": float(np.max(drift_grouped[key])),
        })
    with (RESULT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    pooled = {(row["method"], row["scale"]): row for row in summary}
    report = {
        "n_rows": len(rows), "base_kappa": BASE_KAPPA, "scales": SCALES,
        "radial": [pooled[("radial_mlp", scale)] for scale in SCALES],
        "full_prior": [pooled[("full_prior", scale)] for scale in SCALES],
        "no_prior": [pooled[("no_prior", scale)] for scale in SCALES],
        "exact_bie": [pooled[("correct_yukawa_bie", scale)] for scale in SCALES],
    }
    (RESULT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    colors = {"correct_yukawa_bie": "black", "radial_mlp": "#D55E00",
              "full_prior": "#0072B2", "no_prior": "#009E73"}
    for method, color in colors.items():
        errors = [pooled[(method, scale)]["mean_rel_l2"] for scale in SCALES]
        drifts = [pooled[(method, scale)]["mean_equivariance_drift"] for scale in SCALES]
        axes[0].plot(SCALES, errors, marker="o", color=color, label=method)
        axes[1].plot(SCALES, drifts, marker="o", color=color, label=method)
    for axis in axes:
        axis.set_yscale("log"); axis.set_xlabel("geometry scale a")
    axes[0].set_ylabel("mean rel-L2")
    axes[1].set_ylabel("equivariance drift vs a=1")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220); plt.close(fig)

    radial_max = max(pooled[("radial_mlp", scale)]["mean_equivariance_drift"] for scale in SCALES)
    full_max = max(pooled[("full_prior", scale)]["mean_equivariance_drift"] for scale in SCALES)
    none_max = max(pooled[("no_prior", scale)]["mean_equivariance_drift"] for scale in SCALES)
    REPORT.write_text(f"""# Screening-law scale-equivariance test

Under `x'=a x` and `kappa'=kappa/a`, the dimensionless separation `kappa*r` is invariant. Frozen models
trained near unit scale were evaluated on fresh smooth/rough geometries at scales `{SCALES}` with no
retraining.

Maximum mean prediction drift relative to scale one:

- radial normal form: `{radial_max:.4e}`;
- generic full prior: `{full_max:.4e}`;
- generic no prior: `{none_max:.4e}`.

The exact BIE and radial normal form should be equivariant up to numerical precision. Generic pairwise
kernels receive dimensional coordinate and parameter features and are not constrained to obey this law.
This is a post-confirmation structural diagnostic, not a model-selection gate.
""")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
