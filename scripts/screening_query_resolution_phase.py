#!/usr/bin/env python
"""Post-confirmation query-distance by resolution map for screening-law NBIO."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import iv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.bie import boundary, bem_solve, eval_u, radius_at, sample_charges  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402

RESULT = ROOT / "results" / "screening_law" / "query_resolution_v1"
FIGURE = ROOT / "figures" / "screening_law_query_resolution_v1.png"
REPORT = ROOT / "reports" / "screening_law_query_resolution_v1.md"
KAPPAS = (0.45, 2.55, 3.9)
DISTANCES = (0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
RESOLUTIONS = (64, 128, 256, 512)
FAMILY_SEEDS = {"in_dist": 97001, "rough": 97002}
N_GEOMETRIES = 16
N_QUERY = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fourier_solution(points: np.ndarray, kappa: float, coefficients: np.ndarray) -> np.ndarray:
    """Entire modified-Helmholtz solutions in a polar Fourier/Bessel basis."""
    radius = np.linalg.norm(points, axis=-1)
    angle = np.arctan2(points[..., 1], points[..., 0])
    output = np.zeros_like(radius)
    for mode in range(1, 7):
        scale = max(float(iv(mode, 1.5 * kappa)), 1e-12)
        radial = iv(mode, kappa * radius) / scale
        output += radial * (
            coefficients[mode - 1, 0] * np.cos(mode * angle)
            + coefficients[mode - 1, 1] * np.sin(mode * angle)
        )
    return output


def make_geometries(family: str) -> list[dict]:
    rng = np.random.default_rng(FAMILY_SEEDS[family])
    settings = FAMILIES[family]
    fine = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    geometries = []
    while len(geometries) < N_GEOMETRIES:
        coefficients = ood_coeffs(rng, settings)
        if radius_at(coefficients, fine).min() <= 0.2:
            continue
        anchor = boundary(coefficients, N_QUERY)
        queries = {
            distance: anchor["X"] - distance * anchor["normal"]
            for distance in DISTANCES
        }
        geometries.append({
            "coefficients": coefficients,
            "charges": sample_charges(rng, "interior", coefficients, n_charges=6),
            "fourier": rng.normal(0.0, 1.0, size=(6, 2)) / np.arange(1, 7)[:, None],
            "queries": queries,
        })
    return geometries


def load_models(device: torch.device):
    manifest_path = ROOT / "results" / "screening_law" / "confirmatory_v1" / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    models = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        models.append((record["method"], record["seed_index"], model.to(device).eval()))
    return models


def stack_boundary(geometries: list[dict], n_boundary: int) -> tuple[list[dict], dict[str, torch.Tensor]]:
    curves = [boundary(geometry["coefficients"], n_boundary) for geometry in geometries]
    batch = {
        "Xb": torch.tensor(np.asarray([curve["X"] for curve in curves]), dtype=torch.float32),
        "Nb": torch.tensor(np.asarray([curve["normal"] for curve in curves]), dtype=torch.float32),
        "wb": torch.tensor(np.asarray([curve["w"] for curve in curves]), dtype=torch.float32),
        "curv": torch.tensor(np.asarray([curve["curv"] for curve in curves]), dtype=torch.float32),
    }
    return curves, batch


def relative_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=-1) / np.maximum(
        np.linalg.norm(target, axis=-1), 1e-12,
    )


def append_rows(rows: list[dict], errors: np.ndarray, *, method: str, seed: int,
                family: str, kappa: float, load: str, distance: float,
                n_boundary: int) -> None:
    for sample_id, error in enumerate(errors):
        rows.append({
            "method": method,
            "seed_index": seed,
            "family": family,
            "kappa": kappa,
            "load_family": load,
            "query_distance": distance,
            "n_boundary": n_boundary,
            "sample_id": sample_id,
            "rel_l2": float(error),
            "finite": bool(np.isfinite(error)),
        })


@torch.no_grad()
def run_family(family: str) -> None:
    output = RESULT / "raw" / f"{family}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    geometries = make_geometries(family)
    models = load_models(device)
    rows = []
    for kappa in KAPPAS:
        targets = {}
        traces = defaultdict(dict)
        for load in ("point_source", "fourier"):
            for distance in DISTANCES:
                targets[load, distance] = np.asarray([
                    eval_u(geometry["queries"][distance], geometry["charges"], kappa)
                    if load == "point_source" else
                    fourier_solution(geometry["queries"][distance], kappa, geometry["fourier"])
                    for geometry in geometries
                ])

        for n_boundary in RESOLUTIONS:
            curves, base_cpu = stack_boundary(geometries, n_boundary)
            for load in ("point_source", "fourier"):
                traces[load, n_boundary] = np.asarray([
                    eval_u(curve["X"], geometry["charges"], kappa)
                    if load == "point_source" else
                    fourier_solution(curve["X"], kappa, geometry["fourier"])
                    for curve, geometry in zip(curves, geometries)
                ])

                exact_evaluators = []
                laplace_evaluators = []
                for curve, trace in zip(curves, traces[load, n_boundary]):
                    _, exact_eval = bem_solve(curve, trace, "interior", kappa=kappa)
                    _, laplace_eval = bem_solve(curve, trace, "interior", kappa=0.0)
                    exact_evaluators.append(exact_eval)
                    laplace_evaluators.append(laplace_eval)
                for distance in DISTANCES:
                    exact_prediction = np.asarray([
                        evaluator(geometry["queries"][distance])
                        for evaluator, geometry in zip(exact_evaluators, geometries)
                    ])
                    laplace_prediction = np.asarray([
                        evaluator(geometry["queries"][distance])
                        for evaluator, geometry in zip(laplace_evaluators, geometries)
                    ])
                    target = targets[load, distance]
                    append_rows(rows, relative_errors(exact_prediction, target),
                                method="correct_yukawa_bie", seed=-1, family=family,
                                kappa=kappa, load=load, distance=distance,
                                n_boundary=n_boundary)
                    append_rows(rows, relative_errors(laplace_prediction, target),
                                method="fixed_laplace_bie", seed=-1, family=family,
                                kappa=kappa, load=load, distance=distance,
                                n_boundary=n_boundary)

                batch = {key: value.to(device) for key, value in base_cpu.items()}
                batch["g"] = torch.tensor(traces[load, n_boundary], dtype=torch.float32, device=device)
                batch["kappa"] = torch.full((N_GEOMETRIES,), kappa, device=device)
                for method, seed, model in models:
                    if hasattr(model, "assemble"):
                        matrix = model.assemble(batch)
                    else:
                        matrix = model.bio.assemble(
                            batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
                            batch["kappa"], model.jump,
                        )
                    density = model.solver(matrix, batch["g"], model.jump)
                    for distance in DISTANCES:
                        batch["Xq"] = torch.tensor(
                            np.asarray([geometry["queries"][distance] for geometry in geometries]),
                            dtype=torch.float32, device=device,
                        )
                        if hasattr(model, "potential"):
                            potential = model.potential(batch)
                        else:
                            potential = model.bio.potential(
                                batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"], batch["kappa"],
                            )
                        prediction = torch.bmm(potential, density.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                        append_rows(rows, relative_errors(prediction, targets[load, distance]),
                                    method=method, seed=seed, family=family, kappa=kappa,
                                    load=load, distance=distance, n_boundary=n_boundary)
                print(f"[{family}] k={kappa:g} load={load} nb={n_boundary}", flush=True)

    expected = 3 * 2 * 6 * 4 * 16 * 14
    if len(rows) != expected or not all(row["finite"] for row in rows):
        raise RuntimeError(f"coverage/finite failure: {len(rows)} != {expected}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


def aggregate() -> None:
    rows = []
    for family in FAMILY_SEEDS:
        with (RESULT / "raw" / f"{family}.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["family"], row["kappa"], row["load_family"],
                 row["query_distance"], row["n_boundary"])].append(float(row["rel_l2"]))
    summary = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values)
        summary.append({
            "method": key[0], "family": key[1], "kappa": float(key[2]),
            "load_family": key[3], "query_distance": float(key[4]),
            "n_boundary": int(key[5]), "n_rows": len(values),
            "mean_rel_l2": float(array.mean()), "median_rel_l2": float(np.median(array)),
            "p90_rel_l2": float(np.quantile(array, 0.9)),
        })
    RESULT.mkdir(parents=True, exist_ok=True)
    with (RESULT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    def pooled(method: str, distance: float, resolution: int) -> float:
        values = [float(row["rel_l2"]) for row in rows
                  if row["method"] == method and float(row["query_distance"]) == distance
                  and int(row["n_boundary"]) == resolution]
        return float(np.mean(values))

    methods = ("correct_yukawa_bie", "full_prior", "no_prior", "radial_mlp")
    matrices = {method: np.asarray([
        [pooled(method, distance, resolution) for resolution in RESOLUTIONS]
        for distance in DISTANCES
    ]) for method in methods}
    report = {
        "n_raw_rows": len(rows),
        "methods": {method: matrices[method].tolist() for method in methods},
        "distance_order": DISTANCES,
        "resolution_order": RESOLUTIONS,
        "interpretation": "descriptive post-confirmation phase map; no selection gate",
    }
    (RESULT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), constrained_layout=True)
    lo = min(np.log10(matrix).min() for matrix in matrices.values())
    hi = max(np.log10(matrix).max() for matrix in matrices.values())
    for axis, method in zip(axes, methods):
        image = axis.imshow(np.log10(matrices[method]), aspect="auto", vmin=lo, vmax=hi,
                            cmap="viridis")
        axis.set_title(method.replace("_", " "), fontsize=9)
        axis.set_xticks(range(len(RESOLUTIONS)), RESOLUTIONS)
        axis.set_yticks(range(len(DISTANCES)), [f"{value:g}" for value in DISTANCES])
        axis.set_xlabel("boundary nodes")
    axes[0].set_ylabel("normal query distance")
    fig.colorbar(image, ax=axes, label="log10 mean rel-L2", shrink=0.85)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)

    radial_mid = pooled("radial_mlp", 0.02, 128)
    exact_mid = pooled("correct_yukawa_bie", 0.02, 128)
    radial_close_128 = pooled("radial_mlp", 0.001, 128)
    radial_close_512 = pooled("radial_mlp", 0.001, 512)
    REPORT.write_text(f"""# Screening-law query-distance x resolution phase map

This is a descriptive post-confirmation diagnostic using fresh smooth/rough geometries. It does not alter
the registered Outcome B decision.

- Radial at distance `0.02`, `n_b=128`: `{radial_mid:.4e}` (exact BIE `{exact_mid:.4e}`).
- Radial at distance `0.001`: `{radial_close_128:.4e}` at `n_b=128` and
  `{radial_close_512:.4e}` at `n_b=512`.
- Full results cross three in-range kappa values, two exact manufactured load families, six normal query
  distances, four resolutions, and smooth/rough geometries.

The direct-evaluation phase map separates learned-law error from underresolved close quadrature. Yukawa QBX
is not implemented in this diagnostic, so extreme close-distance rows are numerical stress tests rather
than claims about a production close-evaluation path.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILY_SEEDS))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.family:
        run_family(args.family)
    else:
        parser.error("provide --family or --aggregate")


if __name__ == "__main__":
    main()
