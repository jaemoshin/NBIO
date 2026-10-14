#!/usr/bin/env python
"""Fresh split-QBX validation for exact and learned screened double layers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from src.models.layers import double_layer_kernel  # noqa: E402
from src.numerics.qbx_torch import double_layer_qbx  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402

RESULT_ROOT = ROOT / "results" / "screening_law" / "split_qbx_v2"
REPORT = ROOT / "reports" / "screening_law_split_qbx_v2.md"
FIGURE = ROOT / "figures" / "screening_law_split_qbx_v2.png"
CHECKPOINT_SOURCE = Path(os.environ.get("NBIO_SCREENING_SOURCE", ROOT))
FAMILY_SEEDS = {"in_dist": 113201, "rough": 113202}
KAPPAS = (1.1, 2.2, 3.55)
DISTANCES = (0.01, 0.005, 0.001, 0.0005)
RESOLUTIONS = (128, 256, 512)
LOADS = ("point_source", "fourier_bessel")
N_GEOMETRIES = 16
N_QUERY = 128
QBX_KWARGS = {"order": 24, "r_fac": 2.0, "near_fac": 2.0, "oversample": 4}
PROTOCOL_NAME = "screening_law_split_qbx_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fourier_solution(points: np.ndarray, kappa: float, coefficients: np.ndarray) -> np.ndarray:
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
    fine = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    geometries = []
    while len(geometries) < N_GEOMETRIES:
        coefficients = ood_coeffs(rng, FAMILIES[family])
        if radius_at(coefficients, fine).min() <= 0.2:
            continue
        anchor = boundary(coefficients, N_QUERY)
        geometries.append({
            "coefficients": coefficients,
            "charges": sample_charges(rng, "interior", coefficients, n_charges=6),
            "fourier": rng.normal(size=(6, 2)) / np.arange(1, 7)[:, None],
            "queries": {
                distance: anchor["X"] - distance * anchor["normal"]
                for distance in DISTANCES
            },
        })
    return geometries


def load_radial_models(device: torch.device):
    manifest_path = (
        CHECKPOINT_SOURCE / "results" / "screening_law" / "confirmatory_v1"
        / "checkpoint_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    models = []
    for record in manifest["records"]:
        if record["method"] != "radial_mlp":
            continue
        path = CHECKPOINT_SOURCE / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device, dtype=torch.float64).eval()
        model.qbx_kw = dict(QBX_KWARGS)
        models.append((record["seed_index"], model))
    if len(models) != 3:
        raise RuntimeError(f"expected three radial checkpoints, found {len(models)}")
    return models, manifest_path


def exact_split_qbx(batch: dict[str, torch.Tensor], density: torch.Tensor) -> torch.Tensor:
    singular = double_layer_qbx(
        batch["Xq"], batch["Xb"], density,
        Nb=batch["Nb"], wb=batch["wb"], **QBX_KWARGS,
    )
    radius = torch.cdist(batch["Xq"], batch["Xb"]).clamp_min(1e-14)
    scaled = batch["kappa"].reshape(-1, 1, 1) * radius
    multiplier = scaled * torch.special.modified_bessel_k1(scaled)
    laplace = double_layer_kernel(batch["Xq"], batch["Xb"], batch["Nb"])
    remainder = laplace * (multiplier - 1.0) * batch["wb"][:, None]
    return singular + torch.bmm(remainder, density.unsqueeze(-1)).squeeze(-1)


def rel_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=-1) / np.maximum(
        np.linalg.norm(target, axis=-1), 1e-14,
    )


def append_rows(rows, values, *, method, seed, family, kappa, load, distance, n_boundary):
    for sample_id, value in enumerate(values):
        rows.append({
            "method": method,
            "seed_index": seed,
            "family": family,
            "kappa": kappa,
            "load_family": load,
            "query_distance": distance,
            "n_boundary": n_boundary,
            "sample_id": sample_id,
            "rel_l2": float(value),
            "finite": bool(np.isfinite(value)),
        })


def tensor_batch(curves, traces, queries, kappa, device):
    tensor = lambda values: torch.tensor(np.asarray(values), dtype=torch.float64, device=device)
    return {
        "Xb": tensor([curve["X"] for curve in curves]),
        "Nb": tensor([curve["normal"] for curve in curves]),
        "wb": tensor([curve["w"] for curve in curves]),
        "curv": tensor([curve["curv"] for curve in curves]),
        "g": tensor(traces),
        "Xq": tensor(queries),
        "kappa": torch.full((len(curves),), kappa, dtype=torch.float64, device=device),
    }


def gradient_check(model, batch):
    model.qbx = True
    model.zero_grad(set_to_none=True)
    query = batch["Xq"][:1, :1].detach().clone().requires_grad_(True)
    local = {key: value[:1] for key, value in batch.items()}
    local["Xq"] = query
    output = model(local).sum()
    parameter = model.radial_net[-1].weight
    gradient_query, gradient_parameter = torch.autograd.grad(output, (query, parameter))
    return {
        "output": float(output.detach()),
        "query_gradient_norm": float(gradient_query.norm()),
        "parameter_gradient_norm": float(gradient_parameter.norm()),
        "finite": bool(
            torch.isfinite(gradient_query).all() and torch.isfinite(gradient_parameter).all()
        ),
    }


def run_family(family: str) -> None:
    output = RESULT_ROOT / "raw" / f"{family}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    geometries = make_geometries(family)
    models, checkpoint_manifest = load_radial_models(device)
    rows = []
    gradient = None
    for kappa in KAPPAS:
        for n_boundary in RESOLUTIONS:
            curves = [boundary(geometry["coefficients"], n_boundary) for geometry in geometries]
            for load in LOADS:
                traces = np.asarray([
                    eval_u(curve["X"], geometry["charges"], kappa)
                    if load == "point_source" else
                    fourier_solution(curve["X"], kappa, geometry["fourier"])
                    for curve, geometry in zip(curves, geometries)
                ])
                targets = {
                    distance: np.asarray([
                        eval_u(geometry["queries"][distance], geometry["charges"], kappa)
                        if load == "point_source" else
                        fourier_solution(
                            geometry["queries"][distance], kappa, geometry["fourier"],
                        )
                        for geometry in geometries
                    ])
                    for distance in DISTANCES
                }
                exact_densities = []
                exact_evaluators = []
                for curve, trace in zip(curves, traces):
                    density, evaluator = bem_solve(curve, trace, "interior", kappa=kappa)
                    exact_densities.append(density)
                    exact_evaluators.append(evaluator)
                exact_density = torch.tensor(
                    np.asarray(exact_densities), dtype=torch.float64, device=device,
                )

                base = tensor_batch(curves, traces, [g["queries"][DISTANCES[0]] for g in geometries], kappa, device)
                learned_densities = []
                for seed, model in models:
                    matrix = model.assemble(base)
                    learned_densities.append((seed, model, model.solver(matrix, base["g"], model.jump)))

                for distance in DISTANCES:
                    queries = [geometry["queries"][distance] for geometry in geometries]
                    batch = tensor_batch(curves, traces, queries, kappa, device)
                    target = targets[distance]
                    exact_direct = np.asarray([
                        evaluator(query)
                        for evaluator, query in zip(exact_evaluators, queries)
                    ])
                    exact_split = exact_split_qbx(batch, exact_density).detach().cpu().numpy()
                    append_rows(
                        rows, rel_errors(exact_direct, target), method="exact_yukawa_direct",
                        seed=-1, family=family, kappa=kappa, load=load,
                        distance=distance, n_boundary=n_boundary,
                    )
                    append_rows(
                        rows, rel_errors(exact_split, target), method="exact_yukawa_split_qbx",
                        seed=-1, family=family, kappa=kappa, load=load,
                        distance=distance, n_boundary=n_boundary,
                    )
                    for seed, model, density in learned_densities:
                        model.qbx = False
                        direct = model.evaluate_density(batch, density).detach().cpu().numpy()
                        model.qbx = True
                        split = model.evaluate_density(batch, density).detach().cpu().numpy()
                        append_rows(
                            rows, rel_errors(direct, target), method="radial_direct",
                            seed=seed, family=family, kappa=kappa, load=load,
                            distance=distance, n_boundary=n_boundary,
                        )
                        append_rows(
                            rows, rel_errors(split, target), method="radial_split_qbx",
                            seed=seed, family=family, kappa=kappa, load=load,
                            distance=distance, n_boundary=n_boundary,
                        )
                    if gradient is None and family == "in_dist":
                        gradient = gradient_check(models[0][1], batch)
                print(
                    f"[{family}] kappa={kappa:g} load={load} n_b={n_boundary}",
                    flush=True,
                )
    expected = len(KAPPAS) * len(LOADS) * len(RESOLUTIONS) * len(DISTANCES) * N_GEOMETRIES * 8
    if len(rows) != expected or not all(row["finite"] for row in rows):
        raise RuntimeError(f"coverage/finite failure: {len(rows)} != {expected}")
    write_csv(rows, output)
    manifest = {
        "family": family,
        "seed": FAMILY_SEEDS[family],
        "row_count": len(rows),
        "qbx": QBX_KWARGS,
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
        "gradient_check": gradient,
    }
    (RESULT_ROOT / "raw" / f"{family}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def aggregate() -> None:
    outputs = (RESULT_ROOT / "summary.csv", RESULT_ROOT / "summary.json", REPORT, FIGURE)
    if any(path.exists() for path in outputs):
        raise FileExistsError("write-once aggregate output exists")
    rows = []
    manifests = []
    for family in FAMILY_SEEDS:
        with (RESULT_ROOT / "raw" / f"{family}.csv").open(newline="") as handle:
            family_rows = list(csv.DictReader(handle))
        if len(family_rows) != 9216:
            raise RuntimeError(f"row-count mismatch for {family}: {len(family_rows)}")
        rows.extend(family_rows)
        manifests.append(json.loads((RESULT_ROOT / "raw" / f"{family}_manifest.json").read_text()))
    for row in rows:
        row["kappa"] = float(row["kappa"])
        row["query_distance"] = float(row["query_distance"])
        row["n_boundary"] = int(row["n_boundary"])
        row["rel_l2"] = float(row["rel_l2"])
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["family"], row["kappa"], row["load_family"],
            row["query_distance"], row["n_boundary"],
        )
        grouped[key].append(row["rel_l2"])
    summary = []
    for key, values in sorted(grouped.items()):
        values = np.asarray(values)
        summary.append({
            "method": key[0], "family": key[1], "kappa": key[2],
            "load_family": key[3], "query_distance": key[4], "n_boundary": key[5],
            "n_rows": len(values), "mean_rel_l2": float(values.mean()),
            "median_rel_l2": float(np.median(values)),
            "p90_rel_l2": float(np.quantile(values, 0.9)),
        })
    write_csv(summary, RESULT_ROOT / "summary.csv")

    def pooled(method, distances=DISTANCES, resolutions=RESOLUTIONS):
        values = [
            row["rel_l2"] for row in rows
            if row["method"] == method
            and row["query_distance"] in distances
            and row["n_boundary"] in resolutions
        ]
        return float(np.mean(values))

    exact_split = pooled("exact_yukawa_split_qbx")
    radial_split = pooled("radial_split_qbx")
    radial_direct_close = pooled("radial_direct", distances=(0.005, 0.001, 0.0005))
    radial_split_close = pooled("radial_split_qbx", distances=(0.005, 0.001, 0.0005))
    c1 = exact_split <= 1e-3
    c2_ratio = radial_split_close / radial_direct_close
    c2 = c2_ratio <= 0.2
    c3_ratio = radial_split / exact_split
    c3 = c3_ratio <= 3.0
    c4_ratio = pooled("radial_split_qbx", distances=(0.01,)) / pooled(
        "radial_direct", distances=(0.01,)
    )
    c4 = c4_ratio <= 1.5
    gradient_checks = [manifest["gradient_check"] for manifest in manifests if manifest["gradient_check"]]
    c5 = bool(gradient_checks) and all(
        item["finite"] and item["query_gradient_norm"] > 0
        and item["parameter_gradient_norm"] > 0 for item in gradient_checks
    )
    verdict = "PAPERIZE" if all((c1, c2, c3, c4, c5)) else "NARROW"
    result = {
        "protocol": PROTOCOL_NAME,
        "n_raw_rows": len(rows),
        "pooled_mean_rel_l2": {
            "exact_yukawa_direct": pooled("exact_yukawa_direct"),
            "exact_yukawa_split_qbx": exact_split,
            "radial_direct": pooled("radial_direct"),
            "radial_split_qbx": radial_split,
        },
        "criteria": {
            "C1_exact_split_below_1e-3": c1,
            "C2_radial_close_split_to_direct_ratio": c2_ratio,
            "C2_pass": c2,
            "C3_radial_to_exact_split_ratio": c3_ratio,
            "C3_pass": c3,
            "C4_distance_1e-2_split_to_direct_ratio": c4_ratio,
            "C4_pass": c4,
            "C5_differentiable": c5,
        },
        "gradient_checks": gradient_checks,
        "qbx": QBX_KWARGS,
        "verdict": verdict,
    }
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    methods = (
        "exact_yukawa_direct", "exact_yukawa_split_qbx",
        "radial_direct", "radial_split_qbx",
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    colors = {
        "exact_yukawa_direct": "#7A7A7A", "exact_yukawa_split_qbx": "#111111",
        "radial_direct": "#D55E00", "radial_split_qbx": "#0072B2",
    }
    for method in methods:
        axes[0].plot(
            DISTANCES, [pooled(method, distances=(distance,)) for distance in DISTANCES],
            marker="o", label=method.replace("_", " "), color=colors[method],
        )
        axes[1].plot(
            RESOLUTIONS, [pooled(method, resolutions=(resolution,)) for resolution in RESOLUTIONS],
            marker="o", label=method.replace("_", " "), color=colors[method],
        )
    axes[0].set_xscale("log"); axes[0].invert_xaxis(); axes[0].set_yscale("log")
    axes[0].set_xlabel("normal query distance"); axes[0].set_ylabel("mean rel-L2")
    axes[1].set_yscale("log"); axes[1].set_xlabel("boundary nodes")
    axes[0].legend(fontsize=7)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)

    REPORT.write_text(f"""# Screened-kernel split-QBX validation

Verdict: **{verdict}**

The retained Laplace singular double layer is evaluated by differentiable QBX; the exact or learned regular
screening remainder is evaluated directly. The setting was selected in a preserved one-geometry numerical
diagnostic and then frozen on fresh geometries.

| method | pooled mean rel-L2 |
|---|---:|
| exact Yukawa direct | `{pooled('exact_yukawa_direct'):.4e}` |
| exact Yukawa split-QBX | `{exact_split:.4e}` |
| radial direct | `{pooled('radial_direct'):.4e}` |
| radial split-QBX | `{radial_split:.4e}` |

- Close-distance radial split/direct ratio: `{c2_ratio:.4e}`.
- Radial/exact split-QBX ratio: `{c3_ratio:.3f}`.
- Query and learned-parameter gradients through split-QBX are finite and nonzero: `{c5}`.

This validates the numerical composition of the analytic singular split and learned radial remainder. It
does not remove the data-supported-range limitation of the learned screening law.
""")
    print(json.dumps({"verdict": verdict, "criteria": result["criteria"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILY_SEEDS))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.family:
        run_family(args.family)
    elif args.aggregate:
        aggregate()
    else:
        parser.error("provide --family or --aggregate")


if __name__ == "__main__":
    main()
