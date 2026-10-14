#!/usr/bin/env python3
"""Frozen-checkpoint transfer from interior training to new BIE regimes/topologies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import kn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bie import boundary, radius_at, sample_coeffs  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.layers import double_layer_kernel  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402

RESULT_ROOT = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "topology_transfer"
)
CHECKPOINT_MANIFEST = (
    ROOT / "results" / "screening_law" / "radial_quadratic_v1"
    / "checkpoint_manifest.json"
)
REGIMES = (
    "single_obstacle_exterior",
    "two_hole_interior",
    "two_obstacle_exterior",
    "four_obstacle_exterior",
)
METHODS = (
    "radial_normal", "generic_matched", "generic_no_prior_matched",
)
KAPPAS = (1.2, 2.4, 3.8)
RESOLUTIONS = (64, 128, 256)
LOADS = ("distributed_sources", "dipole_sources")
N_GEOMETRIES = 16
N_QUERY = 256
SEED = 119001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component(
    rng: np.random.Generator,
    center: tuple[float, float],
    scale: float,
    *,
    reverse: bool = False,
) -> dict:
    return {
        "coefficients": sample_coeffs(
            rng, n_modes=4, decay=1.8, amp=0.16, r0=1.0,
        ),
        "center": np.asarray(center, dtype=np.float64),
        "scale": float(scale),
        "rotation": float(rng.uniform(0, 2 * np.pi)),
        "reverse": bool(reverse),
    }


def discretize(specification: dict, n_boundary: int) -> dict[str, np.ndarray]:
    local = boundary(specification["coefficients"], n_boundary)
    angle = specification["rotation"]
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    sign = -1.0 if specification["reverse"] else 1.0
    return {
        "X": (
            specification["scale"] * (local["X"] @ rotation.T)
            + specification["center"]
        ),
        "normal": sign * (local["normal"] @ rotation.T),
        "w": specification["scale"] * local["w"],
        "curv": sign * local["curv"] / specification["scale"],
    }


def normalized_radius(points: np.ndarray, specification: dict) -> np.ndarray:
    shifted = points - specification["center"]
    angle = -specification["rotation"]
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    local = shifted @ rotation.T
    theta = np.arctan2(local[:, 1], local[:, 0])
    curve_radius = (
        specification["scale"]
        * radius_at(specification["coefficients"], theta).reshape(-1)
    )
    return np.linalg.norm(shifted, axis=-1) / curve_radius


def make_geometry(rng: np.random.Generator, regime: str) -> dict:
    if regime == "single_obstacle_exterior":
        components = [component(rng, (0.0, 0.0), 0.95)]
        region = "exterior"
    elif regime == "two_hole_interior":
        components = [
            component(rng, (0.0, 0.0), 1.5),
            component(rng, (-0.48, 0.05), 0.24, reverse=True),
            component(rng, (0.48, -0.05), 0.24, reverse=True),
        ]
        region = "interior"
    elif regime == "two_obstacle_exterior":
        components = [
            component(rng, (-0.62, 0.0), 0.38),
            component(rng, (0.62, 0.0), 0.38),
        ]
        region = "exterior"
    elif regime == "four_obstacle_exterior":
        components = [
            component(rng, (-0.58, -0.58), 0.25),
            component(rng, (-0.58, 0.58), 0.25),
            component(rng, (0.58, -0.58), 0.25),
            component(rng, (0.58, 0.58), 0.25),
        ]
        region = "exterior"
    else:
        raise ValueError(regime)

    queries = []
    while len(queries) < N_QUERY:
        candidates = rng.uniform(-2.35, 2.35, size=(N_QUERY * 3, 2))
        ratios = [normalized_radius(candidates, item) for item in components]
        if region == "exterior":
            valid = np.logical_and.reduce([ratio > 1.15 for ratio in ratios])
            valid &= np.linalg.norm(candidates, axis=-1) < 2.35
        else:
            valid = ratios[0] < 0.86
            for ratio in ratios[1:]:
                valid &= ratio > 1.15
        queries.extend(candidates[valid].tolist())

    return {
        "components": components,
        "region": region,
        "queries": np.asarray(queries[:N_QUERY]),
    }


def make_charges(
    rng: np.random.Generator, geometry: dict, load: str,
) -> dict[str, np.ndarray]:
    components = geometry["components"]
    centers = []
    if geometry["region"] == "exterior":
        for item in components:
            centers.extend([
                item["center"] + item["scale"] * np.asarray((0.08, -0.04)),
                item["center"] + item["scale"] * np.asarray((-0.07, 0.06)),
            ])
    else:
        for item in components[1:]:
            centers.extend([
                item["center"] + item["scale"] * np.asarray((0.05, -0.03)),
                item["center"] + item["scale"] * np.asarray((-0.04, 0.05)),
            ])
        angles = rng.uniform(0, 2 * np.pi, size=4)
        centers.extend(np.stack((2.25 * np.cos(angles), 2.25 * np.sin(angles)), -1))
    centers = np.asarray(centers)
    if load == "dipole_sources":
        charges = np.resize(np.asarray((1.0, -1.0)), len(centers))
        charges *= rng.uniform(0.6, 1.4, size=len(centers))
    else:
        charges = rng.normal(0.4, 0.8, size=len(centers))
    charges /= max(np.linalg.norm(charges), 1e-12)
    return {"c": centers, "q": charges}


def evaluate_potential(
    points: np.ndarray, charges: dict[str, np.ndarray], kappa: float,
) -> np.ndarray:
    distance = np.linalg.norm(
        points[:, None] - charges["c"][None], axis=-1,
    )
    return (kn(0, kappa * distance) @ charges["q"]) / (2 * np.pi)


def tensor_sample(
    geometry: dict, charges: dict, kappa: float, n_boundary: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    curves = [discretize(item, n_boundary) for item in geometry["components"]]
    boundary_points = np.concatenate([curve["X"] for curve in curves], axis=0)
    trace = evaluate_potential(boundary_points, charges, kappa)
    target = evaluate_potential(geometry["queries"], charges, kappa)
    arrays = {
        "Xb": boundary_points,
        "Nb": np.concatenate([curve["normal"] for curve in curves], axis=0),
        "wb": np.concatenate([curve["w"] for curve in curves], axis=0),
        "curv": np.concatenate([curve["curv"] for curve in curves], axis=0),
        "g": trace,
        "Xq": geometry["queries"],
        "uq": target,
    }
    batch = {
        key: torch.tensor(value, dtype=torch.float64, device=device).unsqueeze(0)
        for key, value in arrays.items()
    }
    batch["kappa"] = torch.tensor([kappa], dtype=torch.float64, device=device)
    return batch


def exact_operators(batch: dict[str, torch.Tensor], jump: float):
    known = double_layer_kernel(
        batch["Xb"], batch["Xb"], batch["Nb"],
        curv_s=batch["curv"], self_kernel=True,
    )
    radius = torch.cdist(batch["Xb"], batch["Xb"]).clamp_min(1e-14)
    scaled = batch["kappa"].reshape(-1, 1, 1) * radius
    multiplier = scaled * torch.special.modified_bessel_k1(scaled)
    diagonal = torch.arange(batch["Xb"].shape[1], device=batch["Xb"].device)
    multiplier[:, diagonal, diagonal] = 1.0
    identity = torch.eye(
        batch["Xb"].shape[1], dtype=batch["Xb"].dtype,
        device=batch["Xb"].device,
    ).unsqueeze(0)
    matrix = jump * identity + known * multiplier * batch["wb"][:, None]
    query_known = double_layer_kernel(
        batch["Xq"], batch["Xb"], batch["Nb"],
    )
    query_radius = torch.cdist(
        batch["Xq"], batch["Xb"],
    ).clamp_min(1e-14)
    query_scaled = batch["kappa"].reshape(-1, 1, 1) * query_radius
    potential = (
        query_known
        * query_scaled
        * torch.special.modified_bessel_k1(query_scaled)
        * batch["wb"][:, None]
    )
    return matrix, potential


def load_models(device: torch.device):
    manifest = json.loads(CHECKPOINT_MANIFEST.read_text())
    if manifest["status"] != "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT":
        raise RuntimeError("frozen checkpoint manifest is not sealed")
    models = []
    for record in manifest["records"]:
        if record["method"] not in METHODS:
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"]).to(
            device=device, dtype=torch.float64,
        ).eval()
        models.append((record["method"], record["seed_index"], model))
    if len(models) != 9:
        raise RuntimeError(f"expected 9 frozen learned controls, found {len(models)}")
    return models


def relative_error(prediction, target) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1e-14)
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(regime: str, smoke: bool) -> None:
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(not smoke)
    rng = np.random.default_rng(SEED + REGIMES.index(regime))
    count = 1 if smoke else N_GEOMETRIES
    geometries = [make_geometry(rng, regime) for _ in range(count)]
    models = load_models(device)
    rows = []
    stability = []
    kappas = (1.2,) if smoke else KAPPAS
    resolutions = (64,) if smoke else RESOLUTIONS
    loads = ("distributed_sources",) if smoke else LOADS
    for geometry_id, geometry in enumerate(geometries):
        for load in loads:
            charges = make_charges(rng, geometry, load)
            for kappa in kappas:
                for n_boundary in resolutions:
                    batch = tensor_sample(
                        geometry, charges, kappa, n_boundary, device,
                    )
                    jump = -0.5 if geometry["region"] == "interior" else 0.5
                    exact_matrix, exact_potential = exact_operators(batch, jump)
                    exact_density = torch.linalg.solve(
                        exact_matrix, batch["g"].unsqueeze(-1),
                    ).squeeze(-1)
                    exact_prediction = torch.bmm(
                        exact_potential, exact_density.unsqueeze(-1),
                    ).squeeze(-1)
                    exact_error = relative_error(exact_prediction, batch["uq"])
                    base = {
                        "regime": regime,
                        "region": geometry["region"],
                        "load_family": load,
                        "kappa": kappa,
                        "n_boundary_per_component": n_boundary,
                        "n_components": len(geometry["components"]),
                        "n_boundary_total": batch["Xb"].shape[1],
                        "geometry_id": geometry_id,
                    }
                    rows.append({
                        **base,
                        "method": "exact_yukawa_bie",
                        "seed_index": -1,
                        "rel_l2": exact_error,
                        "finite": bool(np.isfinite(exact_error)),
                    })
                    exact_condition = (
                        float(torch.linalg.cond(exact_matrix))
                        if n_boundary == 128 else float("nan")
                    )
                    for method, seed, model in models:
                        model.jump = jump
                        prediction = model(batch)
                        error = relative_error(prediction, batch["uq"])
                        rows.append({
                            **base,
                            "method": method,
                            "seed_index": seed,
                            "rel_l2": error,
                            "finite": bool(np.isfinite(error)),
                        })
                        if hasattr(model, "assemble"):
                            matrix = model.assemble(batch)
                        else:
                            matrix = model.bio.assemble(
                                batch["Xb"], batch["Nb"], batch["wb"],
                                batch["curv"], batch["kappa"], jump,
                            )
                        density = model.solver(matrix, batch["g"], jump)
                        residual = relative_error(
                            torch.bmm(
                                matrix, density.unsqueeze(-1),
                            ).squeeze(-1),
                            batch["g"],
                        )
                        condition = (
                            float(torch.linalg.cond(matrix))
                            if n_boundary == 128 else float("nan")
                        )
                        stability.append({
                            **base,
                            "method": method,
                            "seed_index": seed,
                            "relative_residual": residual,
                            "condition_number": condition,
                            "exact_condition_number": exact_condition,
                            "condition_ratio": condition / exact_condition,
                        })
                    print(
                        f"[{regime}] geometry={geometry_id} load={load} "
                        f"k={kappa:g} nb/component={n_boundary} exact={exact_error:.3e}",
                        flush=True,
                    )
    if smoke:
        if not all(row["finite"] for row in rows):
            raise RuntimeError("nonfinite topology smoke")
        if rows[0]["rel_l2"] > 5e-3:
            raise RuntimeError(f"exact topology smoke too inaccurate: {rows[0]}")
        print(json.dumps({
            "status": "PASS",
            "regime": regime,
            "exact_rel_l2": rows[0]["rel_l2"],
            "learned_rows": len(rows) - 1,
        }, indent=2))
        return
    write_csv(rows, RESULT_ROOT / "raw" / f"{regime}.csv")
    write_csv(stability, RESULT_ROOT / "raw" / f"{regime}_stability.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.regime, args.smoke)


if __name__ == "__main__":
    main()
