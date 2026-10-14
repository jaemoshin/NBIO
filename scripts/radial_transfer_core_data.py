#!/usr/bin/env python3
"""Generate fresh analytic holdouts for the asymptotic and transformer studies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import iv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.bie import (  # noqa: E402
    boundary,
    eval_u,
    radius_at,
    sample_charges,
    sample_queries,
)
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402

DATA_ROOT = ROOT / "data_cache" / "radial_transfer_core_v1"
STUDIES = {
    "asymptotic": {
        "family_seeds": (119201, 119202, 119203, 119204),
        "kappas": (1.05, 1.65, 2.35, 3.05, 3.75),
    },
    "transformer": {
        "family_seeds": (119601, 119602, 119603, 119604),
        "kappas": (1.1, 1.7, 2.3, 3.0, 3.7),
    },
}
RESOLUTIONS = (64, 128, 256)
ASYMPTOTIC_RESOLUTIONS = (128, 256, 512)
LOADS = ("point_source", "fourier_bessel")
N_GEOMETRIES = 24
N_QUERY = 256
QUERY_MARGIN = 0.12


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def fourier_solution(
    points: np.ndarray, kappa: float, coefficients: np.ndarray,
) -> np.ndarray:
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


def write_once(payload: dict[str, torch.Tensor], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def make_geometries(family: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    fine = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    geometries = []
    while len(geometries) < N_GEOMETRIES:
        coefficients = ood_coeffs(rng, FAMILIES[family])
        if radius_at(coefficients, fine).min() <= 0.08:
            continue
        geometries.append({
            "coefficients": coefficients,
            "charges": sample_charges(
                rng, "interior", coefficients, n_charges=6,
            ),
            "fourier": rng.normal(size=(6, 2)) / np.arange(1, 7)[:, None],
            "queries": sample_queries(
                rng, coefficients, "interior", N_QUERY, margin=QUERY_MARGIN,
            ),
        })
    return geometries


def generate(study: str, family: str) -> None:
    specification = STUDIES[study]
    family_index = tuple(FAMILIES).index(family)
    geometries = make_geometries(
        family, specification["family_seeds"][family_index],
    )
    resolutions = (
        ASYMPTOTIC_RESOLUTIONS if study == "asymptotic" else RESOLUTIONS
    )
    for kappa in specification["kappas"]:
        for n_boundary in resolutions:
            curves = [
                boundary(geometry["coefficients"], n_boundary)
                for geometry in geometries
            ]
            for load in LOADS:
                records = {
                    key: [] for key in (
                        "Xb", "Nb", "wb", "curv", "g", "Xq", "uq",
                    )
                }
                for curve, geometry in zip(curves, geometries):
                    if load == "point_source":
                        trace = eval_u(curve["X"], geometry["charges"], kappa)
                        target = eval_u(
                            geometry["queries"], geometry["charges"], kappa,
                        )
                    else:
                        trace = fourier_solution(
                            curve["X"], kappa, geometry["fourier"],
                        )
                        target = fourier_solution(
                            geometry["queries"], kappa, geometry["fourier"],
                        )
                    records["Xb"].append(curve["X"])
                    records["Nb"].append(curve["normal"])
                    records["wb"].append(curve["w"])
                    records["curv"].append(curve["curv"])
                    records["g"].append(trace)
                    records["Xq"].append(geometry["queries"])
                    records["uq"].append(target)
                tensors = {
                    key: torch.tensor(np.asarray(values), dtype=torch.float32)
                    for key, values in records.items()
                }
                tensors["kappa"] = torch.full(
                    (N_GEOMETRIES,), float(kappa), dtype=torch.float32,
                )
                output = (
                    DATA_ROOT / study / "holdout"
                    / f"{load}_{family}_k{tag(kappa)}_nb{n_boundary}.pt"
                )
                write_once(tensors, output)
        print(f"[{study}] generated {family} kappa={kappa:g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    args = parser.parse_args()
    generate(args.study, args.family)


if __name__ == "__main__":
    main()
