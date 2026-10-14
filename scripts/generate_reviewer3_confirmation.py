#!/usr/bin/env python
"""Generate the preregistered untouched screened-family confirmation bank."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.bie import boundary, bem_solve, eval_u, radius_at, sample_charges, sample_queries  # noqa: E402
from tmlr_revision2_conditioned_transfer import fourier_coefficients, fourier_trace  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402

PROTOCOL_PATH = ROOT / "protocols" / "reviewer3_evidence_v1.json"
DATA_ROOT = ROOT / "data_cache" / "reviewer3_confirmation_v1"
FAMILY_SEEDS = (170201, 170202, 170203, 170204)
KAPPAS = (0.85, 1.45, 2.15, 2.85, 3.55)
RESOLUTIONS = (64, 128, 256)
LOADS = ("point_source", "fourier")
N_GEOMETRIES = 32
N_QUERY = 256
QUERY_MARGIN = 0.12
REFERENCE_RESOLUTION = 512


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def write_once(data: dict[str, torch.Tensor], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["status"] != "LOCKED_BEFORE_EXECUTION":
        raise RuntimeError("protocol is not locked")
    family_index = tuple(FAMILIES).index(args.family)
    rng = np.random.default_rng(FAMILY_SEEDS[family_index])
    fine_theta = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    geometries = []
    while len(geometries) < N_GEOMETRIES:
        coefficients = ood_coeffs(rng, FAMILIES[args.family])
        if radius_at(coefficients, fine_theta).min() <= 0.05:
            continue
        geometries.append({
            "coefficients": coefficients,
            "charges": sample_charges(rng, "interior", coefficients, n_charges=6),
            "fourier": fourier_coefficients(rng),
            "queries": sample_queries(
                rng, coefficients, "interior", N_QUERY, margin=QUERY_MARGIN,
            ),
        })
    geometry_fingerprints = [
        __import__("hashlib").sha256(
            json.dumps(
                geometry["coefficients"], sort_keys=True, separators=(",", ":"),
                default=lambda value: np.asarray(value).tolist(),
            ).encode("utf-8")
        ).hexdigest()
        for geometry in geometries
    ]
    metadata_path = DATA_ROOT / f"{args.family}_geometry_metadata.json"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "family": args.family,
        "seed": FAMILY_SEEDS[family_index],
        "geometry_fingerprints": geometry_fingerprints,
    }, indent=2, sort_keys=True) + "\n")

    for kappa in KAPPAS:
        for geometry in geometries:
            reference_boundary = boundary(geometry["coefficients"], REFERENCE_RESOLUTION)
            trace = fourier_trace(reference_boundary["theta"], geometry["fourier"])
            _, evaluator = bem_solve(reference_boundary, trace, "interior", kappa=kappa)
            geometry["fourier_reference"] = evaluator(geometry["queries"])
        for n_boundary in RESOLUTIONS:
            for load in LOADS:
                records = {key: [] for key in (
                    "Xb", "Nb", "wb", "curv", "g", "Xq", "uq", "bem_uq",
                )}
                for geometry in geometries:
                    discretization = boundary(geometry["coefficients"], n_boundary)
                    if load == "point_source":
                        trace = eval_u(discretization["X"], geometry["charges"], kappa)
                        target = eval_u(geometry["queries"], geometry["charges"], kappa)
                    else:
                        trace = fourier_trace(discretization["theta"], geometry["fourier"])
                        target = geometry["fourier_reference"]
                    _, evaluator = bem_solve(discretization, trace, "interior", kappa=kappa)
                    records["Xb"].append(discretization["X"])
                    records["Nb"].append(discretization["normal"])
                    records["wb"].append(discretization["w"])
                    records["curv"].append(discretization["curv"])
                    records["g"].append(trace)
                    records["Xq"].append(geometry["queries"])
                    records["uq"].append(target)
                    records["bem_uq"].append(evaluator(geometry["queries"]))
                tensors = {
                    key: torch.tensor(np.asarray(values), dtype=torch.float32)
                    for key, values in records.items()
                }
                tensors["kappa"] = torch.full((N_GEOMETRIES,), float(kappa))
                write_once(
                    tensors,
                    DATA_ROOT / f"{load}_{args.family}_k{tag(kappa)}_nb{n_boundary}.pt",
                )
        print(f"generated untouched family={args.family} kappa={kappa:g}", flush=True)


if __name__ == "__main__":
    main()
