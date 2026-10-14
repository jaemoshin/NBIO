#!/usr/bin/env python
"""Generate fresh, blinded data for the parameter-matched radial ablation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_kappa_continuous import generated_split  # noqa: E402
from src.data.bie import boundary, bem_solve, eval_u, radius_at, sample_charges, sample_queries  # noqa: E402
from tmlr_revision2_conditioned_transfer import fourier_coefficients, fourier_trace  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402

DATA_ROOT = ROOT / "data_cache" / "radial_factorization_ablation_v1"
TRAIN_SEEDS = (110001, 110002, 110003)
VAL_SEEDS = (110101, 110102, 110103)
FAMILY_SEEDS = (110201, 110202, 110203, 110204)
KAPPAS = (1.2, 1.75, 2.4, 3.1, 3.8)
RESOLUTIONS = (64, 128, 256)
LOADS = ("point_source", "fourier")
N_GEOMETRIES = 32
N_QUERY = 256
QUERY_MARGIN = 0.12
REFERENCE_RESOLUTION = 512


def kappa_tag(value: float) -> str:
    return str(value).replace(".", "p")


def write_once(data: dict[str, torch.Tensor], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def generate_train(seed_index: int) -> None:
    output = DATA_ROOT / f"data_s{seed_index}"
    write_once(generated_split(2400, TRAIN_SEEDS[seed_index]), output / "yukawa_train.pt")
    write_once(generated_split(400, VAL_SEEDS[seed_index]), output / "yukawa_val.pt")
    print(f"generated train/validation seed index {seed_index}", flush=True)


def generate_family(family: str) -> None:
    family_index = tuple(FAMILIES).index(family)
    rng = np.random.default_rng(FAMILY_SEEDS[family_index])
    fine_theta = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    geometries = []
    while len(geometries) < N_GEOMETRIES:
        coefficients = ood_coeffs(rng, FAMILIES[family])
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

    for kappa in KAPPAS:
        for geometry in geometries:
            reference_boundary = boundary(
                geometry["coefficients"], REFERENCE_RESOLUTION,
            )
            trace = fourier_trace(reference_boundary["theta"], geometry["fourier"])
            _, evaluator = bem_solve(reference_boundary, trace, "interior", kappa=kappa)
            geometry["fourier_reference"] = evaluator(geometry["queries"])

        for n_boundary in RESOLUTIONS:
            for load in LOADS:
                records = {
                    key: [] for key in (
                        "Xb", "Nb", "wb", "curv", "g", "Xq", "uq", "bem_uq",
                    )
                }
                for geometry in geometries:
                    discretization = boundary(geometry["coefficients"], n_boundary)
                    if load == "point_source":
                        trace = eval_u(discretization["X"], geometry["charges"], kappa)
                        target = eval_u(geometry["queries"], geometry["charges"], kappa)
                    else:
                        trace = fourier_trace(
                            discretization["theta"], geometry["fourier"],
                        )
                        target = geometry["fourier_reference"]
                    _, evaluator = bem_solve(
                        discretization, trace, "interior", kappa=kappa,
                    )
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
                output = DATA_ROOT / "holdout" / (
                    f"{load}_{family}_k{kappa_tag(kappa)}_nb{n_boundary}.pt"
                )
                write_once(tensors, output)
        print(f"generated blinded family={family} kappa={kappa:g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train-seed-index", type=int, choices=(0, 1, 2))
    group.add_argument("--family", choices=tuple(FAMILIES))
    args = parser.parse_args()
    if args.train_seed_index is not None:
        generate_train(args.train_seed_index)
    else:
        generate_family(args.family)


if __name__ == "__main__":
    main()
