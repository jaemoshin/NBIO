#!/usr/bin/env python
"""Small pre-protocol diagnostic for split-QBX numerical settings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bie import boundary, bem_solve, eval_u, sample_charges, sample_coeffs  # noqa: E402
from src.models.layers import double_layer_kernel  # noqa: E402
from src.numerics.qbx_torch import double_layer_qbx  # noqa: E402


def rel(prediction, target):
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1e-14)
    )


def exact_split(batch, density, **qbx_kwargs):
    singular = double_layer_qbx(
        batch["Xq"], batch["Xb"], density,
        Nb=batch["Nb"], wb=batch["wb"], **qbx_kwargs,
    )
    radius = torch.cdist(batch["Xq"], batch["Xb"]).clamp_min(1e-14)
    scaled = batch["kappa"].reshape(-1, 1, 1) * radius
    multiplier = scaled * torch.special.modified_bessel_k1(scaled)
    laplace = double_layer_kernel(batch["Xq"], batch["Xb"], batch["Nb"])
    remainder = laplace * (multiplier - 1.0) * batch["wb"][:, None]
    return singular + torch.bmm(remainder, density.unsqueeze(-1)).squeeze(-1)


def main() -> None:
    torch.set_num_threads(8)
    dtype = torch.float64
    rng = np.random.default_rng(113001)
    coefficients = sample_coeffs(rng)
    charges = sample_charges(rng, "interior", coefficients, n_charges=6)
    kappa = 2.55
    settings = (
        (12, 4.0, 4), (24, 2.0, 4), (32, 1.0, 4),
        (40, 0.75, 8), (56, 0.5, 8),
    )
    rows = []
    for n_boundary in (128, 256):
        discretization = boundary(coefficients, n_boundary)
        trace = eval_u(discretization["X"], charges, kappa)
        density, direct = bem_solve(discretization, trace, "interior", kappa=kappa)
        for distance in (0.01, 0.005, 0.001, 0.0005):
            queries = discretization["X"] - distance * discretization["normal"]
            target_np = eval_u(queries, charges, kappa)
            batch = {
                "Xb": torch.tensor(discretization["X"], dtype=dtype).unsqueeze(0),
                "Nb": torch.tensor(discretization["normal"], dtype=dtype).unsqueeze(0),
                "wb": torch.tensor(discretization["w"], dtype=dtype).unsqueeze(0),
                "Xq": torch.tensor(queries, dtype=dtype).unsqueeze(0),
                "kappa": torch.tensor([kappa], dtype=dtype),
            }
            density_tensor = torch.tensor(density, dtype=dtype).unsqueeze(0)
            target = torch.tensor(target_np, dtype=dtype).unsqueeze(0)
            rows.append({
                "method": "direct_exact",
                "n_boundary": n_boundary,
                "query_distance": distance,
                "rel_l2": float(
                    np.linalg.norm(direct(queries) - target_np)
                    / np.linalg.norm(target_np)
                ),
            })
            for order, r_fac, oversample in settings:
                prediction = exact_split(
                    batch, density_tensor, order=order, r_fac=r_fac,
                    near_fac=2.0, oversample=oversample,
                )
                rows.append({
                    "method": "split_qbx",
                    "order": order,
                    "r_fac": r_fac,
                    "oversample": oversample,
                    "n_boundary": n_boundary,
                    "query_distance": distance,
                    "rel_l2": rel(prediction, target),
                })
                print(rows[-1], flush=True)
    output = ROOT / "results" / "screening_law" / "split_qbx_exploratory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "pre_protocol_numerical_setting_diagnostic",
        "seed": 113001,
        "kappa": kappa,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
