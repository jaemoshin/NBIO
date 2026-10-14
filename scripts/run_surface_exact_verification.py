#!/usr/bin/env python3
"""Focused current-code verification of the surface QBX evidence.

This intentionally reruns only the canonical configurations cited by the
surface study instead of repeating broad parameter sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PI = math.pi


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def torus_laplace(torch_qbx, seed: int, device: torch.device) -> tuple[float, float]:
    nt, nph, oversample, p, cbeta = 44, 16, 3, 12, 2.0
    rng = np.random.default_rng(seed)
    xc_np, nrm_np, _ = torch_qbx.torus_np(nt, nph)
    xf_np, _, wf_np = torch_qbx.torus_np(nt * oversample, nph * oversample)
    up_np = torch_qbx.upsample_matrix(nt, nph, nt * oversample, nph * oversample)
    beta = cbeta * (2 * PI / nt)
    charges = torch_qbx._fib(8) * rng.uniform(2.0, 3.0, (8, 1))
    strengths = rng.normal(size=8)
    strengths -= strengths.mean()
    boundary = torch_qbx.u_harm(xc_np, charges, strengths)

    xc = torch.tensor(xc_np, device=device)
    nrm = torch.tensor(nrm_np, device=device)
    xf = torch.tensor(xf_np, device=device)
    wf = torch.tensor(wf_np, device=device)
    upsample = torch.tensor(up_np, device=device)
    g = torch.tensor(boundary, device=device)
    sf = torch_qbx.qbx_single_layer_torch(xc, nrm, xf, wf, beta, p)
    eye = torch.eye(len(xc_np), device=device)
    matrix = sf @ upsample + 1e-13 * eye
    density = torch.linalg.solve(matrix, g)
    density_fine = upsample @ density

    aa = np.linspace(0, 2 * PI, 400, endpoint=False)
    off = rng.uniform(0, 0.4 * 0.35, 400)
    be = rng.uniform(0, 2 * PI, 400)
    ring = 1.0 + off * np.cos(be)
    xq_np = np.stack([ring * np.cos(aa), ring * np.sin(aa), off * np.sin(be)], 1)
    truth = torch.tensor(torch_qbx.u_harm(xq_np, charges, strengths), device=device)
    xq = torch.tensor(xq_np, device=device)
    distance = torch.cdist(xq, xf).clamp_min(1e-15)
    prediction = (
        1.0 / (4 * PI * distance) * (wf * density_fine)[None, :]
    ).sum(1)
    rel_l2 = float(
        torch.linalg.vector_norm(prediction - truth) / torch.linalg.vector_norm(truth)
    )

    correction = torch.zeros_like(matrix, requires_grad=True)
    perturbed_density = torch.linalg.solve(matrix + correction, g)
    loss = (perturbed_density**2).mean()
    loss.backward()
    grad_norm = float(torch.linalg.vector_norm(correction.grad))
    return rel_l2, grad_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.surface_root.resolve()
    sys.path.insert(0, str(root))

    import sphere_qbx
    import torch_qbx
    import torus_helmholtz_qbx

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "device": str(device),
        "protocol": {
            "seeds": [0, 1],
            "sphere": "36x44, oversample=3, p=8, beta=2h",
            "torus_laplace": "44x16, oversample=3, p=12, beta=2h",
            "torus_helmholtz": "60x22, oversample=3, p=12, beta=2h",
        },
        "source_hashes": {
            name: sha256(root / name)
            for name in (
                "sphere_qbx.py",
                "torch_qbx.py",
                "torus_helmholtz_qbx.py",
            )
        },
    }

    sphere_errors = [
        sphere_qbx.run(36, 44, 3, 8, 2.0, 0.3, seed=seed) for seed in (0, 1)
    ]
    print(f"sphere errors: {sphere_errors}", flush=True)
    result["sphere_laplace"] = {
        "per_seed_rel_l2": sphere_errors,
        "mean_rel_l2": float(np.mean(sphere_errors)),
    }

    torus_rows = [torus_laplace(torch_qbx, seed, device) for seed in (0, 1)]
    print(f"torus Laplace (error, grad norm): {torus_rows}", flush=True)
    result["torus_laplace"] = {
        "per_seed_rel_l2": [row[0] for row in torus_rows],
        "mean_rel_l2": float(np.mean([row[0] for row in torus_rows])),
        "per_seed_grad_norm": [row[1] for row in torus_rows],
        "all_gradients_finite_positive": all(
            math.isfinite(row[1]) and row[1] > 0 for row in torus_rows
        ),
    }

    helmholtz = {}
    for kappa in (1.0, 2.0, 4.0):
        errors = [
            torus_helmholtz_qbx.run(
                60, 22, 3, 12, 2.0, kappa, seed=seed
            )
            for seed in (0, 1)
        ]
        print(f"torus Helmholtz k={kappa}: {errors}", flush=True)
        helmholtz[str(kappa)] = {
            "per_seed_rel_l2": errors,
            "mean_rel_l2": float(np.mean(errors)),
        }
    result["torus_helmholtz"] = helmholtz

    result["gates"] = {
        "sphere_below_2e-5": result["sphere_laplace"]["mean_rel_l2"] < 2e-5,
        "torus_laplace_below_2e-5": result["torus_laplace"]["mean_rel_l2"] < 2e-5,
        "helmholtz_all_below_2e-5": all(
            row["mean_rel_l2"] < 2e-5 for row in helmholtz.values()
        ),
        "autograd_finite_positive": result["torus_laplace"][
            "all_gradients_finite_positive"
        ],
    }
    result["all_gates_pass"] = all(result["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
