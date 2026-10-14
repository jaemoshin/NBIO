#!/usr/bin/env python
"""Small shape-inverse problem through exact and learned NBIO layers.

The target uses a refined SciPy/Numpy Yukawa BIE. Candidate shapes are optimized
through a 128-node differentiable solve from four independent boundary loads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bie import bem_solve, boundary  # noqa: E402
from src.models.nbio_yukawa import yukawa_double_layer_kernel  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from eval_kappa_conditioned import checkpoint_model  # noqa: E402
from tmlr_revision_validation import torch_boundary  # noqa: E402


KAPPA = 2.2
TRUE_COEFFICIENTS = np.asarray([0.10, -0.05, 0.07, 0.04], dtype=np.float64)
MAX_COEFFICIENT = 0.18
N_LOADS = 4


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_coefficients(n_b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    theta = torch.arange(n_b, device=device, dtype=dtype) * (2.0 * math.pi / n_b)
    values = []
    for load_id in range(N_LOADS):
        rng = np.random.default_rng(202607220 + load_id)
        a = torch.tensor(rng.normal(size=6) / np.arange(1, 7) ** 1.5, device=device, dtype=dtype)
        b = torch.tensor(rng.normal(size=6) / np.arange(1, 7) ** 1.5, device=device, dtype=dtype)
        modes = torch.arange(1, 7, device=device, dtype=dtype)[:, None]
        trace = a @ torch.cos(modes * theta[None, :]) + b @ torch.sin(modes * theta[None, :])
        values.append(trace / trace.square().mean().sqrt().clamp_min(1e-12))
    return torch.stack(values)


def query_points(n_query: int) -> np.ndarray:
    rng = np.random.default_rng(202607221)
    radius = 0.42 * np.sqrt(rng.uniform(size=n_query))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_query)
    return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=-1)


def refined_target(n_ref: int, xq: np.ndarray) -> np.ndarray:
    coeffs = {
        "r0": 1.0,
        "a": TRUE_COEFFICIENTS[:2].copy(),
        "b": TRUE_COEFFICIENTS[2:].copy(),
    }
    bd = boundary(coeffs, n_ref)
    theta = torch.arange(n_ref, dtype=torch.float64) * (2.0 * math.pi / n_ref)
    loads = load_coefficients(n_ref, torch.device("cpu"), torch.float64).numpy()
    values = []
    for g in loads:
        _, evaluator = bem_solve(bd, g, "interior", kappa=KAPPA)
        values.append(evaluator(xq))
    return np.stack(values)


def shape_tensors(raw: torch.Tensor, n_b: int) -> tuple[torch.Tensor, ...]:
    coefficients = MAX_COEFFICIENT * torch.tanh(raw)
    return torch_boundary(
        coefficients[:2], coefficients[2:], n_b, raw.dtype, raw.device
    )


def exact_forward(raw: torch.Tensor, loads: torch.Tensor, xq: torch.Tensor, n_b: int) -> torch.Tensor:
    x, normal, weights, curvature = shape_tensors(raw, n_b)
    batch = loads.shape[0]
    xb = x.unsqueeze(0).expand(batch, -1, -1)
    nb = normal.unsqueeze(0).expand(batch, -1, -1)
    wb = weights.unsqueeze(0).expand(batch, -1)
    curv = curvature.unsqueeze(0).expand(batch, -1)
    xq_batch = xq.unsqueeze(0).expand(batch, -1, -1)
    kernel = yukawa_double_layer_kernel(
        xb, xb, nb, KAPPA, curv_s=curv, self_kernel=True
    )
    identity = torch.eye(n_b, device=raw.device, dtype=raw.dtype).unsqueeze(0)
    matrix = -0.5 * identity + kernel * wb[:, None, :]
    density = torch.linalg.solve(matrix, loads.unsqueeze(-1)).squeeze(-1)
    potential = yukawa_double_layer_kernel(xq_batch, xb, nb, KAPPA)
    potential = potential * wb[:, None, :]
    return torch.bmm(potential, density.unsqueeze(-1)).squeeze(-1)


def learned_forward(
    model: torch.nn.Module, raw: torch.Tensor, loads: torch.Tensor,
    xq: torch.Tensor, n_b: int,
) -> torch.Tensor:
    x, normal, weights, curvature = shape_tensors(raw, n_b)
    batch = loads.shape[0]
    model_batch = {
        "Xb": x.unsqueeze(0).expand(batch, -1, -1),
        "Nb": normal.unsqueeze(0).expand(batch, -1, -1),
        "wb": weights.unsqueeze(0).expand(batch, -1),
        "curv": curvature.unsqueeze(0).expand(batch, -1),
        "g": loads,
        "Xq": xq.unsqueeze(0).expand(batch, -1, -1),
        "kappa": torch.full((batch,), KAPPA, device=raw.device, dtype=raw.dtype),
    }
    return model(model_batch)


def coefficient_metrics(raw: torch.Tensor) -> tuple[np.ndarray, float, float]:
    estimate = (MAX_COEFFICIENT * torch.tanh(raw)).detach().cpu().numpy()
    error = estimate - TRUE_COEFFICIENTS
    return estimate, float(np.sqrt(np.mean(np.square(error)))), float(np.linalg.norm(error))


def optimize_one(
    method: str, model: torch.nn.Module | None, start_seed: int,
    target: torch.Tensor, loads: torch.Tensor, xq: torch.Tensor,
    n_b: int, steps: int, learning_rate: float,
) -> list[dict]:
    generator = torch.Generator(device=xq.device).manual_seed(202607300 + start_seed)
    raw = (0.35 * torch.randn(4, generator=generator, device=xq.device, dtype=xq.dtype)).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=learning_rate / 50.0)
    rows = []
    for step in range(steps + 1):
        prediction = (
            exact_forward(raw, loads, xq, n_b)
            if model is None else learned_forward(model, raw, loads, xq, n_b)
        )
        loss = (prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)
        if step % 10 == 0 or step == steps:
            estimate, rmse, l2_error = coefficient_metrics(raw)
            forward_rel = float(torch.sqrt(loss).detach())
            rows.append({
                "method": method,
                "start_seed": start_seed,
                "step": step,
                "objective": float(loss.detach()),
                "forward_rel_l2": forward_rel,
                "coefficient_rmse": rmse,
                "coefficient_l2_error": l2_error,
                "a1": float(estimate[0]),
                "a2": float(estimate[1]),
                "b1": float(estimate[2]),
                "b2": float(estimate[3]),
                "finite": bool(torch.isfinite(loss)),
            })
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 10.0)
        optimizer.step()
        scheduler.step()
    return rows


def summarize(rows: list[dict], steps: int) -> dict:
    output = {}
    for method in sorted(set(row["method"] for row in rows)):
        final = [row for row in rows if row["method"] == method and row["step"] == steps]
        output[method] = {
            "n_starts": len(final),
            "mean_forward_rel_l2": float(np.mean([row["forward_rel_l2"] for row in final])),
            "median_forward_rel_l2": float(np.median([row["forward_rel_l2"] for row in final])),
            "mean_coefficient_rmse": float(np.mean([row["coefficient_rmse"] for row in final])),
            "median_coefficient_rmse": float(np.median([row["coefficient_rmse"] for row in final])),
            "fraction_forward_below_1e-2": float(np.mean([row["forward_rel_l2"] < 1e-2 for row in final])),
            "fraction_coefficient_rmse_below_3e-2": float(np.mean([row["coefficient_rmse"] < 3e-2 for row in final])),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-b", type=int, default=128)
    parser.add_argument("--n-ref", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=64)
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument(
        "--checkpoint-root", default="results/kappa_seed_replication"
    )
    parser.add_argument("--out-dir", default="results/tmlr_revision2")
    args = parser.parse_args()
    configure_torch()
    device = get_device(True)
    xq_np = query_points(args.n_query)
    target_np = refined_target(args.n_ref, xq_np)
    xq = torch.tensor(xq_np, device=device, dtype=torch.float32)
    target = torch.tensor(target_np, device=device, dtype=torch.float32)
    loads = load_coefficients(args.n_b, device, torch.float32)

    checkpoint_root = ROOT / args.checkpoint_root
    full_path = checkpoint_root / "data_s0_full" / "best.pt"
    none_path = checkpoint_root / "data_s0_none" / "best.pt"
    full, _ = checkpoint_model(full_path, device)
    none, _ = checkpoint_model(none_path, device)
    for model in (full, none):
        model.requires_grad_(False)

    rows = []
    for method, model in (
        ("correct_yukawa_bie", None),
        ("conditioned_full_prior", full),
        ("conditioned_no_prior", none),
    ):
        for start_seed in range(args.n_starts):
            rows.extend(optimize_one(
                method, model, start_seed, target, loads, xq, args.n_b,
                args.steps, args.learning_rate,
            ))
            print(f"[inverse] method={method} start={start_seed}", flush=True)

    result = {
        "protocol": {
            "pde": "interior modified Helmholtz",
            "kappa": KAPPA,
            "unknowns": ["a1", "a2", "b1", "b2"],
            "true_coefficients": TRUE_COEFFICIENTS.tolist(),
            "n_boundary_optimization": args.n_b,
            "n_boundary_reference": args.n_ref,
            "n_boundary_loads": N_LOADS,
            "n_query": args.n_query,
            "n_starts": args.n_starts,
            "steps": args.steps,
            "optimizer": "Adam with cosine decay and fixed initialization seeds",
            "target": "refined SciPy/Numpy correct-Yukawa BIE",
            "checkpoint_root": str(checkpoint_root.relative_to(ROOT)),
        },
        "summary": summarize(rows, args.steps),
        "kill_criterion": (
            "Claim practical differentiable inverse use only if at least 75% of fixed starts reach "
            "coefficient RMSE < 0.03 and forward rel-L2 < 0.01."
        ),
    }
    out = ROOT / args.out_dir
    write_csv(rows, out / "inverse_demo_raw.csv")
    (out / "inverse_demo_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
