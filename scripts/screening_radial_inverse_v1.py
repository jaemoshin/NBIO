#!/usr/bin/env python
"""Multi-target shape inverse audit for radial and generic screened NBIOs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.bie import bem_solve, boundary  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.nbio_yukawa import yukawa_double_layer_kernel  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from tmlr_revision_validation import torch_boundary  # noqa: E402

SOURCE = Path(os.environ.get("NBIO_SCREENING_SOURCE", ROOT))
RESULT_ROOT = ROOT / "results" / "screening_law" / "inverse_v1"
REPORT = ROOT / "reports" / "screening_law_inverse_v1.md"
MAX_COEFFICIENT = 0.18
N_LOADS = 4
TARGETS = (
    {"kappa": 1.1, "coefficients": (0.10, -0.05, 0.07, 0.04)},
    {"kappa": 2.2, "coefficients": (-0.08, 0.06, 0.03, -0.09)},
    {"kappa": 3.55, "coefficients": (0.04, 0.11, -0.07, 0.05)},
    {"kappa": 2.8, "coefficients": (-0.12, -0.03, 0.08, -0.04)},
)
METHOD_MAP = {
    "full_prior": "generic_full_prior",
    "no_prior": "generic_no_prior",
    "radial_mlp": "radial_normal_form",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(device):
    manifest_path = SOURCE / "results" / "screening_law" / "confirmatory_v1" / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    models = {}
    records = []
    for record in manifest["records"]:
        if record["seed_index"] != 0 or record["method"] not in METHOD_MAP:
            continue
        path = SOURCE / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval().requires_grad_(False)
        models[METHOD_MAP[record["method"]]] = model
        records.append({
            "method": METHOD_MAP[record["method"]],
            "path": str(path),
            "sha256": record["sha256"],
        })
    if len(models) != 3:
        raise RuntimeError(f"expected three models, got {sorted(models)}")
    return models, records


def load_traces(n_boundary, device, dtype):
    theta = torch.arange(n_boundary, device=device, dtype=dtype) * (2.0 * math.pi / n_boundary)
    traces = []
    for load_id in range(N_LOADS):
        rng = np.random.default_rng(114001 + load_id)
        modes = torch.arange(1, 7, device=device, dtype=dtype)[:, None]
        a = torch.tensor(rng.normal(size=6) / np.arange(1, 7) ** 1.5, device=device, dtype=dtype)
        b = torch.tensor(rng.normal(size=6) / np.arange(1, 7) ** 1.5, device=device, dtype=dtype)
        trace = a @ torch.cos(modes * theta[None]) + b @ torch.sin(modes * theta[None])
        traces.append(trace / trace.square().mean().sqrt().clamp_min(1e-12))
    return torch.stack(traces)


def query_points(target_index, count):
    rng = np.random.default_rng(114101 + target_index)
    radius = 0.42 * np.sqrt(rng.uniform(size=count))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def refined_target(target_index, n_reference, query):
    target = TARGETS[target_index]
    coefficients = np.asarray(target["coefficients"])
    geometry = {"r0": 1.0, "a": coefficients[:2], "b": coefficients[2:]}
    discretization = boundary(geometry, n_reference)
    traces = load_traces(n_reference, torch.device("cpu"), torch.float64).numpy()
    values = []
    for trace in traces:
        _, evaluator = bem_solve(
            discretization, trace, "interior", kappa=target["kappa"],
        )
        values.append(evaluator(query))
    return np.stack(values)


def shape_tensors(raw, n_boundary):
    coefficients = MAX_COEFFICIENT * torch.tanh(raw)
    return torch_boundary(
        coefficients[:2], coefficients[2:], n_boundary, raw.dtype, raw.device,
    )


def exact_forward(raw, traces, query, n_boundary, kappa):
    x, normal, weights, curvature = shape_tensors(raw, n_boundary)
    batch_size = traces.shape[0]
    xb = x.unsqueeze(0).expand(batch_size, -1, -1)
    nb = normal.unsqueeze(0).expand(batch_size, -1, -1)
    wb = weights.unsqueeze(0).expand(batch_size, -1)
    curv = curvature.unsqueeze(0).expand(batch_size, -1)
    xq = query.unsqueeze(0).expand(batch_size, -1, -1)
    kernel = yukawa_double_layer_kernel(
        xb, xb, nb, kappa, curv_s=curv, self_kernel=True,
    )
    identity = torch.eye(n_boundary, device=raw.device, dtype=raw.dtype).unsqueeze(0)
    matrix = -0.5 * identity + kernel * wb[:, None]
    density = torch.linalg.solve(matrix, traces.unsqueeze(-1)).squeeze(-1)
    potential = yukawa_double_layer_kernel(xq, xb, nb, kappa) * wb[:, None]
    return torch.bmm(potential, density.unsqueeze(-1)).squeeze(-1)


def learned_forward(model, raw, traces, query, n_boundary, kappa):
    x, normal, weights, curvature = shape_tensors(raw, n_boundary)
    batch_size = traces.shape[0]
    batch = {
        "Xb": x.unsqueeze(0).expand(batch_size, -1, -1),
        "Nb": normal.unsqueeze(0).expand(batch_size, -1, -1),
        "wb": weights.unsqueeze(0).expand(batch_size, -1),
        "curv": curvature.unsqueeze(0).expand(batch_size, -1),
        "g": traces,
        "Xq": query.unsqueeze(0).expand(batch_size, -1, -1),
        "kappa": torch.full((batch_size,), kappa, device=raw.device, dtype=raw.dtype),
    }
    return model(batch)


def objective(method, model, raw, traces, query, target, n_boundary, kappa):
    prediction = (
        exact_forward(raw, traces, query, n_boundary, kappa)
        if model is None else learned_forward(model, raw, traces, query, n_boundary, kappa)
    )
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)


def initial_raw(start_index, device, dtype):
    generator = torch.Generator(device=device).manual_seed(114201 + start_index)
    return 0.35 * torch.randn(4, generator=generator, device=device, dtype=dtype)


def gradient_agreement(method, model, start_index, traces, query, target, n_boundary, kappa):
    raw = initial_raw(start_index, query.device, query.dtype).requires_grad_(True)
    exact_loss = objective("exact_yukawa", None, raw, traces, query, target, n_boundary, kappa)
    exact_gradient = torch.autograd.grad(exact_loss, raw)[0]
    if model is None:
        gradient = exact_gradient
    else:
        raw_model = raw.detach().clone().requires_grad_(True)
        model_loss = objective(method, model, raw_model, traces, query, target, n_boundary, kappa)
        gradient = torch.autograd.grad(model_loss, raw_model)[0]
    cosine = torch.nn.functional.cosine_similarity(
        gradient.reshape(1, -1), exact_gradient.reshape(1, -1), dim=-1,
    )[0]
    relative = (gradient - exact_gradient).norm() / exact_gradient.norm().clamp_min(1e-12)
    return float(cosine), float(relative)


def coefficient_metrics(raw, true_coefficients):
    estimate = (MAX_COEFFICIENT * torch.tanh(raw)).detach().cpu().numpy()
    error = estimate - true_coefficients
    return estimate, float(np.sqrt(np.mean(np.square(error))))


def optimize(method, model, target_index, start_index, traces, query, target,
             n_boundary, steps, learning_rate):
    true_coefficients = np.asarray(TARGETS[target_index]["coefficients"])
    kappa = TARGETS[target_index]["kappa"]
    raw = initial_raw(start_index, query.device, query.dtype).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate / 50.0,
    )
    rows = []
    for step in range(steps + 1):
        loss = objective(method, model, raw, traces, query, target, n_boundary, kappa)
        if step % 10 == 0 or step == steps:
            estimate, rmse = coefficient_metrics(raw, true_coefficients)
            rows.append({
                "target_index": target_index,
                "kappa": kappa,
                "method": method,
                "start_index": start_index,
                "step": step,
                "forward_rel_l2": float(torch.sqrt(loss).detach()),
                "coefficient_rmse": rmse,
                "a1": float(estimate[0]), "a2": float(estimate[1]),
                "b1": float(estimate[2]), "b2": float(estimate[3]),
                "finite": bool(torch.isfinite(loss)),
            })
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 10.0)
        optimizer.step(); scheduler.step()
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run_target(args):
    output = RESULT_ROOT / "raw" / f"target_{args.target_index}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch(); torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    models, checkpoint_records = load_models(device)
    query_np = query_points(args.target_index, args.n_query)
    target_np = refined_target(args.target_index, args.n_reference, query_np)
    query = torch.tensor(query_np, device=device, dtype=torch.float32)
    target = torch.tensor(target_np, device=device, dtype=torch.float32)
    traces = load_traces(args.n_boundary, device, torch.float32)
    rows = []
    gradients = []
    methods = {"exact_yukawa": None, **models}
    for method, model in methods.items():
        for start_index in range(args.n_starts):
            cosine, relative = gradient_agreement(
                method, model, start_index, traces, query, target,
                args.n_boundary, TARGETS[args.target_index]["kappa"],
            )
            gradients.append({
                "target_index": args.target_index, "method": method,
                "start_index": start_index, "gradient_cosine_to_exact": cosine,
                "gradient_relative_error": relative,
            })
            rows.extend(optimize(
                method, model, args.target_index, start_index, traces, query, target,
                args.n_boundary, args.steps, args.learning_rate,
            ))
            print(f"[inverse] target={args.target_index} method={method} start={start_index}", flush=True)
    write_csv(rows, output)
    write_csv(gradients, RESULT_ROOT / "raw" / f"target_{args.target_index}_gradients.csv")
    (RESULT_ROOT / "raw" / f"target_{args.target_index}_manifest.json").write_text(
        json.dumps({
            "target": TARGETS[args.target_index],
            "n_boundary": args.n_boundary,
            "n_reference": args.n_reference,
            "n_query": args.n_query,
            "n_starts": args.n_starts,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "checkpoint_records": checkpoint_records,
        }, indent=2, sort_keys=True) + "\n"
    )


def aggregate(args):
    rows = []
    gradients = []
    for target_index in range(len(TARGETS)):
        with (RESULT_ROOT / "raw" / f"target_{target_index}.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
        with (RESULT_ROOT / "raw" / f"target_{target_index}_gradients.csv").open(newline="") as handle:
            gradients.extend(csv.DictReader(handle))
    final = [row for row in rows if int(row["step"]) == args.steps]
    summary = {}
    for method in sorted({row["method"] for row in final}):
        selected = [row for row in final if row["method"] == method]
        forward = np.asarray([float(row["forward_rel_l2"]) for row in selected])
        coefficient = np.asarray([float(row["coefficient_rmse"]) for row in selected])
        grad = [row for row in gradients if row["method"] == method]
        cosine = np.asarray([float(row["gradient_cosine_to_exact"]) for row in grad])
        summary[method] = {
            "n_runs": len(selected),
            "mean_forward_rel_l2": float(forward.mean()),
            "median_forward_rel_l2": float(np.median(forward)),
            "mean_coefficient_rmse": float(coefficient.mean()),
            "median_coefficient_rmse": float(np.median(coefficient)),
            "success_fraction": float(np.mean((forward < 0.01) & (coefficient < 0.03))),
            "median_gradient_cosine_to_exact": float(np.median(cosine)),
            "minimum_gradient_cosine_to_exact": float(cosine.min()),
        }
    radial = summary["radial_normal_form"]
    best_generic = min(
        summary["generic_full_prior"]["mean_coefficient_rmse"],
        summary["generic_no_prior"]["mean_coefficient_rmse"],
    )
    gates = {
        "radial_success_fraction_at_least_0p75": radial["success_fraction"] >= 0.75,
        "radial_median_coefficient_rmse_below_0p01": radial["median_coefficient_rmse"] < 0.01,
        "radial_median_gradient_cosine_at_least_0p95": radial["median_gradient_cosine_to_exact"] >= 0.95,
        "radial_mean_rmse_at_most_half_best_generic": radial["mean_coefficient_rmse"] <= 0.5 * best_generic,
    }
    verdict = "PAPERIZE" if all(gates.values()) else "NARROW"
    result = {"protocol": "screening_radial_inverse_v1", "summary": summary,
              "gates": gates, "verdict": verdict}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    REPORT.write_text(f"""# Radial NBIO multi-target shape inverse audit

Verdict: **{verdict}**

Four unseen target shapes span `kappa={{{', '.join(str(row['kappa']) for row in TARGETS)}}}`. Every method
uses the same four boundary loads, query points, four fixed starts per target, and 300 optimization steps.
Targets come from a 512-node exact Yukawa BIE; candidate solves use 128 nodes.

| method | success | mean coefficient RMSE | mean observation rel-L2 | median gradient cosine |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {method.replace('_', ' ')} | {values['success_fraction']:.3f} | "
        f"{values['mean_coefficient_rmse']:.4e} | {values['mean_forward_rel_l2']:.4e} | "
        f"{values['median_gradient_cosine_to_exact']:.4f} |"
        for method, values in summary.items()
    ) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=int, choices=range(len(TARGETS)))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--n-boundary", type=int, default=128)
    parser.add_argument("--n-reference", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=64)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args)
    elif args.target_index is not None:
        run_target(args)
    else:
        parser.error("provide --target-index or --aggregate")


if __name__ == "__main__":
    main()
