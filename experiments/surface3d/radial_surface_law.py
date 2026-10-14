#!/usr/bin/env python
"""Audit a dimensionless radial Yukawa law on verified 3D surface QBX cases.

The analytic Laplace single-layer remains in the existing QBX matrices. Models
learn only the continuous Yukawa remainder and use it in both density assembly
and query evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import m1_combined as m1
import surface_varkernel as vk

PI = math.pi
TRAIN_KAPPAS = (0.5, 1.0, 2.0, 3.0, 4.0)
INTERP_KAPPAS = (0.75, 1.5, 2.5, 3.5)
EXTRAP_KAPPAS = (0.25, 4.5, 5.0)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _mkdir_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def torch_yukawa(points, centers, charges, kappa):
    distance = torch.cdist(points, centers).clamp_min(1e-15)
    return (torch.exp(-float(kappa) * distance) / (4 * PI * distance)) @ charges


def at_kappa(base, kappa):
    case = dict(base)
    case["kappa"] = float(kappa)
    case["g"] = torch_yukawa(
        case["Xc"], case["charge_centers"], case["charge_weights"], kappa,
    )
    case["uq"] = torch_yukawa(
        case["Xq"], case["charge_centers"], case["charge_weights"], kappa,
    )
    return case


class RadialLaw(nn.Module):
    """Learn q such that G_kappa = G_0 + kappa*q(kappa*r)/(4*pi)."""

    def __init__(self, variant="product", hidden=32, n_layers=3, kappa_scale=4.0):
        super().__init__()
        if variant not in {"product", "separate"}:
            raise ValueError("variant must be product or separate")
        self.variant = variant
        self.kappa_scale = float(kappa_scale)
        layers = [nn.Linear(2, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self, radius, kappa):
        scaled = float(kappa) * radius
        if self.variant == "product":
            features = torch.stack((scaled, torch.log1p(scaled)), dim=-1)
        else:
            normalized_kappa = torch.full_like(radius, float(kappa) / self.kappa_scale)
            features = torch.stack((normalized_kappa, radius), dim=-1)
        return self.net(features).squeeze(-1)


class RadialMixture(nn.Module):
    """Completely monotone multiplier with a stable q=(m-1)/s evaluation."""

    def __init__(self, n_atoms=8):
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(n_atoms))
        rates = torch.exp(torch.linspace(-2.0, 1.5, n_atoms))
        self.raw_rates = nn.Parameter(torch.log(torch.expm1(rates)))

    def forward(self, radius, kappa):
        scaled = float(kappa) * radius
        rates = torch.nn.functional.softplus(self.raw_rates)
        weights = torch.softmax(self.weight_logits, dim=0)
        scaled_atoms = scaled.unsqueeze(-1) * rates
        regular = torch.expm1(-scaled_atoms) / scaled.unsqueeze(-1).clamp_min(1e-12)
        limits = -rates.expand_as(regular)
        q_atoms = torch.where(
            scaled.unsqueeze(-1) > 1e-12,
            regular,
            limits,
        )
        return (q_atoms * weights).sum(dim=-1)


class KappaPairwise(nn.Module):
    """Existing surface pairwise correction with kappa appended as a feature."""

    def __init__(self, hidden=64, kappa_scale=4.0):
        super().__init__()
        self.kappa_scale = float(kappa_scale)
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, kappa):
        kfeat = torch.full(
            (*features.shape[:-1], 1),
            float(kappa) / self.kappa_scale,
            dtype=features.dtype,
            device=features.device,
        )
        return self.net(torch.cat((features, kfeat), dim=-1)).squeeze(-1)


def exact_q(scaled):
    return torch.where(
        scaled.abs() > 1e-12,
        torch.expm1(-scaled) / scaled,
        -torch.ones_like(scaled),
    )


def correction_matrix(case, model, target_key, feature_key, method):
    targets = case[target_key]
    sources = case["Xc"]
    radius = torch.cdist(targets, sources)
    kappa = case["kappa"]
    if method == "exact_yukawa_split":
        q = exact_q(float(kappa) * radius)
        return float(kappa) * q / (4 * PI)
    if method in {"radial_product", "radial_separate", "radial_mixture"}:
        return float(kappa) * model(radius, kappa) / (4 * PI)
    if method == "generic_pairwise":
        return model(case[feature_key])
    if method == "generic_kappa_pairwise":
        return model(case[feature_key], kappa)
    raise ValueError(method)


def solve_case(case, model, method, return_diagnostics=False):
    matrix = case["Sbase"]
    potential = case["Pbase"]
    if method != "laplace_floor":
        matrix = matrix + correction_matrix(
            case, model, "Xc", "fc", method,
        ) * case["wc"][None, :]
        potential = potential + correction_matrix(
            case, model, "Xq", "fq", method,
        ) * case["wc"][None, :]
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    matrix = matrix + 1e-10 * identity
    density = torch.linalg.solve(matrix, case["g"])
    prediction = potential @ density
    if not return_diagnostics:
        return prediction
    residual = torch.linalg.vector_norm(matrix @ density - case["g"])
    residual = residual / torch.linalg.vector_norm(case["g"]).clamp_min(1e-30)
    condition = torch.linalg.cond(matrix)
    return prediction, float(residual), float(condition)


def train_model(method, cases, steps, seed):
    torch.manual_seed(seed)
    if method == "generic_pairwise":
        model = vk.Corr(h=64, din=11).to(cases[0]["Xc"].device)
    elif method == "generic_kappa_pairwise":
        model = KappaPairwise().to(cases[0]["Xc"].device)
    elif method == "radial_product":
        model = RadialLaw("product").to(cases[0]["Xc"].device)
    elif method == "radial_separate":
        model = RadialLaw("separate").to(cases[0]["Xc"].device)
    elif method == "radial_mixture":
        model = RadialMixture().to(cases[0]["Xc"].device)
    else:
        raise ValueError(method)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5,
    )
    order = np.random.default_rng(seed).integers(0, len(cases), size=steps)
    for step, index in enumerate(order, start=1):
        case = cases[int(index)]
        loss = vk.rel(solve_case(case, model, method), case["uq"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % max(1, steps // 5) == 0:
            print(
                f"[{method}] {step}/{steps} train rel={float(loss):.3e}",
                flush=True,
            )
    model.eval()
    return model


def radial_law_diagnostic(model, kappas):
    scaled = torch.linspace(
        0.0, 12.0, 241, device=next(model.parameters()).device,
    )
    target = exact_q(scaled)
    rows = []
    with torch.no_grad():
        for kappa in kappas:
            radius = scaled / float(kappa)
            prediction = model(radius, kappa)
            error = prediction - target
            multiplier_error = scaled * error
            rows.append({
                "kappa": float(kappa),
                "q_rmse": float(torch.sqrt(torch.mean(error.square()))),
                "q_max_abs": float(error.abs().max()),
                "multiplier_rmse": float(
                    torch.sqrt(torch.mean(multiplier_error.square()))
                ),
                "multiplier_max_abs": float(multiplier_error.abs().max()),
                "q_at_zero": float(prediction[0]),
            })
    return rows


@dataclass(frozen=True)
class EvalSpec:
    split: str
    geometry: str
    kappa: float
    seed: int


def evaluate(method, model, cases, specs, model_seed):
    rows = []
    with torch.no_grad():
        for case, spec in zip(cases, specs):
            prediction, residual, condition = solve_case(
                case, model, method, return_diagnostics=True,
            )
            rows.append({
                "model_seed": model_seed,
                "method": method,
                "params": 0 if model is None else parameter_count(model),
                "split": spec.split,
                "geometry": spec.geometry,
                "kappa": spec.kappa,
                "case_seed": spec.seed,
                "rel_l2": float(vk.rel(prediction, case["uq"])),
                "solve_residual": residual,
                "condition_number": condition,
            })
    return rows


def build_fixed_cases(dev, ntrain, neval):
    train = [
        m1.make_case("torus", 100 + i, 3.0, dev)
        for i in range(ntrain)
    ]
    cases, specs = [], []
    for geometry, seed0 in (("torus", 900), ("sphere", 950)):
        for i in range(neval):
            seed = seed0 + i
            cases.append(m1.make_case(geometry, seed, 3.0, dev))
            specs.append(EvalSpec("fixed", geometry, 3.0, seed))
    return train, cases, specs


def build_family_cases(dev, ntrain_each, neval):
    bases = (
        [m1.make_case("torus", 100 + i, 1.0, dev) for i in range(ntrain_each)]
        + [m1.make_case("sphere", 150 + i, 1.0, dev) for i in range(ntrain_each)]
    )
    train = [at_kappa(base, kappa) for base in bases for kappa in TRAIN_KAPPAS]
    cases, specs = [], []
    groups = (
        ("torus", 900),
        ("sphere", 950),
        ("two_sphere", 990),
    )
    for split, kappas in (
        ("interpolation", INTERP_KAPPAS),
        ("extrapolation", EXTRAP_KAPPAS),
    ):
        for geometry, seed0 in groups:
            for i in range(neval):
                seed = seed0 + i
                base = m1.make_case(geometry, seed, 1.0, dev)
                for kappa in kappas:
                    cases.append(at_kappa(base, kappa))
                    specs.append(EvalSpec(split, geometry, kappa, seed))
    return train, cases, specs


def write_rows(path, rows):
    _mkdir_parent(path)
    fields = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "family"], default="fixed")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--ntrain-each", type=int, default=8)
    parser.add_argument("--neval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-dir", default="")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "generic_pairwise",
            "generic_kappa_pairwise",
            "radial_separate",
            "radial_product",
            "radial_mixture",
        ),
    )
    args = parser.parse_args()
    if args.smoke:
        args.steps = 20
        args.ntrain_each = 1
        args.neval = 1

    torch.set_default_dtype(torch.float64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} mode={args.mode} seed={args.seed}", flush=True)
    if args.mode == "fixed":
        train, eval_cases, specs = build_fixed_cases(
            dev, 2 * args.ntrain_each, args.neval,
        )
        learned_methods = ("generic_pairwise", "radial_separate", "radial_product")
    else:
        train, eval_cases, specs = build_family_cases(
            dev, args.ntrain_each, args.neval,
        )
        learned_methods = (
            "generic_kappa_pairwise",
            "radial_separate",
            "radial_product",
        )
    if args.methods:
        learned_methods = tuple(args.methods)

    rows = []
    diagnostics = {}
    saved_checkpoints = {}
    for method in ("laplace_floor", "exact_yukawa_split"):
        rows.extend(evaluate(method, None, eval_cases, specs, args.seed))
    for method in learned_methods:
        model = train_model(method, train, args.steps, args.seed)
        rows.extend(evaluate(method, model, eval_cases, specs, args.seed))
        if method in {"radial_product", "radial_separate", "radial_mixture"}:
            diagnostic_kappas = (
                (3.0,) if args.mode == "fixed"
                else TRAIN_KAPPAS + INTERP_KAPPAS + EXTRAP_KAPPAS
            )
            diagnostics[method] = radial_law_diagnostic(model, diagnostic_kappas)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(
                args.save_dir, f"{args.mode}_{method}_seed{args.seed}.pt",
            )
            torch.save({
                "format_version": 1,
                "mode": args.mode,
                "method": method,
                "seed": args.seed,
                "steps": args.steps,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "params": parameter_count(model),
                "source_sha256": _sha256(__file__),
            }, path)
            saved_checkpoints[method] = path

    write_rows(args.out, rows)
    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "steps": args.steps,
        "ntrain_each": args.ntrain_each,
        "neval": args.neval,
        "device": str(dev),
        "source_sha256": _sha256(__file__),
        "m1_sha256": _sha256(m1.__file__),
        "methods": sorted(set(row["method"] for row in rows)),
        "rows": len(rows),
        "radial_law_diagnostics": diagnostics,
        "saved_checkpoints": saved_checkpoints,
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)
    print("SMOKE_OK" if args.smoke else "DONE", flush=True)


if __name__ == "__main__":
    main()
