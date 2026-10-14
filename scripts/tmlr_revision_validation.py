#!/usr/bin/env python
"""Validation requested in the first TMLR review.

This script is additive: it never overwrites the original Yukawa shards or result tables. It provides:

* paired fixed-kappa learned-correction transfer data across geometry, resolution, and query margin;
* per-instance/seed transfer evaluation against correct- and wrong-kernel BIE controls;
* per-instance uncertainty for the core interior-Laplace comparison;
* finite-difference checks of gradients through g, correction parameters, geometry, and QBX;
* same-device NBIO component timings (assembly, solve, potential evaluation).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import (
    boundary,
    bem_solve,
    eval_u,
    radius_at,
    sample_charges,
    sample_queries,
)
from src.data.loader import BIEDataset, to_device
from src.models import build_model
from src.models.geom_baselines import GeoFNO
from src.models.layers import double_layer_kernel
from src.utils.common import configure_torch, get_device


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_cache" / "tmlr_revision"
RESULT_ROOT = ROOT / "results" / "tmlr_revision"
FIG_ROOT = ROOT / "figures"
KAPPA = 4.0
NBS = (64, 128, 256)
MARGINS = (0.12, 0.02)
EPS_GRID = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)

FAMILIES = {
    "in_dist": dict(n_modes=5, decay=1.6, amp=0.32, dev_cap=0.55),
    "highmode": dict(n_modes=9, decay=1.2, amp=0.45, dev_cap=0.55),
    "rough": dict(n_modes=8, decay=1.0, amp=0.60, dev_cap=0.75),
    "deform": dict(n_modes=5, decay=1.6, amp=0.60, dev_cap=0.85),
}


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def margin_tag(margin):
    return f"m{int(round(1000 * margin)):03d}"


def shard_path(family, margin, n_b):
    return DATA_ROOT / f"yukawa_k4_{family}_{margin_tag(margin)}_nb{n_b}.pt"


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ood_coeffs(rng, settings):
    n_modes = settings["n_modes"]
    k = np.arange(1, n_modes + 1)
    scale = settings["amp"] / k ** settings["decay"]
    a = rng.normal(size=n_modes) * scale
    b = rng.normal(size=n_modes) * scale
    dev = np.sum(np.abs(a) + np.abs(b))
    if dev > settings["dev_cap"]:
        a *= settings["dev_cap"] / dev
        b *= settings["dev_cap"] / dev
    return {"r0": 1.0, "a": a, "b": b}


def _stack(rows):
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}


def generate_family(family, margin, n_samples, n_q, seed):
    """Generate paired cases: coefficients, charges, and Xq are identical for all n_b."""
    ensure_dirs()
    settings = FAMILIES[family]
    rng = np.random.default_rng(seed)
    bases = []
    fine_theta = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    rejects = 0
    while len(bases) < n_samples:
        coeffs = ood_coeffs(rng, settings)
        if radius_at(coeffs, fine_theta).min() <= 0.05:
            rejects += 1
            continue
        charges = sample_charges(rng, "interior", coeffs, n_charges=6)
        Xq = sample_queries(rng, coeffs, "interior", n_q=n_q, margin=margin)
        uq = eval_u(Xq, charges, KAPPA)
        bases.append((coeffs, charges, Xq, uq))

    summary = {
        "family": family,
        "settings": settings,
        "margin": margin,
        "n_samples": n_samples,
        "n_q": n_q,
        "seed": seed,
        "rejects": rejects,
        "resolutions": {},
    }
    for n_b in NBS:
        rows = []
        for coeffs, charges, Xq, uq in bases:
            bd = boundary(coeffs, n_b)
            g = eval_u(bd["X"], charges, KAPPA)
            mu, correct_eval = bem_solve(bd, g, "interior", kappa=KAPPA)
            _, wrong_eval = bem_solve(bd, g, "interior", kappa=0.0)
            row = {
                "Xb": bd["X"],
                "Nb": bd["normal"],
                "wb": bd["w"],
                "curv": bd["curv"],
                "speed": bd["speed"],
                "g": g,
                "Xq": Xq,
                "uq": uq,
                "bem_uq": correct_eval(Xq),
                "laplace_bem_uq": wrong_eval(Xq),
                "mu": mu,
                "a": coeffs["a"],
                "b": coeffs["b"],
            }
            rows.append({k: np.asarray(v, dtype=np.float32) for k, v in row.items()})
        data = _stack(rows)
        path = shard_path(family, margin, n_b)
        torch.save(data, path)
        bem_err = instance_rel(data["bem_uq"], data["uq"]).numpy()
        wrong_err = instance_rel(data["laplace_bem_uq"], data["uq"]).numpy()
        summary["resolutions"][str(n_b)] = {
            "path": str(path.relative_to(ROOT)),
            "correct_bem_mean": float(bem_err.mean()),
            "correct_bem_median": float(np.median(bem_err)),
            "wrong_laplace_mean": float(wrong_err.mean()),
            "wrong_laplace_median": float(np.median(wrong_err)),
        }
        print(
            f"[generate] {family} margin={margin:.3f} n_b={n_b} n={n_samples} "
            f"correct={bem_err.mean():.3e} wrong={wrong_err.mean():.3e}",
            flush=True,
        )
    out = RESULT_ROOT / f"generation_{family}_{margin_tag(margin)}.json"
    save_json(summary, out)
    print(f"[generate] wrote {out}")


def instance_rel(pred, target):
    return torch.linalg.vector_norm(pred - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)


def load_checkpoint(path, device, dtype=torch.float32):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device=device, dtype=dtype).eval(), ckpt["config"]


def model_specs(args):
    return [
        ("split", 0, Path(args.split_s0) / "best.pt"),
        ("split", 1, ROOT / "results/nbio_yuk_split_s1/best.pt"),
        ("split", 2, ROOT / "results/nbio_yuk_split_s2/best.pt"),
        ("nosplit", 0, Path(args.nosplit_s0) / "best.pt"),
        ("nosplit", 1, ROOT / "results/nbio_yuk_nosplit_s1/best.pt"),
        ("nosplit", 2, ROOT / "results/nbio_yuk_nosplit_s2/best.pt"),
    ]


def bootstrap_mean_ci(values, seed=20260721, n_boot=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@torch.no_grad()
def evaluate_transfer(args):
    ensure_dirs()
    configure_torch()
    device = get_device(True)
    specs = model_specs(args)
    for _, _, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")
    models = [(kind, seed, *load_checkpoint(path, device)) for kind, seed, path in specs]
    rows = []
    for family in FAMILIES:
        for margin in MARGINS:
            for n_b in NBS:
                path = shard_path(family, margin, n_b)
                if not path.exists():
                    raise FileNotFoundError(f"missing generated shard: {path}")
                ds = BIEDataset(path)
                data = ds.d
                controls = {
                    "correct_yukawa_bie": instance_rel(data["bem_uq"], data["uq"]).numpy(),
                    "fixed_laplace_bie": instance_rel(data["laplace_bem_uq"], data["uq"]).numpy(),
                }
                for method, values in controls.items():
                    for i, value in enumerate(values):
                        rows.append({
                            "method": method,
                            "seed": -1,
                            "family": family,
                            "query_margin": margin,
                            "n_b": n_b,
                            "sample_id": i,
                            "rel_l2": float(value),
                        })
                loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
                for kind, seed, model, _ in models:
                    offset = 0
                    for batch in loader:
                        b = to_device(batch, device)
                        pred = model(b)
                        errs = instance_rel(pred, b["uq"]).detach().cpu().numpy()
                        for j, value in enumerate(errs):
                            rows.append({
                                "method": f"learned_{kind}",
                                "seed": seed,
                                "family": family,
                                "query_margin": margin,
                                "n_b": n_b,
                                "sample_id": offset + j,
                                "rel_l2": float(value),
                            })
                        offset += len(errs)
                    print(
                        f"[transfer] {kind} seed={seed} {family} margin={margin:.3f} "
                        f"n_b={n_b} done",
                        flush=True,
                    )

    raw_path = RESULT_ROOT / "learned_transfer_raw.csv"
    save_csv(rows, raw_path)
    summary_rows = summarize_transfer(rows)
    save_csv(summary_rows, RESULT_ROOT / "learned_transfer_summary.csv")
    analysis = transfer_analysis(rows, summary_rows)
    save_json(analysis, RESULT_ROOT / "learned_transfer.json")
    make_transfer_figure(summary_rows)
    print(f"[transfer] wrote {raw_path} ({len(rows)} rows)")


def summarize_transfer(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["family"], row["query_margin"], row["n_b"])].append(row)
    out = []
    for key, group in sorted(groups.items()):
        method, family, margin, n_b = key
        per_sample = defaultdict(list)
        per_seed = defaultdict(list)
        for row in group:
            per_sample[row["sample_id"]].append(row["rel_l2"])
            per_seed[row["seed"]].append(row["rel_l2"])
        sample_values = np.array([np.mean(v) for _, v in sorted(per_sample.items())])
        seed_means = np.array([np.mean(v) for _, v in sorted(per_seed.items())])
        lo, hi = bootstrap_mean_ci(sample_values)
        out.append({
            "method": method,
            "family": family,
            "query_margin": margin,
            "n_b": n_b,
            "n_instances": len(sample_values),
            "n_seeds": len(per_seed) if method.startswith("learned_") else 0,
            "mean_rel_l2": float(sample_values.mean()),
            "median_rel_l2": float(np.median(sample_values)),
            "p90_rel_l2": float(np.quantile(sample_values, 0.9)),
            "std_instances": float(sample_values.std(ddof=1)),
            "seed_mean_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "bootstrap_mean_ci_low": lo,
            "bootstrap_mean_ci_high": hi,
        })
    return out


def transfer_analysis(rows, summaries):
    lookup = {
        (r["method"], r["family"], float(r["query_margin"]), int(r["n_b"])): r
        for r in summaries
    }
    resolution_ratios = []
    for method in ("learned_split", "learned_nosplit"):
        for family in FAMILIES:
            for margin in MARGINS:
                base = lookup[(method, family, margin, 128)]["mean_rel_l2"]
                fine = lookup[(method, family, margin, 256)]["mean_rel_l2"]
                resolution_ratios.append({
                    "method": method,
                    "family": family,
                    "query_margin": margin,
                    "nb256_over_nb128": fine / max(base, 1e-30),
                })
    paired = []
    keyed = defaultdict(dict)
    for row in rows:
        if row["method"] in ("learned_split", "learned_nosplit", "fixed_laplace_bie"):
            key = (row["family"], row["query_margin"], row["n_b"], row["sample_id"], row["seed"])
            keyed[key][row["method"]] = row["rel_l2"]
    for key, vals in keyed.items():
        if "learned_split" in vals and "learned_nosplit" in vals:
            paired.append(vals["learned_split"] - vals["learned_nosplit"])
    return {
        "protocol": {
            "kappa": KAPPA,
            "n_b": list(NBS),
            "margins": list(MARGINS),
            "families": FAMILIES,
            "epsilon": "not applicable",
        },
        "resolution_ratios": resolution_ratios,
        "paired_split_minus_nosplit": {
            "n": len(paired),
            "mean": float(np.mean(paired)),
            "median": float(np.median(paired)),
            "split_win_fraction": float(np.mean(np.asarray(paired) < 0.0)),
        },
        "claim_boundary": "fixed-kappa transfer only; no kappa-family transfer is claimed",
    }


def make_transfer_figure(summary_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["correct_yukawa_bie", "fixed_laplace_bie", "learned_split", "learned_nosplit"]
    colors = {
        "correct_yukawa_bie": "black",
        "fixed_laplace_bie": "#8c564b",
        "learned_split": "#0072B2",
        "learned_nosplit": "#D55E00",
    }
    labels = {
        "correct_yukawa_bie": "correct Yukawa BIE",
        "fixed_laplace_bie": "fixed Laplace BIE",
        "learned_split": "Laplace prior + learned correction",
        "learned_nosplit": "learned BIE kernel (no split)",
    }
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.6), sharex=True, sharey=True)
    for col, family in enumerate(FAMILIES):
        for row_idx, margin in enumerate(MARGINS):
            ax = axes[row_idx, col]
            for method in methods:
                vals = [
                    r for r in summary_rows
                    if r["method"] == method and r["family"] == family
                    and abs(float(r["query_margin"]) - margin) < 1e-8
                ]
                vals.sort(key=lambda r: int(r["n_b"]))
                ax.plot(
                    [r["n_b"] for r in vals],
                    [r["mean_rel_l2"] for r in vals],
                    marker="o",
                    color=colors[method],
                    linewidth=1.8,
                    label=labels[method],
                )
                ax.fill_between(
                    [r["n_b"] for r in vals],
                    [r["bootstrap_mean_ci_low"] for r in vals],
                    [r["bootstrap_mean_ci_high"] for r in vals],
                    color=colors[method],
                    alpha=0.12,
                    linewidth=0,
                )
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{family}; margin={margin:.02f}")
            if row_idx == 1:
                ax.set_xlabel("boundary nodes")
            if col == 0:
                ax.set_ylabel("mean relative L2")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Fixed-kappa learned-correction transfer (trained at n_b=128, kappa=4)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG_ROOT / "tmlr_learned_transfer.png", dpi=180)


@torch.no_grad()
def core_uncertainty(args):
    """Audit test-instance dispersion without changing the frozen core protocol."""
    ensure_dirs()
    configure_torch()
    device = get_device(True)
    ds = BIEDataset(ROOT / "data_cache/interior_test.pt")
    rows = []

    controls = instance_rel(ds.d["bem_uq"], ds.d["uq"]).numpy()
    for sample_id, value in enumerate(controls):
        rows.append({
            "method": "classical_bem",
            "seed": -1,
            "sample_id": sample_id,
            "rel_l2": float(value),
        })

    specs = []
    for seed in range(3):
        specs.extend((
            ("nbio_full", seed, ROOT / f"results/ablM_nbio_full_s{seed}/best.pt"),
            ("nbio_nosplit", seed, ROOT / f"results/ablM_nbio_nosplit_s{seed}/best.pt"),
            ("geofno", seed, ROOT / f"results/strong_geofno_s{seed}/best.pt"),
        ))
    for method, seed, path in specs:
        if not path.exists():
            raise FileNotFoundError(path)
        if method == "geofno":
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = GeoFNO(**checkpoint["kwargs"]).to(device).eval()
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model, _ = load_checkpoint(path, device)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        offset = 0
        for batch in loader:
            b = to_device(batch, device)
            values = instance_rel(model(b), b["uq"]).cpu().numpy()
            for j, value in enumerate(values):
                rows.append({
                    "method": method,
                    "seed": seed,
                    "sample_id": offset + j,
                    "rel_l2": float(value),
                })
            offset += len(values)
        print(f"[core-uncertainty] {method} seed={seed} done", flush=True)

    summary = []
    for method in ("classical_bem", "nbio_full", "nbio_nosplit", "geofno"):
        selected = [row for row in rows if row["method"] == method]
        per_sample = defaultdict(list)
        per_seed = defaultdict(list)
        for row in selected:
            per_sample[row["sample_id"]].append(row["rel_l2"])
            per_seed[row["seed"]].append(row["rel_l2"])
        values = np.array([np.mean(v) for _, v in sorted(per_sample.items())])
        seed_means = np.array([np.mean(v) for _, v in sorted(per_seed.items())])
        lo, hi = bootstrap_mean_ci(values, seed=20260722)
        summary.append({
            "method": method,
            "n_instances": len(values),
            "n_seeds": len(per_seed) if method != "classical_bem" else 0,
            "mean_rel_l2": float(values.mean()),
            "median_rel_l2": float(np.median(values)),
            "std_instances": float(values.std(ddof=1)),
            "p90_rel_l2": float(np.quantile(values, 0.9)),
            "p99_rel_l2": float(np.quantile(values, 0.99)),
            "seed_mean_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "bootstrap_mean_ci_low": lo,
            "bootstrap_mean_ci_high": hi,
        })
    save_csv(rows, RESULT_ROOT / "core_instance_uncertainty_raw.csv")
    save_csv(summary, RESULT_ROOT / "core_instance_uncertainty_summary.csv")
    save_json({"protocol": "frozen interior_test.pt; predictions averaged by instance across seeds", "summary": summary},
              RESULT_ROOT / "core_instance_uncertainty.json")
    print(json.dumps(summary, indent=2))


def cast_batch(batch, device, dtype=torch.float64):
    return {k: v.to(device=device, dtype=dtype) for k, v in batch.items()}


def scalar_objective(model, batch, probe):
    out = model(batch)
    return (out * probe).sum()


def directional_rows(name, ad_value, fd_fn, extra=None):
    rows = []
    extra = extra or {}
    for eps in EPS_GRID:
        plus, minus = fd_fn(eps)
        fd = (plus - minus) / (2.0 * eps)
        abs_err = abs(ad_value - fd)
        rel_err = abs_err / max(abs(ad_value), abs(fd), 1e-12)
        rows.append({
            "check": name,
            "epsilon": eps,
            "autograd": ad_value,
            "finite_difference": fd,
            "abs_error": abs_err,
            "rel_error": rel_err,
            **extra,
        })
    return rows


def torch_boundary(a, b, n_b, dtype, device):
    th = torch.arange(n_b, dtype=dtype, device=device) * (2.0 * math.pi / n_b)
    k = torch.arange(1, a.numel() + 1, dtype=dtype, device=device)[:, None]
    C = torch.cos(k * th[None, :])
    S = torch.sin(k * th[None, :])
    r = 1.0 + a @ C + b @ S
    rp = (-a * k[:, 0]) @ S + (b * k[:, 0]) @ C
    rpp = (-a * k[:, 0] ** 2) @ C + (-b * k[:, 0] ** 2) @ S
    ct, st = torch.cos(th), torch.sin(th)
    X = torch.stack((r * ct, r * st), dim=1)
    xp = rp * ct - r * st
    yp = rp * st + r * ct
    speed = torch.sqrt(xp.square() + yp.square())
    normal = torch.stack((yp, -xp), dim=1) / speed[:, None]
    curv = (r.square() + 2.0 * rp.square() - r * rpp) / (r.square() + rp.square()).pow(1.5)
    w = (2.0 * math.pi / n_b) * speed
    return X, normal, w, curv


def gradient_checks(args):
    ensure_dirs()
    device = torch.device("cpu") if args.cpu else get_device(True)
    model, _ = load_checkpoint(Path(args.split_s0) / "best.pt", device, dtype=torch.float64)
    path = shard_path("in_dist", 0.02, 64)
    ds = BIEDataset(path)
    base = cast_batch(ds[0], device)
    base = {k: v.unsqueeze(0) for k, v in base.items()}
    torch.manual_seed(20260721)
    probe = torch.randn_like(base["uq"])
    probe = probe / probe.norm()
    rows = []

    # 1. Boundary data through solve and evaluation.
    g0 = base["g"].detach().clone()
    direction_g = torch.randn_like(g0)
    direction_g /= direction_g.norm()
    g = g0.clone().requires_grad_(True)
    b = dict(base); b["g"] = g
    obj = scalar_objective(model, b, probe)
    ad = float((torch.autograd.grad(obj, g)[0] * direction_g).sum())

    def fd_g(eps):
        vals = []
        for sign in (1.0, -1.0):
            bb = dict(base); bb["g"] = g0 + sign * eps * direction_g
            vals.append(float(scalar_objective(model, bb, probe).detach()))
        return vals

    rows += directional_rows("boundary_data_g", ad, fd_g, {"path": "assemble-independent RHS -> solve -> potential"})

    # 2. Learned correction parameter through assembly, solve, and evaluation.
    parameter = model.bio.smooth.net[-1].weight
    p0 = parameter.detach().clone()
    direction_p = torch.randn_like(p0)
    direction_p /= direction_p.norm()
    model.zero_grad(set_to_none=True)
    obj = scalar_objective(model, base, probe)
    grad_p = torch.autograd.grad(obj, parameter)[0]
    ad = float((grad_p * direction_p).sum())

    def fd_param(eps):
        vals = []
        with torch.no_grad():
            for sign in (1.0, -1.0):
                parameter.copy_(p0 + sign * eps * direction_p)
                vals.append(float(scalar_objective(model, base, probe)))
            parameter.copy_(p0)
        return vals

    rows += directional_rows("learned_correction_parameter", ad, fd_param, {"path": "assembly -> solve -> potential"})

    # 3. Fourier geometry parameter through all geometry-dependent tensors.
    a0 = ds.d["a"][0].to(device=device, dtype=torch.float64)
    b0 = ds.d["b"][0].to(device=device, dtype=torch.float64)
    alpha0 = a0[0].detach().clone()
    g_fixed = base["g"].detach()
    xq_fixed = base["Xq"].detach()

    def geometry_objective(alpha):
        a = a0 + (alpha - alpha0) * torch.nn.functional.one_hot(
            torch.tensor(0, device=device), num_classes=a0.numel()
        ).to(dtype=torch.float64)
        X, N, w, curv = torch_boundary(a, b0, 64, torch.float64, device)
        bb = dict(base)
        bb.update({
            "Xb": X.unsqueeze(0),
            "Nb": N.unsqueeze(0),
            "wb": w.unsqueeze(0),
            "curv": curv.unsqueeze(0),
            "g": g_fixed,
            "Xq": xq_fixed,
        })
        return scalar_objective(model, bb, probe)

    alpha = alpha0.clone().requires_grad_(True)
    ad = float(torch.autograd.grad(geometry_objective(alpha), alpha)[0])

    def fd_geometry(eps):
        return (
            float(geometry_objective(alpha0 + eps).detach()),
            float(geometry_objective(alpha0 - eps).detach()),
        )

    rows += directional_rows("geometry_fourier_a1", ad, fd_geometry, {"path": "X,N,w,curvature -> assembly -> solve -> potential"})

    # 4. Near-boundary query coordinate through the QBX branch.
    model.qbx = True
    model.qbx_kw = dict(order=10, r_fac=4.0, near_fac=2.0)
    q0 = base["Xq"][:, :1].detach().clone()
    # Use the closest available target to force the near branch.
    distances = torch.cdist(base["Xq"], base["Xb"])[0].min(dim=1).values
    qidx = int(distances.argmin())
    q0 = base["Xq"][:, qidx:qidx + 1].detach().clone()
    direction_q = torch.tensor([[[0.61, -0.79]]], device=device, dtype=torch.float64)
    direction_q /= direction_q.norm()
    q = q0.clone().requires_grad_(True)
    bq = dict(base); bq["Xq"] = q
    qprobe = torch.ones((1, 1), device=device, dtype=torch.float64)
    ad = float((torch.autograd.grad(scalar_objective(model, bq, qprobe), q)[0] * direction_q).sum())

    nearest = []
    def fd_qbx(eps):
        vals = []
        local_nearest = []
        for sign in (1.0, -1.0):
            qq = q0 + sign * eps * direction_q
            local_nearest.append(int(torch.cdist(qq, base["Xb"])[0, 0].argmin()))
            bb = dict(base); bb["Xq"] = qq
            vals.append(float(scalar_objective(model, bb, qprobe).detach()))
        nearest.append((eps, *local_nearest))
        return vals

    qbx_rows = directional_rows(
        "qbx_near_query_coordinate",
        ad,
        fd_qbx,
        {"path": "near-boundary query -> QBX potential"},
    )
    for row, (_, plus_idx, minus_idx) in zip(qbx_rows, nearest):
        row["nearest_node_plus"] = plus_idx
        row["nearest_node_minus"] = minus_idx
        row["nearest_node_stable"] = plus_idx == minus_idx
    rows += qbx_rows

    save_csv(rows, RESULT_ROOT / "gradient_checks_raw.csv")
    primary = {}
    for check in sorted({r["check"] for r in rows}):
        selected = next(r for r in rows if r["check"] == check and abs(r["epsilon"] - 1e-5) < 1e-12)
        primary[check] = selected
    result = {
        "dtype": "float64",
        "epsilon_grid": list(EPS_GRID),
        "primary_epsilon": 1e-5,
        "primary": primary,
        "pass_threshold_relative": 1e-3,
        "pass": all(
            r["rel_error"] <= 1e-3 or max(abs(r["autograd"]), abs(r["finite_difference"])) < 1e-10
            for r in primary.values()
        ),
        "qbx_piecewise_differentiability_note": "nearest-node selection must remain unchanged for a classical derivative",
    }
    save_json(result, RESULT_ROOT / "gradient_checks.json")
    print(json.dumps(result, indent=2))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(fn, device, repeats=20, warmup=5):
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / repeats


@torch.no_grad()
def runtime_breakdown(args):
    ensure_dirs()
    configure_torch()
    device = get_device(True)
    model, _ = load_checkpoint(Path(args.split_s0) / "best.pt", device)
    rows = []
    for n_b in NBS:
        ds = BIEDataset(shard_path("in_dist", 0.12, n_b))
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=args.runtime_batch_size, shuffle=False)))
        batch = to_device(batch, device)
        jump = model.jump

        def learned_assembly():
            return model.bio.assemble(batch["Xb"], batch["Nb"], batch["wb"], batch["curv"], jump)

        A = learned_assembly()
        def learned_solve():
            return model.solver(A, batch["g"], jump)

        mu = learned_solve()
        def learned_potential():
            P = model.bio.potential(batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"])
            return torch.bmm(P, mu.unsqueeze(-1)).squeeze(-1)

        def structure_assembly():
            k = double_layer_kernel(
                batch["Xb"], batch["Xb"], batch["Nb"],
                curv_s=batch["curv"], self_kernel=True,
            )
            return jump * torch.eye(n_b, device=device).unsqueeze(0) + k * batch["wb"][:, None, :]

        A0 = structure_assembly()
        def structure_solve():
            return torch.linalg.solve(A0, batch["g"].unsqueeze(-1)).squeeze(-1)

        mu0 = structure_solve()
        def structure_potential():
            P = double_layer_kernel(batch["Xq"], batch["Xb"], batch["Nb"])
            P = P * batch["wb"][:, None, :]
            return torch.bmm(P, mu0.unsqueeze(-1)).squeeze(-1)

        for method, fns in (
            ("structure_only_same_gpu", (structure_assembly, structure_solve, structure_potential)),
            ("learned_split_same_gpu", (learned_assembly, learned_solve, learned_potential)),
        ):
            times = [timed_ms(fn, device, args.runtime_repeats) for fn in fns]
            rows.append({
                "method": method,
                "device": str(device),
                "dtype": str(batch["Xb"].dtype).replace("torch.", ""),
                "batch_size": args.runtime_batch_size,
                "n_b": n_b,
                "n_q": int(batch["Xq"].shape[1]),
                "assembly_ms_batch": times[0],
                "solve_ms_batch": times[1],
                "potential_ms_batch": times[2],
                "total_ms_sample": sum(times) / args.runtime_batch_size,
            })
    save_csv(rows, RESULT_ROOT / "runtime_breakdown.csv")
    print(json.dumps(rows, indent=2))


def smoke(args):
    ensure_dirs()
    settings = FAMILIES["in_dist"]
    coeffs = ood_coeffs(np.random.default_rng(7), settings)
    assert radius_at(coeffs, np.linspace(0, 2 * np.pi, 256, endpoint=False)).min() > 0
    device = torch.device("cpu")
    if (Path(args.split_s0) / "best.pt").exists():
        model, _ = load_checkpoint(Path(args.split_s0) / "best.pt", device)
    else:
        config = {
            "model": {"name": "nbio", "kwargs": {"hidden": 8, "n_layers": 1, "split": True, "region": "interior"}}
        }
        model = build_model(config).eval()
    rng = np.random.default_rng(8)
    bd = boundary(coeffs, 16)
    charges = sample_charges(rng, "interior", coeffs, 3)
    Xq = sample_queries(rng, coeffs, "interior", 4, margin=0.1)
    sample = {
        "Xb": torch.tensor(bd["X"], dtype=torch.float32)[None],
        "Nb": torch.tensor(bd["normal"], dtype=torch.float32)[None],
        "wb": torch.tensor(bd["w"], dtype=torch.float32)[None],
        "curv": torch.tensor(bd["curv"], dtype=torch.float32)[None],
        "g": torch.tensor(eval_u(bd["X"], charges, KAPPA), dtype=torch.float32)[None],
        "Xq": torch.tensor(Xq, dtype=torch.float32)[None],
    }
    out = model(sample)
    assert out.shape == (1, 4) and torch.isfinite(out).all()
    print("[smoke] PASS", tuple(out.shape), float(out.norm()))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("generate", "transfer", "core_uncertainty", "gradients", "runtime", "smoke"))
    ap.add_argument("--family", choices=tuple(FAMILIES), default="in_dist")
    ap.add_argument("--margin", type=float, default=0.12, choices=MARGINS)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-q", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--split-s0", default=str(RESULT_ROOT / "nbio_yuk_split_s0"))
    ap.add_argument("--nosplit-s0", default=str(RESULT_ROOT / "nbio_yuk_nosplit_s0"))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--runtime-batch-size", type=int, default=8)
    ap.add_argument("--runtime-repeats", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "generate":
        family_index = list(FAMILIES).index(args.family)
        margin_index = list(MARGINS).index(args.margin)
        generate_family(
            args.family,
            args.margin,
            args.n_samples,
            args.n_q,
            args.seed + 1000 * family_index + 100 * margin_index,
        )
    elif args.mode == "transfer":
        evaluate_transfer(args)
    elif args.mode == "core_uncertainty":
        core_uncertainty(args)
    elif args.mode == "gradients":
        gradient_checks(args)
    elif args.mode == "runtime":
        runtime_breakdown(args)
    else:
        smoke(args)


if __name__ == "__main__":
    main()
