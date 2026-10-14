#!/usr/bin/env python
"""Test continuous-time and parabolic-boundary-integral NBIO variants.

This is a boundary-driven heat-equation benchmark with zero initial condition:

    u_t = Delta u in Omega,   u|_Gamma = g(t),   u(x,0)=0.

Data are generated from a smooth causal heat double-layer density

    u(x,t) = int_0^t int_Gamma dH/dn_y(x-y,t-s) mu(y,s) ds_y ds,

with the boundary trace jump term included, so the exact target at arbitrary query times is known by
quadrature and the boundary solve is second-kind.  The experiment evaluates:

  1. `parabolic_double_layer_nbio`: zero-parameter second-kind causal solve, then arbitrary `(x,t)` potential.
  2. `continuous_time_double_layer`: learned density + analytic heat double layer, query-continuous.
  3. `continuous_time_deepset`: direct black-box boundary-time set baseline.

The goal is not to claim this is a production parabolic solver.  It is a controlled test of whether NBIO can
be extended from "elliptic solve per time step" to a one-shot causal space-time boundary operator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.bie import boundary, sample_coeffs, sample_queries
from src.models.parabolic_nbio import (
    ContinuousTimeDeepSet,
    ContinuousTimeDoubleLayer,
    ContinuousTimeHeatLayer,
    ParabolicBIENBIO,
    ParabolicDoubleLayerNBIO,
)
from src.utils.common import configure_torch, count_params, get_device, save_json, set_seed

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rel_np(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(np.asarray(b)) + 1e-30))


def rel_torch(a, b):
    return torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-30)


def heat_kernel_np(Xt, tq, Xs, ts):
    tau = tq[:, None] - ts[None, :]
    mask = tau > 1e-12
    tau_safe = np.maximum(tau, 1e-12)
    r2 = ((Xt[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    H = np.exp(-r2 / (4.0 * tau_safe)) / (4.0 * np.pi * tau_safe)
    return np.where(mask, H, 0.0)


def heat_double_layer_kernel_np(Xt, tq, Xs, ts, Ns):
    tau = tq[:, None] - ts[None, :]
    mask = tau > 1e-12
    tau_safe = np.maximum(tau, 1e-12)
    diff = Xt[:, None, :] - Xs[None, :, :]
    r2 = (diff ** 2).sum(-1)
    H = np.exp(-r2 / (4.0 * tau_safe)) / (4.0 * np.pi * tau_safe)
    num = (diff * Ns[None, :, :]).sum(-1)
    K = num * H / (2.0 * tau_safe)
    return np.where(mask, K, 0.0)


def flatten_sources(Xb, wb, mu, t_src, dt):
    nt, nb = mu.shape
    Xs = np.broadcast_to(Xb[None, :, :], (nt, nb, 2)).reshape(nt * nb, 2)
    ts = np.broadcast_to(t_src[:, None], (nt, nb)).reshape(nt * nb)
    weights = np.broadcast_to(wb[None, :], (nt, nb)).reshape(nt * nb) * dt
    muf = mu.reshape(nt * nb)
    return Xs, ts, weights, muf


def heat_potential_np(Xt, tq, Xb, wb, mu, t_src, dt):
    Xs, ts, weights, muf = flatten_sources(Xb, wb, mu, t_src, dt)
    return heat_kernel_np(Xt, tq, Xs, ts) @ (muf * weights)


def heat_double_potential_np(Xt, tq, Xb, Nb, wb, mu, t_src, dt):
    Xs, ts, weights, muf = flatten_sources(Xb, wb, mu, t_src, dt)
    nt, nb = mu.shape
    Ns = np.broadcast_to(Nb[None, :, :], (nt, nb, 2)).reshape(nt * nb, 2)
    return heat_double_layer_kernel_np(Xt, tq, Xs, ts, Ns) @ (muf * weights)


def smooth_density(rng, theta, t_src, T, n_spatial=4, n_temporal=3):
    th = theta[None, :]
    tt = (t_src[:, None] / T)
    mu = np.zeros((len(t_src), len(theta)), dtype=np.float64)
    for _ in range(7):
        k = int(rng.integers(1, n_spatial + 1))
        ell = int(rng.integers(1, n_temporal + 1))
        amp = rng.normal() / (k ** 1.2 * ell ** 1.1)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        spatial = np.cos(k * th + p1) + 0.7 * np.sin((k + 1) * th - p1)
        temporal = np.sin(math.pi * ell * tt + p2) + 0.5 * np.cos(math.pi * (ell + 1) * tt - p2)
        mu += amp * temporal * spatial
    ramp = np.sin(0.5 * math.pi * np.clip(t_src / T, 0.0, 1.0))[:, None]
    mu *= ramp
    mu -= mu.mean()
    mu /= (np.std(mu) + 1e-8)
    return mu


def make_sample(rng, n_b, n_t, n_q, T, geometry="star", layer_kind="double", jump=-0.5):
    coeffs = sample_coeffs(rng, n_modes=5, amp=(0.0 if geometry == "circle" else 0.26))
    bd = boundary(coeffs, n_b)
    dt = T / n_t
    t_src = (np.arange(n_t, dtype=np.float64) + 0.5) * dt
    t_obs = (np.arange(n_t, dtype=np.float64) + 1.0) * dt
    mu = smooth_density(rng, bd["theta"], t_src, T)

    # Boundary observations are at right endpoints, source intervals at midpoints.  This avoids the tau=0
    # endpoint singularity while retaining a causal Volterra system.
    g = np.zeros((n_t, n_b), dtype=np.float64)
    for n, tn in enumerate(t_obs):
        tq = np.full(n_b, tn)
        if layer_kind == "single":
            g[n] = heat_potential_np(bd["X"], tq, bd["X"], bd["w"], mu, t_src, dt)
        else:
            g[n] = heat_double_potential_np(bd["X"], tq, bd["X"], bd["normal"], bd["w"], mu, t_src, dt)
            g[n] += jump * mu[n]

    Xq = sample_queries(rng, coeffs, "interior", n_q, margin=0.16)
    # Off-grid query times: not tied to the boundary observation grid.
    tq = rng.uniform(0.75 * dt, T, n_q)
    if layer_kind == "single":
        uq = heat_potential_np(Xq, tq, bd["X"], bd["w"], mu, t_src, dt)
    else:
        uq = heat_double_potential_np(Xq, tq, bd["X"], bd["normal"], bd["w"], mu, t_src, dt)
    return {
        # Keep float64 in the generated dataset so both first-kind diagnostics and second-kind reference
        # solves can be evaluated cleanly. Learned models are cast to float32 in `add_constants`.
        "Xb": bd["X"].astype(np.float64),
        "Nb": bd["normal"].astype(np.float64),
        "wb": bd["w"].astype(np.float64),
        "theta": bd["theta"].astype(np.float64),
        "g": g.astype(np.float64),
        "mu": mu.astype(np.float64),
        "Xq": Xq.astype(np.float64),
        "tq": tq.astype(np.float64),
        "uq": uq.astype(np.float64),
        "t_obs": t_obs.astype(np.float64),
        "t_src": t_src.astype(np.float64),
    }


class HeatLayerDataset(Dataset):
    def __init__(self, samples: List[Dict[str, np.ndarray]], T: float, dt: float):
        self.samples = samples
        self.T = float(T)
        self.dt = float(dt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.as_tensor(v) for k, v in self.samples[idx].items()}


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def add_constants(batch, T, dt, device, dtype=torch.float32):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
    # t_obs/t_src are identical across samples; keep a 1D copy for model code.
    out["t_obs"] = out["t_obs"][0]
    out["t_src"] = out["t_src"][0]
    out["T"] = float(T)
    out["dt"] = float(dt)
    return out


def make_splits(args):
    rng = np.random.default_rng(args.seed)
    T = args.T
    dt = T / args.n_t
    def make_many(n, split):
        return [make_sample(rng, args.n_b, args.n_t, args.n_q, T, args.geometry,
                            layer_kind=args.layer_kind) for _ in range(n)]
    return {
        "train": HeatLayerDataset(make_many(args.n_train, "train"), T, dt),
        "val": HeatLayerDataset(make_many(args.n_val, "val"), T, dt),
        "test": HeatLayerDataset(make_many(args.n_test, "test"), T, dt),
        "T": T,
        "dt": dt,
    }


@torch.no_grad()
def evaluate_model(model, dataset, device, T, dt, batch_size, dtype=torch.float32, return_density=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    rows = []
    total, n = 0.0, 0
    mu_total = 0.0
    bin_num2 = [0.0, 0.0, 0.0, 0.0]
    bin_den2 = [0.0, 0.0, 0.0, 0.0]
    for batch in loader:
        b = add_constants(batch, T, dt, device, dtype=dtype)
        if return_density:
            pred, mu = model(b, return_density=True)
            mu_err = rel_torch(mu, b["mu"]).item()
            mu_total += mu_err * pred.shape[0]
        else:
            pred = model(b)
        err = rel_torch(pred, b["uq"]).item()
        total += err * pred.shape[0]
        n += pred.shape[0]
        rows.append(err)
        diff = pred - b["uq"]
        for j in range(4):
            lo = j * T / 4.0
            hi = (j + 1) * T / 4.0 + (1e-8 if j == 3 else 0.0)
            mask = (b["tq"] >= lo) & (b["tq"] < hi)
            if mask.any():
                bin_num2[j] += float((diff[mask] ** 2).sum().detach().cpu())
                bin_den2[j] += float((b["uq"][mask] ** 2).sum().detach().cpu())
    out = {"mean_rel_l2": total / max(n, 1), "batch_rel_l2": rows}
    out["time_bin_rel_l2"] = [
        math.sqrt(bin_num2[j] / max(bin_den2[j], 1e-30)) if bin_den2[j] > 0.0 else None
        for j in range(4)
    ]
    if return_density:
        out["mean_density_rel_l2"] = mu_total / max(n, 1)
    return out


def train_model(name, model, splits, args, device):
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    hist = []
    best = {"val": float("inf"), "epoch": -1, "state": None}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            b = add_constants(batch, splits["T"], splits["dt"], device)
            opt.zero_grad(set_to_none=True)
            pred = model(b)
            loss = rel_torch(pred, b["uq"])
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            total += float(loss.detach().cpu()) * pred.shape[0]
            n += pred.shape[0]
        sched.step()
        train_loss = total / max(n, 1)
        val = evaluate_model(model, splits["val"], device, splits["T"], splits["dt"], args.batch_size)["mean_rel_l2"]
        hist.append({"model": name, "epoch": ep, "train_rel_l2": train_loss, "val_rel_l2": val,
                     "lr": float(sched.get_last_lr()[0])})
        if val < best["val"]:
            best = {"val": val, "epoch": ep, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            print(f"[ct] {name:24s} ep={ep:04d} train={train_loss:.3e} val={val:.3e} "
                  f"best={best['val']:.3e} wall={(time.time()-t0)/60:.1f}m", flush=True)
    model.load_state_dict(best["state"])
    return hist, best


def write_curve_csv(histories: Dict[str, List[Dict]], path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [r for h in histories.values() for r in h]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "epoch", "train_rel_l2", "val_rel_l2", "lr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = out["methods"]
    names = list(methods)
    vals = [methods[k]["test"]["mean_rel_l2"] for k in names]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    palette = ["#2f6f9f", "#5aa469", "#8f63a2", "#b85c38", "#6b7280"]
    ax[0].bar(range(len(names)), vals, color=[palette[i % len(palette)] for i in range(len(names))])
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(names)), names, rotation=18, ha="right")
    ax[0].set_ylabel("test rel-L2")
    ax[0].set_title("Arbitrary-time heat queries")
    ax[0].grid(axis="y", which="both", alpha=0.25)
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.25, f"{v:.1e}", ha="center", va="bottom", fontsize=8)

    for name, rows in out["training_curves"].items():
        if rows:
            ax[1].semilogy([r["epoch"] for r in rows], [r["val_rel_l2"] for r in rows], label=name)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val rel-L2")
    ax[1].set_title("Learned continuous-time prototypes")
    ax[1].grid(alpha=0.25, which="both")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def write_report(out, path):
    m = out["methods"]
    lines = [
        "# Continuous-time and parabolic boundary-integral NBIO",
        "",
        "This test uses a zero-initial, boundary-driven heat equation generated by a smooth causal heat",
        f"{out['meta']['layer_kind']}-layer density. Query times are sampled continuously off the boundary observation grid.",
        "",
        "## Results",
        "",
        "| method | params | precision | arbitrary time? | rel-L2 | density rel-L2 | note |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for name, r in m.items():
        dens = r["test"].get("mean_density_rel_l2")
        dens_s = f"`{dens:.3e}`" if dens is not None else "-"
        precision = r.get("precision", "-")
        lines.append(f"| {name} | `{r['params']}` | {precision} | yes | `{r['test']['mean_rel_l2']:.3e}` | {dens_s} | {r['note']} |")
    lines += [
        "",
        "## Time-bin errors",
        "",
        "| method | t in [0,T/4) | [T/4,T/2) | [T/2,3T/4) | [3T/4,T] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in m.items():
        bins = r["test"].get("time_bin_rel_l2") or []
        vals = [f"`{x:.3e}`" if x is not None else "-" for x in bins]
        while len(vals) < 4:
            vals.append("-")
        lines.append(f"| {name} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The second-kind parabolic BIE/NBIO is the clean arbitrary-time extension: it solves the causal",
        "space-time boundary system for the density and then evaluates the heat double-layer potential at any",
        "requested `(x,t)`. This is closer to a parabolic boundary-integral solver than to a rollout FNO.",
        "",
        "The earlier first-kind single-layer prototype required float64 to avoid density-error amplification.",
        "The double-layer version fixes that caveat: it has a jump term and is evaluated directly in float32 as",
        "well as float64.",
        "",
        "The learned continuous-time layer model has the same arbitrary-time query contract but replaces",
        "the causal solve by a learned density map. It is useful as a prototype, but the current result should",
        "not be described as solver-grade unless its measured error supports that.",
        "",
        "The correct paper framing is: NBIO can be extended beyond steady elliptic solves through second-kind",
        "causal space-time boundary integrals. The present test is a controlled proof of concept, not yet a",
        "full general-purpose parabolic neural operator.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def write_paper_insert(out, path):
    bie = out["methods"]["parabolic_double_layer_nbio_fp32"]["test"]["mean_rel_l2"]
    bie64 = out["methods"]["parabolic_double_layer_nbio_fp64"]["test"]["mean_rel_l2"]
    layer = out["methods"]["continuous_time_double_layer"]["test"]["mean_rel_l2"]
    deepset = out["methods"]["continuous_time_deepset"]["test"]["mean_rel_l2"]
    text = f"""# Continuous-Time Parabolic NBIO Insert

To test whether NBIO is restricted to steady-state maps, we built a causal space-time heat-layer benchmark.
For zero-initial heat flow driven by time-dependent Dirichlet data, the solution is represented by a
second-kind causal heat double layer,

```text
u(x,t) = int_0^t int_Gamma dH/dn_y(x-y,t-s) mu(y,s) ds_y ds,
g(y,t) = -1/2 mu(y,t) + causal history.
```

The structure-only parabolic double-layer NBIO solves the resulting second-kind causal Volterra boundary
system and evaluates `u(x,t)` at off-grid query times with rel-L2 `{bie:.3e}` in float32
(`{bie64:.3e}` in float64). A learned continuous-time double-layer prototype, which keeps the heat potential
but learns the density map, reaches `{layer:.3e}`; a black-box continuous-time DeepSet reaches `{deepset:.3e}`.

This supports a scoped claim: NBIO can be extended from elliptic steady-state solves to causal space-time
boundary-integral operators with arbitrary-time queries. The important fix over the first-kind prototype is
the second-kind jump term: the solver-grade result is no longer dependent on a fragile first-kind inversion.
The learned density model is preliminary; the solver-grade result comes from the parabolic boundary-integral
structure, not from a generic trajectory network.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--n-b", type=int, default=48)
    ap.add_argument("--n-t", type=int, default=12)
    ap.add_argument("--n-q", type=int, default=160)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--geometry", choices=["star", "circle"], default="star")
    ap.add_argument("--layer-kind", choices=["double", "single"], default="double")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="Force CPU, useful for local smoke tests.")
    ap.add_argument("--out", default="results/parabolic_ct_nbio.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_train = min(args.n_train, 8)
        args.n_val = min(args.n_val, 4)
        args.n_test = min(args.n_test, 4)
        args.n_b = min(args.n_b, 24)
        args.n_t = min(args.n_t, 6)
        args.n_q = min(args.n_q, 32)
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 4)

    set_seed(args.seed)
    configure_torch()
    device = get_device(not args.cpu)
    print(f"[ct] device={device} seed={args.seed} smoke={args.smoke}", flush=True)
    splits = make_splits(args)

    methods = {}
    histories = {}

    if args.layer_kind == "double":
        # Headline: second-kind causal double-layer NBIO.  Report both float32 and float64; the float32 row is
        # the caveat-fixing result.
        bie32 = ParabolicDoubleLayerNBIO().to(device=device, dtype=torch.float32)
        methods["parabolic_double_layer_nbio_fp32"] = {
            "params": count_params(bie32),
            "precision": "float32",
            "test": evaluate_model(bie32, splits["test"], device, splits["T"], splits["dt"], args.batch_size,
                                   dtype=torch.float32, return_density=True),
            "note": "zero-param second-kind causal heat double-layer solve",
        }
        bie64 = ParabolicDoubleLayerNBIO().to(device=device, dtype=torch.float64)
        methods["parabolic_double_layer_nbio_fp64"] = {
            "params": count_params(bie64),
            "precision": "float64",
            "test": evaluate_model(bie64, splits["test"], device, splits["T"], splits["dt"], args.batch_size,
                                   dtype=torch.float64, return_density=True),
            "note": "double-precision reference for the same second-kind solve",
        }
        layer = ContinuousTimeDoubleLayer(hidden=args.hidden).to(device)
        layer_name = "continuous_time_double_layer"
    else:
        # Diagnostic path retained for reproducing the original first-kind caveat.
        bie = ParabolicBIENBIO().to(device=device, dtype=torch.float64)
        methods["parabolic_single_layer_nbio_fp64"] = {
            "params": count_params(bie),
            "precision": "float64",
            "test": evaluate_model(bie, splits["test"], device, splits["T"], splits["dt"], args.batch_size,
                                   dtype=torch.float64, return_density=True),
            "note": "first-kind causal heat single-layer diagnostic",
        }
        layer = ContinuousTimeHeatLayer(hidden=args.hidden).to(device)
        layer_name = "continuous_time_heat_layer"

    # Train learned continuous-time density approximation using the same analytic layer as the dataset.
    h, best = train_model(layer_name, layer, splits, args, device)
    histories[layer_name] = h
    methods[layer_name] = {
        "params": count_params(layer),
        "precision": "float32",
        "best_val": best["val"],
        "best_epoch": best["epoch"],
        "test": evaluate_model(layer, splits["test"], device, splits["T"], splits["dt"], args.batch_size,
                               return_density=True),
        "note": "learned density map plus analytic layer; no explicit causal solve",
    }

    # Train black-box direct continuous-time baseline.
    deepset = ContinuousTimeDeepSet(hidden=max(args.hidden, 160)).to(device)
    h, best = train_model("continuous_time_deepset", deepset, splits, args, device)
    histories["continuous_time_deepset"] = h
    methods["continuous_time_deepset"] = {
        "params": count_params(deepset),
        "precision": "float32",
        "best_val": best["val"],
        "best_epoch": best["epoch"],
        "test": evaluate_model(deepset, splits["test"], device, splits["T"], splits["dt"], args.batch_size),
        "note": "black-box boundary-time set model; no heat-kernel or causal BIE structure",
    }

    out = {
        "meta": vars(args),
        "device": str(device),
        "dt": splits["dt"],
        "methods": methods,
        "training_curves": histories,
    }
    save_json(out, args.out)
    write_curve_csv(histories, ROOT / "results/parabolic_ct_nbio_training_curve.csv")
    make_figure(out, ROOT / "figures/parabolic_ct_nbio.png")
    write_report(out, ROOT / "reports/parabolic_ct_nbio.md")
    write_paper_insert(out, ROOT / "paper/nbio_continuous_time_insert.md")
    print("[ct] DONE")
    print(json.dumps({k: v["test"]["mean_rel_l2"] for k, v in methods.items()}, indent=2))


if __name__ == "__main__":
    main()
