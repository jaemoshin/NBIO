#!/usr/bin/env python
"""Parabolic heat-equation demo using NBIO as the implicit-Euler elliptic solve.

Heat equation:

    u_t = Delta u  in Omega,   u|_Gamma = g(t).

Implicit Euler with dt gives, at each step,

    (-Delta + kappa^2) u^{n+1} = kappa^2 u^n,      kappa^2 = 1/dt,
    u^{n+1}|_Gamma = g(t_{n+1}).

We solve the inhomogeneous screened-Poisson problem by the same split used elsewhere in NBIO:

    u = N[f] + u_h,
    (-Delta + kappa^2) N = f       via FFT screened Poisson on a large box,
    (-Delta + kappa^2) u_h = 0     via Yukawa BEM or Yukawa NBIO.

The manufactured exact solution is a sum of free-space heat kernels, so boundary data and interior truth are
known at every time.  The key metric is NBIO-vs-BEM inside the transient loop; exact error includes the
expected implicit-Euler and FFT-particular floors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from src.data.bie import bem_solve, boundary, sample_coeffs, sample_queries
from src.models.nbio_yukawa import NBIOYukawa
from src.utils.common import configure_torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


class HeatKernelField:
    def __init__(self, rng, n_src=4):
        self.c = rng.uniform(-0.35, 0.35, (n_src, 2))
        self.A = rng.normal(size=n_src)
        # Keep the transient smooth enough that dt=1/16 (kappa=4) is not dominated by time-discretization
        # error; this demo is about NBIO-vs-BEM inside the implicit solve, not stiff heat time stepping.
        self.s0 = rng.uniform(0.80, 1.05, n_src)

    def __call__(self, X, t):
        X = np.asarray(X)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for c, A, s0 in zip(self.c, self.A, self.s0):
            var = s0 ** 2 + 2.0 * t
            r2 = ((X - c[None, :]) ** 2).sum(axis=1)
            out += A * (s0 ** 2 / var) * np.exp(-0.5 * r2 / var)
        return out


def screened_potential_fft(f_func, kappa, L=2.8, n=384):
    """Bicubic interpolator for (-Delta + kappa^2) N = f on a periodic box."""
    xs = np.linspace(-L, L, n, endpoint=False)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    F = f_func(pts).reshape(n, n)
    kk = 2 * np.pi * np.fft.fftfreq(n, d=h)
    KX, KY = np.meshgrid(kk, kk, indexing="ij")
    denom = KX ** 2 + KY ** 2 + kappa ** 2
    N = np.fft.ifft2(np.fft.fft2(F) / denom).real

    def interp(P):
        P = np.asarray(P)
        gx = (P[:, 0] + L) / h
        gy = (P[:, 1] + L) / h
        return map_coordinates(N, np.array([gx, gy]), order=3, mode="wrap")

    return interp


def load_yukawa_nbio(path, kappa, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = NBIOYukawa(kappa=kappa, hidden=64, n_layers=3, region="interior", split=True, solve="direct")
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model


def nbio_hom_solver(model, bd, device, q_chunk=20000):
    Xb = torch.as_tensor(bd["X"], dtype=torch.float32, device=device)[None]
    Nb = torch.as_tensor(bd["normal"], dtype=torch.float32, device=device)[None]
    wb = torch.as_tensor(bd["w"], dtype=torch.float32, device=device)[None]
    curv = torch.as_tensor(bd["curv"], dtype=torch.float32, device=device)[None]

    @torch.no_grad()
    def solve(g_b, Xq):
        g_t = torch.as_tensor(g_b, dtype=torch.float32, device=device)[None]
        Xq = np.asarray(Xq)
        out = np.empty(Xq.shape[0], dtype=np.float64)
        for s in range(0, Xq.shape[0], q_chunk):
            xq = torch.as_tensor(Xq[s:s + q_chunk], dtype=torch.float32, device=device)[None]
            b = {"Xb": Xb, "Nb": Nb, "wb": wb, "curv": curv, "g": g_t, "Xq": xq}
            out[s:s + q_chunk] = model(b)[0].detach().cpu().numpy().astype(np.float64)
        return out

    return solve


def bem_hom_solver(bd, kappa):
    def solve(g_b, Xq):
        _, ev = bem_solve(bd, np.asarray(g_b, dtype=np.float64), region="interior", kappa=kappa)
        return np.asarray(ev(Xq), dtype=np.float64)
    return solve


def run_loop(name, hom_solve, bd, Xq, heat, kappa, dt, n_steps, fft_n, fft_L):
    Xb = bd["X"]
    prev_eval = lambda P: heat(P, 0.0)
    rows = []
    preds = []
    for step in range(1, n_steps + 1):
        t = step * dt

        def f_func(P, prev=prev_eval):
            return (kappa ** 2) * prev(P)

        N = screened_potential_fft(f_func, kappa=kappa, L=fft_L, n=fft_n)
        g_h = heat(Xb, t) - N(Xb)

        def current_eval(P, N=N, g_h=g_h):
            return N(P) + hom_solve(g_h, P)

        u = current_eval(Xq)
        preds.append(u.copy())
        u_exact = heat(Xq, t)
        rows.append({
            "solver": name,
            "step": step,
            "time": t,
            "rel_l2_vs_exact_heat": rel(u, u_exact),
            "max_abs_error_vs_exact_heat": float(np.max(np.abs(u - u_exact))),
        })
        prev_eval = current_eval
        print(f"[parabolic] {name:5s} step={step:02d} t={t:.3f} rel_exact={rows[-1]['rel_l2_vs_exact_heat']:.3e}", flush=True)
    return rows, prev_eval, preds


def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for solver in ("bem", "nbio"):
        r = [x for x in rows if x["solver"] == solver]
        ax[0].semilogy([x["time"] for x in r], [x["rel_l2_vs_exact_heat"] for x in r], "o-", label=solver)
    nb = [x for x in rows if x["solver"] == "nbio"]
    bm = [x for x in rows if x["solver"] == "bem"]
    diff = [abs(n["rel_l2_vs_bem_loop"] or 0.0) for n in nb]
    ax[1].semilogy([x["time"] for x in nb], diff, "s-", color="C2")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("rel-L2 vs exact heat solution")
    ax[0].set_title("Implicit-Euler heat solve")
    ax[0].grid(alpha=0.25, which="both")
    ax[0].legend()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("rel-L2 NBIO vs BEM loop")
    ax[1].set_title("Structured neural solve tracks BEM")
    ax[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def write_report(out, path):
    nbio = [r for r in out["rows"] if r["solver"] == "nbio"]
    bem = [r for r in out["rows"] if r["solver"] == "bem"]
    lines = [
        "# Parabolic heat demo",
        "",
        "NBIO is used as the homogeneous modified-Helmholtz solve inside an implicit-Euler heat equation loop.",
        "Each step solves `(-Delta + kappa^2)u^{n+1}=kappa^2 u^n` by a screened-FFT particular potential plus a Yukawa boundary solve.",
        "",
        "## Headline",
        "",
        f"- `kappa={out['meta']['kappa']}`, `dt={out['meta']['dt']}`, steps `{out['meta']['n_steps']}`.",
        f"- Final BEM rel-L2 vs exact heat solution: `{bem[-1]['rel_l2_vs_exact_heat']:.3e}`.",
        f"- Final NBIO rel-L2 vs exact heat solution: `{nbio[-1]['rel_l2_vs_exact_heat']:.3e}`.",
        f"- Final NBIO-vs-BEM loop rel-L2: `{nbio[-1]['rel_l2_vs_bem_loop']:.3e}`.",
        "",
        "The exact heat error includes the implicit-Euler and FFT particular-potential floor. The direct method-comparison metric is NBIO-vs-BEM inside the same time-stepping split.",
    ]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-b", type=int, default=128)
    ap.add_argument("--n-q", type=int, default=400)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--fft-n", type=int, default=384)
    ap.add_argument("--fft-L", type=float, default=2.8)
    ap.add_argument("--ckpt", default="results/yukawa_nbio/best.pt")
    ap.add_argument("--out", default="results/parabolic_heat.json")
    args = ap.parse_args()

    configure_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    coeffs = sample_coeffs(rng)
    bd = boundary(coeffs, args.n_b)
    Xq = sample_queries(rng, coeffs, "interior", args.n_q, margin=0.18)
    heat = HeatKernelField(rng)
    dt = 1.0 / (args.kappa ** 2)

    nbio_model = load_yukawa_nbio(args.ckpt, args.kappa, device)
    bem_rows, bem_eval, bem_preds = run_loop("bem", bem_hom_solver(bd, args.kappa), bd, Xq, heat, args.kappa, dt,
                                             args.n_steps, args.fft_n, args.fft_L)
    nbio_rows, nbio_eval, nbio_preds = run_loop("nbio", nbio_hom_solver(nbio_model, bd, device), bd, Xq, heat,
                                                args.kappa, dt, args.n_steps, args.fft_n, args.fft_L)

    for r in bem_rows:
        r["rel_l2_vs_bem_loop"] = None
    for r, up, ub in zip(nbio_rows, nbio_preds, bem_preds):
        r["rel_l2_vs_bem_loop"] = rel(up, ub)
    final_bem = bem_eval(Xq)
    final_nbio = nbio_eval(Xq)
    nbio_rows[-1]["rel_l2_vs_bem_loop"] = rel(final_nbio, final_bem)

    out = {
        "meta": {
            "seed": args.seed,
            "n_b": args.n_b,
            "n_q": args.n_q,
            "kappa": args.kappa,
            "dt": dt,
            "n_steps": args.n_steps,
            "fft_n": args.fft_n,
            "fft_L": args.fft_L,
            "device": str(device),
            "yukawa_ckpt": args.ckpt,
            "note": "Exact heat error includes implicit-Euler and FFT particular floors; NBIO-vs-BEM is the method-comparison metric.",
        },
        "rows": bem_rows + nbio_rows,
        "final": {
            "bem_rel_l2_vs_exact_heat": bem_rows[-1]["rel_l2_vs_exact_heat"],
            "nbio_rel_l2_vs_exact_heat": nbio_rows[-1]["rel_l2_vs_exact_heat"],
            "nbio_rel_l2_vs_bem_loop": nbio_rows[-1]["rel_l2_vs_bem_loop"],
        },
    }
    save_json(out, args.out)
    make_figure(out, ROOT / "figures/parabolic_heat.png")
    write_report(out, ROOT / "reports/parabolic.md")
    print("[parabolic] DONE")
    print(json.dumps(out["final"], indent=2))


if __name__ == "__main__":
    main()
