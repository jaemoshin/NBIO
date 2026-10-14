#!/usr/bin/env python
"""Semilinear elliptic BVP via Picard iteration with a learned linear solve in the loop.

PROBLEM (semilinear reaction-diffusion, no Green's function for the full nonlinear operator):
    -Delta u + kappa u^3 = f   in Omega (smooth star domain),   u = g on dOmega.

MANUFACTURED solution: u* = sum of point-charge log-potentials with charges OUTSIDE Omega, so u* is
HARMONIC (-Delta u* = 0). Set f := kappa (u*)^3 and g := u*|_dOmega. Then u* is the exact solution of the
semilinear BVP (it is the Picard fixed point: -Delta u* = f - kappa u*^3 = 0, trivially satisfied).

PICARD (iterative linearization): u_{n+1} solves the LINEAR inhomogeneous Laplace Dirichlet problem
    -Delta u_{n+1} = f - kappa u_n^3  =: f_n   in Omega,   u_{n+1} = g on dOmega.
Each linear step is solved by the EXISTING particular-solution split:
    u_{n+1} = N_n + u_h,  with -Delta N_n = f_n (volume_potential_fft), u_h HARMONIC with data g - N_n|_dOmega.
The HARMONIC piece u_h is solved by THREE interchangeable linear solvers, to show that per-step accuracy
of the linear solve sets the achievable accuracy of the nonlinear solve:
    (i)   classical Nystrom BEM (reference)         -- bem_solve
    (ii)  NBIO (proposed structured neural operator) -- results/nbio_s0/best.pt, zero-shot (linear-only train)
    (iii) DeepONet (black-box neural operator)       -- results/deeponet_s0/best.pt, zero-shot

This addresses the paper's open question: NBIO's novelty must show up where a learned/structured operator
is the only viable option. Here there is NO kernel for the full nonlinear operator; only the LINEAR step
has structure, and NBIO can act as a drop-in solver-grade linear solve inside the outer nonlinear iteration.

Outputs: results/nonlinear_bvp.json, figures/nonlinear_bvp.png. Verifies everything against u*.

STRICT: creates only new files; imports existing functions; does not modify any existing file.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.common import configure_torch
from src.data.bie import (sample_coeffs, boundary, sample_charges, eval_u, sample_queries, bem_solve,
                          radius_at)
from src.data.volume_source import volume_potential_fft
from src.models import build_model

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ------------------------------------------------------------------ config
GEOM_SEED = 0          # fixed star geometry reused every Picard iteration
N_B = 256              # boundary nodes
N_Q = 400              # interior query points
N_CHARGES = 6
KAPPAS = [0.1, 0.3, 0.6, 1.0]
MAX_ITERS = 30
TOL_STEP = 1e-10       # stop when ||u_{n+1}-u_n|| / ||u_n|| < TOL_STEP
FFT_L = 2.8            # FFT box half-width (must contain Omega; r0~1 here)
FFT_N = 768            # FFT grid for the periodic Newtonian-potential solve
REP_KAPPA = 0.3        # representative kappa for the left convergence panel
WIN_LO = 1.02          # source window: 1 for r < WIN_LO*r_bd(theta)
WIN_BAND = 0.20        # taper width as a fraction of r_bd(theta); 0 at (WIN_LO+WIN_BAND)*r_bd


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def source_window(coeffs, X):
    """Smooth radial window: 1 strictly inside Omega, C^1-taper to 0 in a thin band just outside dOmega.

    WHY: volume_potential_fft solves the PERIODIC Poisson eq -Delta N = f - mean_box(f); the manufactured
    source f = kappa*(u*)^3 BLOWS UP near the point charges, which sit just OUTSIDE Omega but INSIDE the FFT
    box. Those out-of-domain values dominate mean_box(f) (~18% of std), and that constant offset does NOT
    cancel in the u = N + u_h split -> caps Picard accuracy at ~2e-3 even with the exact BEM solver. Making
    the source compactly supported inside Omega removes the contamination (the source value OUTSIDE Omega is
    irrelevant to the interior BVP; only -Delta N = f INSIDE Omega matters, and that is preserved here)."""
    ang = np.arctan2(X[:, 1], X[:, 0])
    rb = radius_at(coeffs, ang).ravel()
    rr = np.linalg.norm(X, axis=1)
    t = np.clip((rr - WIN_LO * rb) / (WIN_BAND * rb), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))      # 1 (r<=WIN_LO rb) -> 0 (r>=(WIN_LO+WIN_BAND) rb)


# ------------------------------------------------------------------ model loading
def load_torch_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model


def torch_uh_solver(model, bd, device, q_chunk=20000):
    """Return uh(boundary_data g_b, Xq) -> u_h at Xq, using a learned operator with the NBIO batch dict.
    g_b: (n_b,) boundary Dirichlet data for the HARMONIC subproblem. Xq: (Q,2).
    Query points are chunked so the FFT-grid evaluations (~1e6 pts) do not exhaust GPU memory; the
    boundary density solve is identical for every chunk (depends only on g_b), so chunking is exact."""
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
            batch = {"Xb": Xb, "Nb": Nb, "wb": wb, "curv": curv, "g": g_t, "Xq": xq}
            out[s:s + q_chunk] = model(batch)[0].cpu().numpy().astype(np.float64)
        return out
    return solve


def bem_uh_solver(bd):
    """Classical Nystrom BEM reference: uh(g_b, Xq) -> harmonic u_h at Xq (interior double-layer)."""
    def solve(g_b, Xq):
        _, evaluate = bem_solve(bd, np.asarray(g_b, dtype=np.float64), "interior")
        return np.asarray(evaluate(Xq), dtype=np.float64)
    return solve


# ------------------------------------------------------------------ Picard driver
def picard_solve(uh_solve, bd, coeffs, charges, kappa, Xq, u_star_q,
                 u_star_b, max_iters=MAX_ITERS, tol=TOL_STEP):
    """Run Picard iteration for -Delta u + kappa u^3 = f, u=g, with a given harmonic linear solver `uh_solve`.

    The current iterate u_n is held as a python closure u_n_func(X)->(P,) so volume_potential_fft can sample
    f_n on its grid:  f_n(X) = kappa*u*(X)^3 - kappa*u_n(X)^3   (since f = kappa u*^3).
    u_0 = 0  =>  f_0 = f. Returns convergence curve (rel-L2 of u_n vs u* at Xq), step-norm curve, iters.
    """
    g_b = u_star_b                                   # Dirichlet data g = u*|_dOmega (n_b,)
    Xb = bd["X"]

    # u_n as a callable. Start with u_0 = 0.
    u_n_func = lambda X: np.zeros(X.shape[0], dtype=np.float64)

    curve = []          # rel-L2(u_n, u*) at query points, per iterate produced
    step_curve = []     # ||u_{n+1}-u_n|| / ||u_n||
    u_prev_q = np.zeros(Xq.shape[0], dtype=np.float64)
    iters_done = 0
    diverged = False

    for it in range(max_iters):
        # f_n(X) = window(X) * [ kappa*(u*(X))^3 - kappa*(u_n(X))^3 ]
        # window = 1 inside Omega, tapered to 0 just outside, so the FFT source is compactly supported in
        # Omega (removes the periodic mean-subtraction contamination from the point charges; see source_window).
        def f_func(X, _u_n=u_n_func):
            us = eval_u(X, charges, 0.0)
            un = _u_n(X)
            return source_window(coeffs, X) * (kappa * us ** 3 - kappa * un ** 3)

        # particular solution N_n: -Delta N_n = f_n  (FFT Newtonian potential, bicubic interpolator)
        N_interp = volume_potential_fft(f_func, L=FFT_L, n=FFT_N)

        # harmonic correction data on the boundary, then solve harmonic BVP for u_h at Xq AND at Xb
        N_b = N_interp(Xb)
        g_h = g_b - N_b
        u_h_q = uh_solve(g_h, Xq)                    # (Q,)
        N_q = N_interp(Xq)
        u_next_q = N_q + u_h_q                        # u_{n+1}(Xq)

        # build the next iterate as a callable: u_{n+1}(X) = N_n(X) + u_h(X)
        # (we must re-solve the harmonic part at arbitrary X; capture solver + this iteration's data)
        def make_next(_N_interp=N_interp, _g_h=g_h):
            def u_next_func(X):
                uh = uh_solve(_g_h, X)
                return _N_interp(X) + uh
            return u_next_func
        u_n_func = make_next()

        err = rel_l2(u_next_q, u_star_q)
        curve.append(err)
        sc = float(np.linalg.norm(u_next_q - u_prev_q) / (np.linalg.norm(u_prev_q) + 1e-30)) if it > 0 else float("inf")
        step_curve.append(sc)
        iters_done = it + 1

        if not np.isfinite(err) or err > 1e3:
            diverged = True
            break
        if it > 0 and sc < tol:
            break
        u_prev_q = u_next_q

    final_err = curve[-1]
    return {
        "convergence_rel_l2": curve,
        "step_rel": [None if (np.isinf(s)) else s for s in step_curve],
        "iters": iters_done,
        "final_rel_l2": final_err,
        "diverged": bool(diverged),
        "best_rel_l2": float(np.min(curve)),
    }


# ------------------------------------------------------------------ single-shot linear-problem accuracy
def single_shot_linear_accuracy(solvers, bd, charges_lin, Xq):
    """Per-step linear solver accuracy on a pure linear Laplace Dirichlet problem (no source), for context.
    Uses the SAME geometry, an independent set of harmonic charges -> exact harmonic u_lin; g = u_lin|_dOmega.
    """
    g_b = eval_u(bd["X"], charges_lin, 0.0)
    u_lin_q = eval_u(Xq, charges_lin, 0.0)
    out = {}
    for name, solve in solvers.items():
        uq = solve(g_b, Xq)
        out[name] = rel_l2(uq, u_lin_q)
    return out


# ------------------------------------------------------------------ main
def main():
    configure_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nonlinear_bvp] device={device}")

    # ---- fixed geometry + interior queries (reused every iteration / kappa / solver) ----
    rng = np.random.default_rng(GEOM_SEED)
    coeffs = sample_coeffs(rng)
    bd = boundary(coeffs, N_B)
    charges = sample_charges(rng, "interior", coeffs, N_CHARGES)   # charges OUTSIDE Omega -> u* harmonic
    Xq = sample_queries(rng, coeffs, "interior", N_Q)
    charges_lin = sample_charges(rng, "interior", coeffs, N_CHARGES)  # independent harmonic field for context

    # manufactured exact solution u* (harmonic) on queries and boundary
    u_star_q = eval_u(Xq, charges, 0.0)
    u_star_b = eval_u(bd["X"], charges, 0.0)
    print(f"[nonlinear_bvp] |u*|_q range [{u_star_q.min():.3f},{u_star_q.max():.3f}], n_b={N_B}, n_q={N_Q}")

    # sanity: u* must lie strictly inside the FFT box
    rmax = float(np.linalg.norm(bd["X"], axis=1).max())
    assert rmax < FFT_L, f"domain radius {rmax} >= FFT box {FFT_L}"

    # ---- load solvers ----
    nbio = load_torch_model(RESULTS / "nbio_s0" / "best.pt", device)
    deeponet = load_torch_model(RESULTS / "deeponet_s0" / "best.pt", device)
    solvers = {
        "bem": bem_uh_solver(bd),
        "nbio": torch_uh_solver(nbio, bd, device),
        "deeponet": torch_uh_solver(deeponet, bd, device),
    }

    # ---- per-step LINEAR solver accuracy (context) ----
    lin_acc = single_shot_linear_accuracy(solvers, bd, charges_lin, Xq)
    print(f"[nonlinear_bvp] single-shot linear rel-L2: {lin_acc}")

    # ---- sweep kappa x solver ----
    results = {
        "meta": {
            "problem": "-Delta u + kappa u^3 = f, u=g; manufactured harmonic u* (charges outside Omega)",
            "geom_seed": GEOM_SEED, "n_b": N_B, "n_q": N_Q, "n_charges": N_CHARGES,
            "kappas": KAPPAS, "max_iters": MAX_ITERS, "tol_step": TOL_STEP,
            "fft_L": FFT_L, "fft_n": FFT_N, "rep_kappa": REP_KAPPA,
            "source_window": {"win_lo": WIN_LO, "win_band": WIN_BAND,
                              "note": ("source masked to compact support in Omega; removes periodic-FFT "
                                       "mean-subtraction contamination from the off-domain point charges")},
            "device": str(device),
        },
        "single_shot_linear_rel_l2": lin_acc,
        "sweep": {},
    }

    for kappa in KAPPAS:
        results["sweep"][str(kappa)] = {}
        for name, solve in solvers.items():
            res = picard_solve(solve, bd, coeffs, charges, kappa, Xq, u_star_q, u_star_b)
            results["sweep"][str(kappa)][name] = res
            print(f"[nonlinear_bvp] kappa={kappa} {name:9s} -> "
                  f"final={res['final_rel_l2']:.3e} best={res['best_rel_l2']:.3e} "
                  f"iters={res['iters']} diverged={res['diverged']}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(RESULTS / "nonlinear_bvp.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[nonlinear_bvp] wrote {RESULTS/'nonlinear_bvp.json'}")

    make_figure(results)
    print(json.dumps(results, indent=2))


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = str(REP_KAPPA)
    kappas = results["meta"]["kappas"]
    colors = {"bem": "tab:green", "nbio": "tab:blue", "deeponet": "tab:red"}
    labels = {"bem": "Classical BEM (ref)", "nbio": "NBIO (zero-shot)", "deeponet": "DeepONet (black-box)"}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: Picard convergence at representative kappa
    for name in ["bem", "nbio", "deeponet"]:
        c = results["sweep"][rep][name]["convergence_rel_l2"]
        ax0.semilogy(range(1, len(c) + 1), c, "-o", ms=3, color=colors[name], label=labels[name])
    ax0.set_xlabel("Picard iteration n")
    ax0.set_ylabel(r"rel-$L^2$  $\|u_n-u^*\|/\|u^*\|$")
    ax0.set_title(f"Picard convergence ($\\kappa={REP_KAPPA}$)")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    # right: final achieved accuracy vs kappa
    for name in ["bem", "nbio", "deeponet"]:
        finals = [results["sweep"][str(k)][name]["final_rel_l2"] for k in kappas]
        ax1.semilogy(kappas, finals, "-s", ms=5, color=colors[name], label=labels[name])
    # per-step linear floor lines for the learned solvers (context)
    for name in ["nbio", "deeponet"]:
        ax1.axhline(results["single_shot_linear_rel_l2"][name], ls="--", lw=1, color=colors[name], alpha=0.6)
    ax1.set_xlabel(r"nonlinearity strength $\kappa$")
    ax1.set_ylabel(r"final rel-$L^2$ vs $u^*$")
    ax1.set_title("Final accuracy vs nonlinearity")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    fig.suptitle(r"Semilinear $-\Delta u+\kappa u^3=f$: per-step linear-solve accuracy sets nonlinear accuracy",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(FIGURES / "nonlinear_bvp.png", dpi=150)
    print(f"[nonlinear_bvp] wrote {FIGURES/'nonlinear_bvp.png'}")


if __name__ == "__main__":
    main()
