#!/usr/bin/env python
"""Generate REAL manufactured-Helmholtz operator-learning shards for the Method-4 rank stress test.

For each wavenumber k = 2*pi*N/diam (N wavelengths across the star-domain diameter) we build a
train/val/test set of random star-shaped geometries, each with a FRESH random real plane-wave solution

    u*(x) = sum_{m=1}^{M} a_m cos(k (d_m . x) + phi_m),   d_m unit vectors, a ~ N(0,1) (normalized), phi ~ U[0,2pi)

Each term cos(k d.x + phi) satisfies Delta u + k^2 u = 0 EXACTLY (d unit), so u* is an exact real interior
Helmholtz solution with NO singularities anywhere. Boundary Dirichlet data g = u*|_dOmega; interior targets
uq = u* at sampled query points. This is a clean (geometry + g) -> u operator-learning family at fixed k.

Shards saved under data_cache/helm_N{N}_{train,val,test}.pt with the SAME keys BIEDataset expects
(Xb,Nb,wb,curv,speed,g,Xq,uq, all float32). No classical BIE solve / mu (target is the exact u*).

ADDITIVE / NEW FILE ONLY: imports boundary, sample_coeffs, sample_queries from src.data.bie unchanged.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import boundary, sample_coeffs, sample_queries

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_cache")

N_WL = [1, 2, 4, 8, 16]
N_B = 512            # fixed across the sweep; >= 12 pts/wavelength along perimeter for the largest N
N_Q = 256
N_MODES = 5
M_WAVES = 12         # number of plane waves in each manufactured solution
N_TRAIN, N_VAL, N_TEST = 2000, 256, 256
SEED0 = 1000


# ----------------------------- domain diameter -----------------------------
def compute_diam(n_geom=64, n_b=512, seed=7):
    """Diameter ~ max pairwise distance across many star geometries, from src.data.bie.boundary.
    We use the mean max-extent (2*max|X|) as the representative diameter the wavenumber is scaled by;
    also report the true max pairwise diameter for reference."""
    rng = np.random.default_rng(seed)
    extents, pair = [], []
    for _ in range(n_geom):
        X = boundary(sample_coeffs(rng, N_MODES), n_b)["X"]
        extents.append(2.0 * float(np.linalg.norm(X, axis=1).max()))
        # exact max pairwise distance (n_b=512 -> fine)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        pair.append(float(np.sqrt(d2.max())))
    return float(np.mean(extents)), float(np.mean(pair)), float(np.max(pair))


# ----------------------------- manufactured plane-wave solution -----------------------------
def plane_wave_params(rng, M, dim=2):
    """Random unit directions d_m, amplitudes a_m ~ N(0,1) normalized to unit L2, phases phi_m ~ U[0,2pi)."""
    d = rng.standard_normal((M, dim))
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    a = rng.standard_normal(M)
    a /= np.linalg.norm(a) + 1e-12
    phi = rng.uniform(0, 2 * np.pi, M)
    return d, a, phi


def eval_plane_wave(points, k, d, a, phi):
    """u(x) = sum_m a_m cos(k (d_m . x) + phi_m). points (P,2) -> (P,)."""
    arg = k * (points @ d.T) + phi[None, :]      # (P, M)
    return np.cos(arg) @ a                        # (P,)


def laplacian_residual(points, k, d, a, phi, h=1e-3):
    """Finite-difference (Delta u + k^2 u) / (k^2 |u|) on a set of points; should be ~ O(h^2) -> tiny."""
    u0 = eval_plane_wave(points, k, d, a, phi)
    lap = np.zeros_like(u0)
    for ax in range(2):
        e = np.zeros(2); e[ax] = h
        up = eval_plane_wave(points + e, k, d, a, phi)
        um = eval_plane_wave(points - e, k, d, a, phi)
        lap += (up - 2 * u0 + um) / h ** 2
    res = lap + k ** 2 * u0
    scale = k ** 2 * (np.abs(u0).mean() + 1e-12)
    return float(np.abs(res).mean() / scale)


# ----------------------------- shard assembly -----------------------------
def gen_split(rng, n, k):
    Xb_, Nb_, wb_, curv_, speed_, g_, Xq_, uq_ = [], [], [], [], [], [], [], []
    for _ in range(n):
        coeffs = sample_coeffs(rng, N_MODES)
        bd = boundary(coeffs, N_B)
        d, a, phi = plane_wave_params(rng, M_WAVES)
        Xq = sample_queries(rng, coeffs, "interior", N_Q)
        g = eval_plane_wave(bd["X"], k, d, a, phi)
        uq = eval_plane_wave(Xq, k, d, a, phi)
        Xb_.append(bd["X"]); Nb_.append(bd["normal"]); wb_.append(bd["w"])
        curv_.append(bd["curv"]); speed_.append(bd["speed"])
        g_.append(g); Xq_.append(Xq); uq_.append(uq)
    out = {
        "Xb": np.stack(Xb_), "Nb": np.stack(Nb_), "wb": np.stack(wb_),
        "curv": np.stack(curv_), "speed": np.stack(speed_),
        "g": np.stack(g_), "Xq": np.stack(Xq_), "uq": np.stack(uq_),
    }
    return {kk: torch.from_numpy(vv.astype(np.float32)) for kk, vv in out.items()}


def main():
    os.makedirs(DATA, exist_ok=True)
    diam_extent, diam_pair_mean, diam_pair_max = compute_diam()
    diam = diam_extent
    perimeter_ref = None
    # report a representative perimeter for the resolution check
    rng0 = np.random.default_rng(7)
    perims = []
    for _ in range(16):
        bd = boundary(sample_coeffs(rng0, N_MODES), N_B)
        perims.append(float(bd["w"].sum()))   # arc weights sum to perimeter
    perimeter_ref = float(np.mean(perims))
    print(f"[gen_helmholtz] diam(mean 2*max|X|)={diam:.4f}  diam(mean pairwise)={diam_pair_mean:.4f}  "
          f"diam(max pairwise)={diam_pair_max:.4f}  perimeter~{perimeter_ref:.4f}  n_b={N_B}")

    # ----- residual sanity check (manufactured u* satisfies Helmholtz) -----
    print("\n[gen_helmholtz] === manufactured-solution Helmholtz residual check ===")
    rngc = np.random.default_rng(123)
    coeffs = sample_coeffs(rngc, N_MODES)
    pts = sample_queries(rngc, coeffs, "interior", 1024)
    resid_report = {}
    for N in N_WL:
        k = 2 * np.pi * N / diam
        d, a, phi = plane_wave_params(rngc, M_WAVES)
        r = laplacian_residual(pts, k, d, a, phi, h=1e-3)
        resid_report[N] = r
        wl_perim = perimeter_ref * N / diam            # wavelengths along perimeter
        ppw = N_B / max(wl_perim, 1e-9)                # points per wavelength
        print(f"  N={N:>2}  k={k:7.3f}  |Δu*+k²u*|/(k²|u*|)={r:.2e}  "
              f"wavelengths along perimeter~{wl_perim:5.1f}  pts/wavelength~{ppw:5.1f}")
    assert max(resid_report.values()) < 1e-3, f"Helmholtz residual too large: {resid_report}"
    print("[gen_helmholtz] residual check PASSED (all manufactured u* satisfy Helmholtz)")

    # ----- generate shards -----
    print("\n[gen_helmholtz] === generating shards ===")
    for N in N_WL:
        k = 2 * np.pi * N / diam
        for split, n, soff in [("train", N_TRAIN, 0), ("val", N_VAL, 1), ("test", N_TEST, 2)]:
            rng = np.random.default_rng(SEED0 + 100 * N + soff)
            d = gen_split(rng, n, k)
            path = os.path.join(DATA, f"helm_N{N}_{split}.pt")
            torch.save(d, path)
            # quick GT magnitude sanity (rel-L2 of GT vs itself trivially 0; just report norm scale)
            unorm = float(torch.linalg.vector_norm(d["uq"], dim=1).mean())
            print(f"  wrote {path}  n={n}  k={k:.3f}  mean|uq|={unorm:.3f}")
    print("\n[gen_helmholtz] DONE")
    print(f"[gen_helmholtz] DIAM={diam:.6f}")   # parsed by the experiment driver if needed


if __name__ == "__main__":
    main()
