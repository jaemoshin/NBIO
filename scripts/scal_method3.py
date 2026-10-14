#!/usr/bin/env python
"""Method 3 scalability evidence (additive; does not modify any existing file).

(a) GMRES iteration-count scaling: the NBIO/BEM system is a SECOND-KIND Fredholm equation
    (-1/2 I + D0), so its spectrum is bounded away from 0 and GMRES converges in O(1) iterations
    INDEPENDENT of n_b. We assemble the interior double-layer system at growing n_b and count GMRES
    iterations + condition number. Flat iteration count => matrix-free O(n log n)-matvec solves scale.

(b) Hierarchical low-rank (FMM-exploitable) structure: off-diagonal blocks between well-separated
    boundary arcs are numerically LOW RANK with rank ~ O(log 1/eps) independent of n_b. We measure the
    eps=1e-8 numerical rank of a fixed well-separated off-diagonal block as n_b grows. Constant rank =>
    H-matrix / FMM storage and matvec are O(n log n). (Full FMM tree is future work; this demonstrates the
    structural property the FMM exploits.)
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import scipy.sparse.linalg as spla

from src.data.bie import boundary, sample_coeffs, sample_charges, eval_u, _double_layer_kernel


def assemble_interior(bd):
    """A = -1/2 I + D0  (interior double-layer, the validated NBIO/BEM 2nd-kind system)."""
    n = bd["X"].shape[0]
    K0, _ = _double_layer_kernel(bd["X"], bd["X"], bd["normal"])
    np.fill_diagonal(K0, -bd["curv"] / (4 * np.pi))
    D0 = K0 * bd["w"][None, :]
    return -0.5 * np.eye(n) + D0


def gmres_scaling(ns=(128, 256, 512, 1024, 2048, 4096), n_geom=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        iters, conds, t_dense, t_gmres, resids = [], [], [], [], []
        for _ in range(n_geom):
            coeffs = sample_coeffs(rng, 5)
            bd = boundary(coeffs, n)
            ch = sample_charges(rng, "interior", coeffs, 6)
            g = eval_u(bd["X"], ch)                      # harmonic boundary trace
            A = assemble_interior(bd)
            if n <= 1024:
                conds.append(float(np.linalg.cond(A)))
            t0 = time.perf_counter(); _ = np.linalg.solve(A, g); t_dense.append(time.perf_counter() - t0)
            cnt = [0]
            def cb(*a): cnt[0] += 1
            op = spla.LinearOperator((n, n), matvec=lambda v: A @ v)   # matvec is O(n^2) dense here;
            t0 = time.perf_counter()                                   # FMM would make it O(n log n)
            try:
                mu, info = spla.gmres(op, g, rtol=1e-10, restart=60, maxiter=200, callback=cb)
            except TypeError:
                mu, info = spla.gmres(op, g, tol=1e-10, restart=60, maxiter=200, callback=cb)
            t_gmres.append(time.perf_counter() - t0)
            iters.append(cnt[0]); resids.append(float(np.linalg.norm(A @ mu - g) / np.linalg.norm(g)))
        rows.append({"n_b": n, "gmres_iters_mean": float(np.mean(iters)), "gmres_iters_max": int(np.max(iters)),
                     "cond_mean": (float(np.mean(conds)) if conds else None),
                     "resid_mean": float(np.mean(resids)),
                     "t_dense_ms": 1e3 * float(np.mean(t_dense)), "t_gmres_ms": 1e3 * float(np.mean(t_gmres))})
        print(f"  n_b={n:5d}: gmres_iters={rows[-1]['gmres_iters_mean']:.1f} (max {rows[-1]['gmres_iters_max']}) "
              f"cond={rows[-1]['cond_mean']} resid={rows[-1]['resid_mean']:.1e} "
              f"dense={rows[-1]['t_dense_ms']:.1f}ms gmres={rows[-1]['t_gmres_ms']:.1f}ms")
    return rows


def offdiag_rank(ns=(256, 512, 1024, 2048, 4096), eps=1e-8, seed=1):
    """Numerical rank (sing. vals > eps*sigma_max) of a fixed well-separated off-diagonal block.
    Target = first quarter of the curve, source = third quarter (well separated) -> low rank, ~const in n."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        coeffs = sample_coeffs(rng, 5)
        bd = boundary(coeffs, n)
        q = n // 4
        ti = slice(0, q); sj = slice(2 * q, 3 * q)        # well-separated arcs
        K, _ = _double_layer_kernel(bd["X"][ti], bd["X"][sj], bd["normal"][sj])
        blk = K * bd["w"][None, sj]
        sv = np.linalg.svd(blk, compute_uv=False)
        rank = int(np.sum(sv > eps * sv[0]))
        rows.append({"n_b": n, "block_size": q, "numerical_rank": rank, "compression": q / max(rank, 1)})
        print(f"  n_b={n:5d}: off-diag block {q}x{q}, eps=1e-8 rank={rank}  (compression {q/max(rank,1):.0f}x)")
    return rows


def main():
    print("=== Method 3a: GMRES iteration scaling (2nd-kind => O(1) iters) ===")
    gmres = gmres_scaling()
    print("\n=== Method 3b: off-diagonal numerical rank (FMM/H-matrix compressibility) ===")
    ranks = offdiag_rank()
    os.makedirs("results", exist_ok=True)
    json.dump({"gmres_scaling": gmres, "offdiag_rank": ranks},
              open("results/scal_method3.json", "w"), indent=2)
    print("\n[scal_method3] DONE -> results/scal_method3.json")


if __name__ == "__main__":
    main()
