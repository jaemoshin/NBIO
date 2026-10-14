"""End-to-end near-linear NBIO solve: recursive H-matrix matvec + GMRES, validated and scaled.

Closes the "scalability only in principle" caveat. For small n (<=4096) we form the dense operator,
take the dense direct solve as reference, and check the H-matrix+GMRES density matches it. For large n
(8k..65k) the dense matrix is infeasible, so we report only the H-matrix path and show that setup time,
storage, matvec time and total solve time scale near-linearly while GMRES stays at O(1) iterations.

Pure numpy/scipy. Additive new file. Saves results/scal_hmatrix.json + figures/hmatrix_scaling.png.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.sparse.linalg as sla

from src.data.bie import boundary, sample_coeffs, _double_layer_kernel
from src.numerics.hmatrix import Operator, HMatrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def dense_operator(bd):
    n = bd["X"].shape[0]
    K, _ = _double_layer_kernel(bd["X"], bd["X"], bd["normal"])
    np.fill_diagonal(K, -bd["curv"] / (4 * np.pi))
    return -0.5 * np.eye(n) + K * bd["w"][None, :]


def gmres_solve(matvec, b, n, tol=1e-10):
    it = {"n": 0}
    op = sla.LinearOperator((n, n), matvec=matvec)

    def cb(_):
        it["n"] += 1
    try:
        x, info = sla.gmres(op, b, rtol=tol, atol=0.0, restart=100, maxiter=300,
                            callback=cb, callback_type="pr_norm")
    except TypeError:
        x, info = sla.gmres(op, b, tol=tol, atol=0.0, restart=100, maxiter=300, callback=cb)
    return x, it["n"], info


def run(ns_small=(512, 1024, 2048, 4096), ns_big=(8192, 16384, 32768, 65536),
        leaf=64, tol=1e-8, max_rank=96, eta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    print("=== H-matrix + GMRES: validation vs dense (small n) ===")
    for n in ns_small:
        bd = boundary(sample_coeffs(rng), n)
        op = Operator(bd)
        A = dense_operator(bd)
        b = rng.standard_normal(n)
        t0 = time.time(); xd = np.linalg.solve(A, b); t_dense = (time.time() - t0) * 1e3
        t0 = time.time(); H = HMatrix(op, leaf=leaf, tol=tol, max_rank=max_rank, eta=eta); t_setup = (time.time() - t0) * 1e3
        # matvec timing
        x0 = rng.standard_normal(n)
        t0 = time.time()
        for _ in range(20):
            H.matvec(x0)
        t_mv = (time.time() - t0) / 20 * 1e3
        t0 = time.time(); xh, iters, info = gmres_solve(H.matvec, b, n); t_gmres = (time.time() - t0) * 1e3
        dens_rel = float(np.linalg.norm(xh - xd) / (np.linalg.norm(xd) + 1e-300))
        nd, nl = H.n_blocks
        row = dict(n=n, dense=True, t_dense_ms=t_dense, t_setup_ms=t_setup, t_matvec_ms=t_mv,
                   t_gmres_ms=t_gmres, gmres_iters=iters, gmres_info=int(info),
                   storage_fraction=H.storage_fraction, dense_blocks=nd, low_blocks=nl,
                   density_rel_to_dense=dens_rel)
        rows.append(row)
        print(f"  n={n:6d} | dense_solve={t_dense:8.1f}ms | setup={t_setup:7.1f}ms | matvec={t_mv:6.2f}ms "
              f"| gmres={t_gmres:7.1f}ms ({iters} it) | storage={H.storage_fraction:.3f} "
              f"| density rel-to-dense={dens_rel:.2e}")

    print("\n=== H-matrix + GMRES: large n (dense infeasible) ===")
    for n in ns_big:
        bd = boundary(sample_coeffs(rng), n)
        op = Operator(bd)
        b = rng.standard_normal(n)
        t0 = time.time(); H = HMatrix(op, leaf=leaf, tol=tol, max_rank=max_rank, eta=eta); t_setup = (time.time() - t0) * 1e3
        x0 = rng.standard_normal(n)
        t0 = time.time()
        for _ in range(10):
            H.matvec(x0)
        t_mv = (time.time() - t0) / 10 * 1e3
        t0 = time.time(); xh, iters, info = gmres_solve(H.matvec, b, n); t_gmres = (time.time() - t0) * 1e3
        nd, nl = H.n_blocks
        row = dict(n=n, dense=False, t_dense_ms=None, t_setup_ms=t_setup, t_matvec_ms=t_mv,
                   t_gmres_ms=t_gmres, gmres_iters=iters, gmres_info=int(info),
                   storage_fraction=H.storage_fraction, dense_blocks=nd, low_blocks=nl,
                   density_rel_to_dense=None)
        rows.append(row)
        print(f"  n={n:6d} | setup={t_setup:8.1f}ms | matvec={t_mv:6.2f}ms | gmres={t_gmres:8.1f}ms "
              f"({iters} it) | storage={H.storage_fraction:.4f} | total={(t_setup+t_gmres):.0f}ms")

    out = dict(params=dict(leaf=leaf, tol=tol, max_rank=max_rank, eta=eta), rows=rows)
    path = os.path.join(ROOT, "results", "scal_hmatrix.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[scal_hmatrix] DONE -> {path}")
    _figure(out)
    return out


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    n = np.array([r["n"] for r in rows], float)
    setup = np.array([r["t_setup_ms"] for r in rows])
    gmres = np.array([r["t_gmres_ms"] for r in rows])
    total = setup + gmres
    storf = np.array([r["storage_fraction"] for r in rows])
    iters = np.array([r["gmres_iters"] for r in rows])
    dense_t = np.array([r["t_dense_ms"] if r["t_dense_ms"] is not None else np.nan for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (A) total solve time vs n + O(n log n) and O(n^3) guides
    ax[0].loglog(n, total, "o-", color="C2", label="H-matrix setup+GMRES")
    ax[0].loglog(n, dense_t, "s--", color="C1", label="dense direct $O(n^3)$")
    g = n * np.log2(n); g = g / g[0] * total[0]
    ax[0].loglog(n, g, ":", color="k", alpha=0.6, label="$O(n\\log n)$ guide")
    ax[0].set_xlabel("boundary nodes $n$"); ax[0].set_ylabel("time (ms)")
    ax[0].set_title("(A) End-to-end solve time: near-linear\nvs dense $O(n^3)$")
    ax[0].legend(fontsize=8); ax[0].grid(True, which="both", alpha=0.2)
    # (B) storage fraction (of dense) -> O(log n / n)
    ax[1].loglog(n, storf, "o-", color="C0")
    g2 = np.log2(n) / n; g2 = g2 / g2[0] * storf[0]
    ax[1].loglog(n, g2, ":", color="k", alpha=0.6, label="$O(\\log n / n)$ guide")
    ax[1].set_xlabel("boundary nodes $n$"); ax[1].set_ylabel("storage / dense $n^2$")
    ax[1].set_title("(B) H-matrix storage fraction\nshrinks like $\\log n / n$")
    ax[1].legend(fontsize=8); ax[1].grid(True, which="both", alpha=0.2)
    # (C) GMRES iters flat
    ax[2].semilogx(n, iters, "o-", color="C3")
    ax[2].set_ylim(0, max(15, iters.max() + 3))
    ax[2].set_xlabel("boundary nodes $n$"); ax[2].set_ylabel("GMRES iterations to $10^{-10}$")
    ax[2].set_title("(C) Second-kind: $O(1)$ iterations\nindependent of $n$")
    ax[2].grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"hmatrix_scaling.{ext}"), dpi=150, bbox_inches="tight")
    print("[scal_hmatrix] wrote figures/hmatrix_scaling.{png,pdf}")


if __name__ == "__main__":
    run()
