"""Helmholtz stress test of the Method-3 scalability claims (off-diagonal low-rank + O(1) GMRES).

The paper's scalability subsection shows, for interior LAPLACE, that the second-kind double-layer system
(a) needs ~O(1) GMRES iterations and (b) has CONSTANT off-diagonal numerical rank (FMM/H-matrix friendly).
Oscillatory Helmholtz is the obvious place this can break: the Helmholtz Green's function H_0^(1)(k r) is
known to be HIGH-RANK in the far field, with off-diagonal block rank growing ~ linearly in the wavenumber k
(number of wavelengths spanned). This script measures both properties as a function of k and contrasts with
the k=0 (Laplace) baseline -- an honest test of how far the constant-rank claim generalizes.

Pure numpy/scipy; additive (new file). Saves results/scal_helmholtz_rank.json + figures/helmholtz_rank.png.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.special import hankel1

from src.data.bie import boundary, sample_coeffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def helmholtz_double_layer(Xt, Xs, Ns, k):
    """Complex Helmholtz double-layer kernel dG/dn_y, G=(i/4)H_0^(1)(k r).
       dG/dn_y = -(i k /4) H_1^(1)(k r) (y-x).n_y / r.   Xt(T,2) targets, Xs/Ns(S,2) sources."""
    diff = Xs[None, :, :] - Xt[:, None, :]                 # (T,S,2) = y - x
    r = np.sqrt((diff ** 2).sum(-1))                       # (T,S)
    num = (diff * Ns[None, :, :]).sum(-1)                  # (y-x).n_y
    if k == 0.0:                                           # Laplace baseline (k=0): real double layer
        with np.errstate(divide="ignore", invalid="ignore"):
            return (-1.0 / (2 * np.pi)) * num / (r ** 2), r
    with np.errstate(divide="ignore", invalid="ignore"):
        K = -(1j * k / 4.0) * hankel1(1, k * r) * num / r
    return K, r


def assemble_interior_helm(bd, k):
    """Interior 2nd-kind Helmholtz double-layer system A=(-1/2 I + D0), D0=K*w, diag = -curv/(4 pi)."""
    X, N, w, curv = bd["X"], bd["normal"], bd["w"], bd["curv"]
    n = X.shape[0]
    K, _ = helmholtz_double_layer(X, X, N, k)
    K = K.astype(np.complex128)
    np.fill_diagonal(K, -curv / (4 * np.pi))               # double-layer diagonal limit (k-independent)
    D0 = K * w[None, :]
    A = -0.5 * np.eye(n, dtype=np.complex128) + D0
    return A


def _gmres(A, b, tol):
    import scipy.sparse.linalg as sla
    it = {"n": 0}

    def cb(_):
        it["n"] += 1
    op = sla.aslinearoperator(A)
    try:
        x, _ = sla.gmres(op, b, rtol=tol, atol=0.0, restart=200, maxiter=400, callback=cb,
                         callback_type="pr_norm")
    except TypeError:
        x, _ = sla.gmres(op, b, tol=tol, atol=0.0, restart=200, maxiter=400, callback=cb)
    res = np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-30)
    return x, it["n"], res


def numerical_rank(M, eps=1e-8):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > eps * s[0]).sum())


def fixed_rank_err(M, ranks=(4, 8, 16)):
    """Relative Frobenius error of the best rank-r approximation of M (maps onto the M4 rank-r claim:
    a learned separable rank-r correction can do no better than the truncated SVD at rank r)."""
    s = np.linalg.svd(M, compute_uv=False)
    tot = float(np.sqrt((s ** 2).sum())) + 1e-30
    out = {}
    for r in ranks:
        tail = float(np.sqrt((s[r:] ** 2).sum())) if r < len(s) else 0.0
        out[str(r)] = tail / tot
    return out


def offdiag_block(bd, k, frac=0.25):
    """Well-separated off-diagonal block: targets = first quarter of arc, sources = third quarter."""
    X, N = bd["X"], bd["normal"]
    n = X.shape[0]
    q = int(n * frac)
    ti = np.arange(0, q)
    si = np.arange(2 * q, 3 * q)
    K, _ = helmholtz_double_layer(X[ti], X[si], N[si], k)
    return K


def run(n_b=2048, n_geom=3, N_wl=(0, 1, 2, 4, 8, 16, 32), seed=0):
    rng = np.random.default_rng(seed)
    geoms = [boundary(sample_coeffs(rng), n_b) for _ in range(n_geom)]
    diam = float(2 * max(np.linalg.norm(g["X"], axis=1).max() for g in geoms))
    print(f"domain diameter ~ {diam:.3f}, n_b={n_b}\n")

    rows = []
    print("=== Helmholtz: off-diag rank + GMRES iters vs wavenumber (vs Laplace baseline) ===")
    for N in N_wl:
        k = 2 * np.pi * N / diam                 # N wavelengths across the domain diameter
        ranks, iters, resids, conds, r4, r8 = [], [], [], [], [], []
        for g in geoms:
            blk = offdiag_block(g, k)
            ranks.append(numerical_rank(blk, eps=1e-8))
            fr = fixed_rank_err(blk)
            r4.append(fr["4"]); r8.append(fr["8"])
            A = assemble_interior_helm(g, k)
            b = (rng.standard_normal(n_b) + 1j * rng.standard_normal(n_b))
            _, ni, rr = _gmres(A, b, 1e-8)
            iters.append(ni); resids.append(rr)
            if n_b <= 2048:
                conds.append(float(np.linalg.cond(A)))
        row = dict(N_wavelengths=N, k=k,
                   offdiag_rank_mean=float(np.mean(ranks)), offdiag_rank_max=int(np.max(ranks)),
                   rank4_relerr=float(np.mean(r4)), rank8_relerr=float(np.mean(r8)),
                   gmres_iters_mean=float(np.mean(iters)), gmres_iters_max=int(np.max(iters)),
                   resid_mean=float(np.mean(resids)),
                   cond_mean=(float(np.mean(conds)) if conds else None))
        rows.append(row)
        ck = f"{row['cond_mean']:.1f}" if row['cond_mean'] is not None else "NA"
        print(f"  N={N:>2} wl (k={k:6.2f}): off-diag rank={row['offdiag_rank_mean']:5.1f} "
              f"(max {row['offdiag_rank_max']:>3}) | rank-4 trunc err={row['rank4_relerr']:.1e} "
              f"| GMRES iters={row['gmres_iters_mean']:5.1f} (max {row['gmres_iters_max']:>3}) "
              f"| cond={ck} | resid={row['resid_mean']:.1e}")

    # rank vs block size at fixed moderate k -> show rank saturates with block size but at a k-dependent level
    print("\n=== off-diag rank vs block size (fixed geometry) at k for N=8 vs Laplace ===")
    g = geoms[0]
    rank_vs_block = []
    for N in (0, 8):
        k = 2 * np.pi * N / diam
        for bs in (64, 128, 256, 512):
            X, Nn = g["X"], g["normal"]
            ti = np.arange(0, bs); si = np.arange(2 * bs, 3 * bs)
            if si[-1] >= n_b:
                continue
            K, _ = helmholtz_double_layer(X[ti], X[si], Nn[si], k)
            r = numerical_rank(K, 1e-8)
            rank_vs_block.append(dict(N_wavelengths=N, block=bs, rank=r))
            print(f"  N={N:>2} wl, block {bs:>4}x{bs:<4}: rank={r}")

    out = dict(domain_diameter=diam, n_b=n_b, n_geom=n_geom, eps=1e-8,
               sweep=rows, rank_vs_block=rank_vs_block)
    path = os.path.join(ROOT, "results", "scal_helmholtz_rank.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[scal_helmholtz_rank] DONE -> {path}")
    _figure(out)
    return out


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["sweep"]
    N = [r["N_wavelengths"] for r in rows]
    rank = [r["offdiag_rank_mean"] for r in rows]
    iters = [r["gmres_iters_mean"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(N, rank, "o-", color="C3")
    ax[0].axhline(rank[0], ls="--", color="C0", alpha=0.6, label=f"Laplace baseline (rank {rank[0]:.0f})")
    ax[0].set_xlabel("wavelengths across domain ($k\\,\\mathrm{diam}/2\\pi$)")
    ax[0].set_ylabel("off-diagonal numerical rank ($\\epsilon=10^{-8}$)")
    ax[0].set_title("(A) Helmholtz off-diag rank GROWS ~linearly in k\n(Laplace constant-rank claim is k=0 only)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    ax[1].plot(N, iters, "s-", color="C1")
    ax[1].axhline(iters[0], ls="--", color="C0", alpha=0.6, label=f"Laplace baseline (~{iters[0]:.0f} iters)")
    ax[1].set_xlabel("wavelengths across domain")
    ax[1].set_ylabel("GMRES iterations to $10^{-8}$")
    ax[1].set_title("(B) GMRES iterations vs k\n(pure double-layer; spikes near interior resonances)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"helmholtz_rank.{ext}"), dpi=150, bbox_inches="tight")
    print("[scal_helmholtz_rank] wrote figures/helmholtz_rank.{png,pdf}")


if __name__ == "__main__":
    run()
