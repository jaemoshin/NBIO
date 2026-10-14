"""Helmholtz scalability fix: directional (butterfly) admissibility restores bounded block rank.

scal_helmholtz_rank.json showed the off-diagonal numerical rank of the Helmholtz double-layer operator
grows ~linearly with the wavenumber k -- so a plain H-matrix / flat low-rank is inadequate at high k. This
script demonstrates the STANDARD remedy (directional / butterfly FMM, Engquist-Ying): the far field of a
source cluster is split into O(k) directional cones of half-width ~ 1/(k a); after removing the source-side
plane-wave phase e^{ik d.(y-c_S)} within a cone, the dephased sub-block is low-rank with rank BOUNDED
independent of k. This converts the Helmholtz caveat from "limitation" to "limitation with a demonstrated
standard fix".

Pure numpy. Additive new file. Saves results/helmholtz_directional.json + figures/helmholtz_directional.png.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import hankel1

from src.data.bie import boundary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


def helm_dl(Xt, Xs, Ns, k):
    """Complex Helmholtz double-layer kernel block (T x S), well-separated (no self terms)."""
    diff = Xs[None, :, :] - Xt[:, None, :]
    r = np.sqrt((diff ** 2).sum(-1))
    num = (diff * Ns[None, :, :]).sum(-1)
    return -(1j * k / 4.0) * hankel1(1, k * r) * num / r


def numerical_rank(M, eps=1e-7):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > eps * s[0]).sum())


def run(n_b=8192, N_wl=(2, 4, 8, 16, 32, 64, 128), src_nodes=512, cone_const=1.0, eps=1e-7,
        far_cap=4000, seed=0):
    # exact circle for a clean, controlled geometry
    coeffs = {"r0": 1.0, "a": np.zeros(5), "b": np.zeros(5)}
    bd = boundary(coeffs, n_b)
    X, Nn = bd["X"], bd["normal"]
    diam = 2.0  # unit circle
    rng = np.random.default_rng(seed)

    # source cluster: a contiguous arc spanning many wavelengths; far targets: well-separated nodes
    s0 = 0
    S = np.arange(s0, s0 + src_nodes)
    c_S = X[S].mean(0)
    a = float(np.sqrt(((X[S] - c_S) ** 2).sum(1)).max())   # cluster radius
    dist = np.sqrt(((X - c_S) ** 2).sum(1))
    far = np.where(dist > 0.7)[0]                          # far field of the cluster
    if len(far) > far_cap:                                 # cap rows to keep SVDs tractable (rank << cap)
        far = np.sort(rng.choice(far, far_cap, replace=False))
    print(f"n_b={n_b}, src arc={src_nodes} nodes, cluster radius a={a:.4f}, far targets={len(far)}")

    rows = []
    for N in N_wl:
        k = 2 * np.pi * N / diam
        # (i) raw far-field block rank (the limitation)
        Braw = helm_dl(X[far], X[S], Nn[S], k)
        raw = numerical_rank(Braw, eps)
        # (ii) directional: partition far targets into cones of half-width beta ~ cone_const/(k a)
        beta = cone_const / max(k * a, 1.0)                # radians (half-width)
        ang = np.arctan2(X[far, 1] - c_S[1], X[far, 0] - c_S[0])   # direction of each target
        n_cone = max(1, int(np.ceil(np.pi / beta)))        # cones across 2pi (full circle of directions)
        edges = np.linspace(-np.pi, np.pi, n_cone + 1)
        bin_id = np.clip(np.digitize(ang, edges) - 1, 0, n_cone - 1)
        ysrc = X[S] - c_S[None, :]
        dephased_ranks = []
        used_cones = 0
        for cb in range(n_cone):
            sel = far[bin_id == cb]
            if len(sel) < 16:
                continue
            dctr = edges[cb] + (edges[cb + 1] - edges[cb]) / 2
            dhat = np.array([np.cos(dctr), np.sin(dctr)])
            phase = np.exp(1j * k * (ysrc @ dhat))         # remove source-side oscillation (col scaling)
            Bp = helm_dl(X[sel], X[S], Nn[S], k) * phase[None, :]
            dephased_ranks.append(numerical_rank(Bp, eps))
            used_cones += 1
        dr = np.array(dephased_ranks)
        row = dict(N_wavelengths=N, k=k, cluster_radius=a, raw_rank=raw,
                   cone_halfwidth_rad=beta, n_cones_total=n_cone, n_cones_used=used_cones,
                   dephased_rank_max=int(dr.max()), dephased_rank_mean=float(dr.mean()))
        rows.append(row)
        print(f"  N={N:>2} wl (k={k:6.2f}): raw block rank={raw:>4} | cones~{used_cones:>3} "
              f"(half-width {np.degrees(beta):5.1f} deg) | dephased rank: max={int(dr.max()):>3} "
              f"mean={dr.mean():4.1f}")

    out = dict(n_b=n_b, diam=diam, src_nodes=src_nodes, cluster_radius=a, eps=eps,
               cone_const=cone_const, rows=rows,
               note="raw far-field rank grows ~linearly in k; directional dephasing within cones of "
                    "half-width ~1/(k a) keeps per-cone rank bounded -> directional/butterfly FMM is O(N log N).")
    path = os.path.join(ROOT, "results", "helmholtz_directional.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[helmholtz_directional] DONE -> {path}")
    _figure(out)
    return out


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    N = [r["N_wavelengths"] for r in rows]
    raw = [r["raw_rank"] for r in rows]
    dmax = [r["dephased_rank_max"] for r in rows]
    ncone = [r["n_cones_used"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(N, raw, "o-", color="C3", label="raw far-field block rank")
    ax[0].plot(N, dmax, "s-", color="C2", label="directional (dephased) per-cone rank")
    ax[0].set_xlabel("wavelengths across domain $k\\,\\mathrm{diam}/2\\pi$")
    ax[0].set_ylabel(f"numerical rank ($\\epsilon={out['eps']:g}$)")
    ax[0].set_title("(A) Directional dephasing keeps per-block rank BOUNDED\nwhile the raw block rank grows with $k$")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    ax[1].plot(N, ncone, "o-", color="C0")
    ax[1].set_xlabel("wavelengths across domain")
    ax[1].set_ylabel("# directional cones used")
    ax[1].set_title("(B) #cones grows $\\sim$linearly in $k$\n(the $O(k)$ butterfly direction count)")
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"helmholtz_directional.{ext}"), dpi=150, bbox_inches="tight")
    print("[helmholtz_directional] wrote figures/helmholtz_directional.{png,pdf}")


if __name__ == "__main__":
    run()
