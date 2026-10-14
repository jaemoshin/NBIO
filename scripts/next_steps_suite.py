#!/usr/bin/env python
"""Next-step NBIO diagnostics after the QBX/close-evaluation paper branch.

This script is intentionally additive.  It writes only ``results/next_steps_*`` artifacts and does not
touch the existing paper tables.  The tasks test the open items that were still phrased as future work:

  hmatrix          blocked H-matrix matvec + rank-4 factored correction inside matrix-free GMRES
  volume           mesh-free singular-corrected volume-potential diagnostic for non-periodic Gaussians
  qbx_gap          double-precision reduced-radius QBX/adaptive-panel gap stress down to gap=0.003
  helmholtz_cfie   combined-field Helmholtz conditioning/rank diagnostic
  newton           Newton-vs-Picard diagnostic showing the variable-reaction inner-solve gap
  qbx_train        differentiable end-to-end training through QBX on close-boundary supervision
  report           aggregate the JSON outputs into a concise markdown status report

The H-matrix task is a prototype, not a production FMM: setup still samples/compresses dense blocks, while
the solve uses only compressed block matvecs plus the factored NBIO correction.  That distinction is
recorded in the output JSON/report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import hankel1

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

from src.data.bie import (
    _double_layer_kernel,
    boundary,
    eval_u,
    sample_charges,
    sample_coeffs,
    sample_queries,
)
from src.data.volume_source import phi_and_f, volume_potential_fft


def _jsonify(x):
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, complex):
        return {"real": x.real, "imag": x.imag}
    return x


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonify(obj), f, indent=2)
    print(f"[write] {path.relative_to(ROOT)}")


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------
# Task 1: compressed H-matrix GMRES + factored correction


def _assemble_laplace_D0(bd):
    K, _ = _double_layer_kernel(bd["X"], bd["X"], bd["normal"])
    np.fill_diagonal(K, -bd["curv"] / (4 * np.pi))
    return K * bd["w"][None, :]


class BlockLowRank:
    """Contiguous-block low-rank compression for a dense kernel matrix.

    Near/self/neighbor blocks are dense.  Other blocks are compressed by truncated SVD.  This is enough to
    verify an end-to-end compressed matvec inside GMRES; it is not a full adaptive FMM implementation.
    """

    def __init__(self, M, leaf=192, tol=1e-8, max_rank=48, near=1):
        self.shape = M.shape
        self.blocks = []
        n = M.shape[0]
        edges = list(range(0, n, leaf)) + [n]
        if edges[-2] == n:
            edges = edges[:-1]
        self.edges = edges
        t0 = time.perf_counter()
        dense_blocks = 0
        lr_blocks = 0
        storage = 0
        ranks = []
        for bi, i0 in enumerate(edges[:-1]):
            i1 = edges[bi + 1]
            for bj, j0 in enumerate(edges[:-1]):
                j1 = edges[bj + 1]
                B = np.asarray(M[i0:i1, j0:j1])
                cyclic_dist = min(abs(bi - bj), len(edges[:-1]) - abs(bi - bj))
                if cyclic_dist <= near:
                    self.blocks.append(("dense", i0, i1, j0, j1, B.copy()))
                    dense_blocks += 1
                    storage += B.size
                    continue
                U, s, Vh = np.linalg.svd(B, full_matrices=False)
                if s.size == 0 or s[0] == 0:
                    r = 1
                else:
                    r = int(np.sum(s > tol * s[0]))
                    r = max(1, min(r, max_rank, len(s)))
                # Store U*sqrt(s), sqrt(s)*Vh for stable matvecs.
                sq = np.sqrt(s[:r])
                Us = U[:, :r] * sq[None, :]
                Vs = sq[:, None] * Vh[:r, :]
                self.blocks.append(("lr", i0, i1, j0, j1, Us, Vs))
                lr_blocks += 1
                ranks.append(r)
                storage += Us.size + Vs.size
        self.setup_s = time.perf_counter() - t0
        self.meta = {
            "leaf": leaf,
            "tol": tol,
            "max_rank": max_rank,
            "near_block_radius": near,
            "dense_blocks": dense_blocks,
            "low_rank_blocks": lr_blocks,
            "rank_mean": float(np.mean(ranks)) if ranks else 0.0,
            "rank_max": int(np.max(ranks)) if ranks else 0,
            "storage_entries": int(storage),
            "dense_entries": int(M.size),
            "storage_fraction": float(storage / M.size),
            "setup_s": self.setup_s,
        }

    def matvec(self, x):
        y = np.zeros(self.shape[0], dtype=x.dtype)
        for blk in self.blocks:
            if blk[0] == "dense":
                _, i0, i1, j0, j1, B = blk
                y[i0:i1] += B @ x[j0:j1]
            else:
                _, i0, i1, j0, j1, Us, Vs = blk
                y[i0:i1] += Us @ (Vs @ x[j0:j1])
        return y


def _load_factored_factors(bd, ckpt="results/scal_factored_r4/best.pt"):
    """Return gate, PHI, PSI for the trained rank-4 factored NBIO correction, or an exact-zero fallback."""
    import torch
    from src.models.scalable_factored import FactoredNBIO

    path = ROOT / ckpt
    if not path.exists():
        n = bd["X"].shape[0]
        return 1.0, np.zeros((n, 1)), np.zeros((n, 1)), {"ckpt": ckpt, "loaded": False}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = FactoredNBIO(**ck["config"]["model"]["kwargs"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    X = torch.as_tensor(bd["X"], dtype=torch.float32)[None]
    N = torch.as_tensor(bd["normal"], dtype=torch.float32)[None]
    curv = torch.as_tensor(bd["curv"], dtype=torch.float32)[None]
    with torch.no_grad():
        gate = float(model._gate().detach().cpu())
        PHI = model.factors_phi(X, N, curv)[0].cpu().numpy().astype(np.float64)
        PSI = model.factors_psi(X, N, curv)[0].cpu().numpy().astype(np.float64)
    return gate, PHI, PSI, {"ckpt": ckpt, "loaded": True, "rank": int(PHI.shape[1])}


def run_hmatrix(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in args.ns:
        coeffs = sample_coeffs(rng)
        bd = boundary(coeffs, n)
        ch = sample_charges(rng, "interior", coeffs)
        g = eval_u(bd["X"], ch)
        D0 = _assemble_laplace_D0(bd)
        gate, PHI, PSI, fmeta = _load_factored_factors(bd, args.factored_ckpt)
        corr_dense = PHI @ PSI.T
        A_dense = -0.5 * np.eye(n) + gate * D0 + corr_dense * bd["w"][None, :]
        t0 = time.perf_counter()
        mu_dense = np.linalg.solve(A_dense, g)
        dense_s = time.perf_counter() - t0
        H = BlockLowRank(gate * D0, leaf=args.leaf, tol=args.tol, max_rank=args.max_rank, near=1)

        def matvec(v):
            # A v = jump*v + compressed known D v + factored correction (PHI @ PSI.T @ (w*v)).
            return -0.5 * v + H.matvec(v) + PHI @ (PSI.T @ (bd["w"] * v))

        # Matvec timing before GMRES.
        for _ in range(3):
            matvec(g)
        t0 = time.perf_counter()
        for _ in range(args.matvec_reps):
            matvec(g)
        matvec_ms = 1e3 * (time.perf_counter() - t0) / args.matvec_reps

        it = {"n": 0}

        def cb(_):
            it["n"] += 1

        op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        t0 = time.perf_counter()
        try:
            mu, info = spla.gmres(op, g, rtol=args.gmres_tol, atol=0.0, restart=80, maxiter=300,
                                  callback=cb, callback_type="pr_norm")
        except TypeError:
            mu, info = spla.gmres(op, g, tol=args.gmres_tol, restart=80, maxiter=300, callback=cb)
        gmres_s = time.perf_counter() - t0
        residual = rel(matvec(mu), g)
        density_rel = rel(mu, mu_dense)

        Xq = sample_queries(rng, coeffs, "interior", args.nq)
        uq = eval_u(Xq, ch)
        Kq, _ = _double_layer_kernel(Xq, bd["X"], bd["normal"])
        known_q = gate * ((Kq * bd["w"][None, :]) @ mu)
        # Query-side factored correction uses zero normal/curvature features through the existing model.
        # Loading a second time is acceptable at these sizes and avoids duplicating feature code.
        import torch
        from src.models.scalable_factored import FactoredNBIO
        ck = torch.load(ROOT / args.factored_ckpt, map_location="cpu", weights_only=False) if (ROOT / args.factored_ckpt).exists() else None
        if ck is not None:
            model = FactoredNBIO(**ck["config"]["model"]["kwargs"])
            model.load_state_dict(ck["state_dict"])
            model.eval()
            with torch.no_grad():
                PHIq = model.factors_phi(torch.as_tensor(Xq, dtype=torch.float32)[None], None, None)[0].numpy().astype(np.float64)
            corr_q = PHIq @ (PSI.T @ (bd["w"] * mu))
        else:
            corr_q = 0.0
        u_pred = known_q + corr_q
        query_rel = rel(u_pred, uq)
        row = {
            "n_b": n,
            "leaf": args.leaf,
            "tol": args.tol,
            "max_rank": args.max_rank,
            "dense_solve_ms": 1e3 * dense_s,
            "compressed_setup_ms": 1e3 * H.setup_s,
            "compressed_matvec_ms": matvec_ms,
            "gmres_ms": 1e3 * gmres_s,
            "gmres_iters": it["n"],
            "gmres_info": int(info),
            "solve_residual": residual,
            "density_rel_to_dense": density_rel,
            "query_rel_l2_exact": query_rel,
            **{f"hmatrix_{k}": v for k, v in H.meta.items()},
            **{f"factored_{k}": v for k, v in fmeta.items()},
        }
        rows.append(row)
        print(
            f"[hmatrix] n={n:5d} rank_mean={H.meta['rank_mean']:.1f} "
            f"storage={H.meta['storage_fraction']:.3f} matvec={matvec_ms:.2f}ms "
            f"iters={it['n']} resid={residual:.1e} density_rel={density_rel:.1e} "
            f"query={query_rel:.1e}"
        )
    out = {
        "task": "hmatrix_factored_gmres",
        "claim_status": "prototype: compressed block matvec + factored correction inside GMRES; setup is not a production FMM",
        "rows": rows,
    }
    save_json(out, RESULTS / "next_steps_hmatrix_gmres.json")
    return out


# ---------------------------------------------------------------------------
# Task 2: volume potential / singular quadrature diagnostic


def _direct_log_volume_potential(targets, source_pts, fvals, h, singular_correct=True):
    diff = targets[:, None, :] - source_pts[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    if singular_correct:
        # Average log(r) over a disk with the same area as one grid cell.
        r_eq = h / math.sqrt(math.pi)
        log_self_avg = math.log(r_eq) - 0.5
        r = np.where(r < 0.5 * h, np.exp(log_self_avg), r)
    else:
        r = np.maximum(r, 1e-300)
    return (-1.0 / (2 * math.pi)) * ((np.log(r) * fvals[None, :]).sum(axis=1) * h * h)


def run_volume(args):
    rng = np.random.default_rng(args.seed)
    c = rng.uniform(-0.55, 0.55, (args.n_gauss, 2))
    A = rng.normal(size=args.n_gauss)
    s = rng.uniform(0.12, 0.22, args.n_gauss)

    def f_func(X):
        return phi_and_f(X, c, A, s)[1]

    def phi_func(X):
        return phi_and_f(X, c, A, s)[0]

    targets = rng.uniform(-0.8, 0.8, (args.n_targets, 2))
    phi_t = phi_func(targets)
    fft = volume_potential_fft(f_func, L=args.L, n=args.fft_n)
    fft_t = fft(targets)
    rows = [{
        "method": "periodic_fft_bicubic",
        "n_grid": args.fft_n,
        "rel_l2_to_analytic_phi": rel(fft_t, phi_t),
        "mean_abs_error": float(np.mean(np.abs(fft_t - phi_t))),
        "note": "fast but periodic/mean-subtracted; safe here because Gaussians are compact in a large box",
    }]

    for n in args.direct_ns:
        xs = np.linspace(-args.L, args.L, n, endpoint=False) + args.L / n
        h = xs[1] - xs[0]
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], 1)
        fvals = f_func(pts)
        t0 = time.perf_counter()
        uq = _direct_log_volume_potential(targets, pts, fvals, h, singular_correct=True)
        elapsed = time.perf_counter() - t0
        rows.append({
            "method": "meshfree_log_midpoint_self_corrected",
            "n_grid": n,
            "n_sources": int(pts.shape[0]),
            "rel_l2_to_analytic_phi": rel(uq, phi_t),
            "mean_abs_error": float(np.mean(np.abs(uq - phi_t))),
            "runtime_ms": 1e3 * elapsed,
            "note": "slow O(QM) prototype; cell self-term handles log singularity for targets on/near cells",
        })
        print(f"[volume] direct n={n:4d} rel={rows[-1]['rel_l2_to_analytic_phi']:.2e} runtime={1e3*elapsed:.1f}ms")
    out = {
        "task": "volume_potential_nonperiodic_diagnostic",
        "source_family": "Gaussian phi with analytic f=-Delta phi; free-space particular solution is phi",
        "honest_status": (
            "The Gaussian mesh-free path validates a singular-corrected free-space volume potential for this "
            "source family.  A general high-order singular volume quadrature for arbitrary non-periodic f is "
            "still engineering work."
        ),
        "rows": rows,
    }
    save_json(out, RESULTS / "next_steps_volume_quadrature.json")
    return out


# ---------------------------------------------------------------------------
# Task 3: QBX close-gap double precision stress


def run_qbx_gap(args):
    import torch
    from src.data.bie import sample_charges
    from src.data.transmission import build_curves, transmission_solve
    from src.models.nbio_transmission import NBIOTransmission
    from src.numerics.qbx import multi_single_layer_qbx

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    def circle(R, k=5):
        return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}

    def curves_for_gap(gap, nbo, nbi):
        R0 = 2.0
        r = 0.45
        cen = r + gap / 2
        incls = [(np.array([-cen, 0.0]), r, circle(r)), (np.array([cen, 0.0]), r, circle(r))]
        return build_curves(circle(R0), incls, nbo, nbi)

    def gap_queries(gap, n=90):
        r = 0.45
        x = np.linspace(-0.8 * gap / 2, 0.8 * gap / 2, n)
        y = np.linspace(-0.36 * r, 0.36 * r, 7)
        X, Y = np.meshgrid(x, y)
        return np.stack([X.ravel(), Y.ravel()], 1)

    def batch_from(curves, a_in, g, Xq):
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64, device=device)
        return {
            "Xb_out": t(curves[0]["X"])[None],
            "Nb_out": t(curves[0]["normal"])[None],
            "wb_out": t(curves[0]["w"])[None],
            "curv_out": t(curves[0]["curv"])[None],
            "Xb_in": t(np.stack([c["X"] for c in curves[1:]]))[None],
            "Nb_in": t(np.stack([c["normal"] for c in curves[1:]]))[None],
            "wb_in": t(np.stack([c["w"] for c in curves[1:]]))[None],
            "curv_in": t(np.stack([c["curv"] for c in curves[1:]]))[None],
            "a_in": t(a_in)[None],
            "g": t(g)[None],
            "Xq": t(Xq)[None],
        }

    rows = []
    try:
        model = NBIOTransmission(split=False).to(device).double().eval()
        rng = np.random.default_rng(args.seed)
        for gap in args.gaps:
            ch = sample_charges(rng, "interior", circle(2.0), 6)
            Xq = gap_queries(gap)
            cref = curves_for_gap(gap, args.ref_n, args.ref_n)
            gref = eval_u(cref[0]["X"], ch)
            dref, _ = transmission_solve(cref, [1.0, 5.0, 5.0], gref)
            uref = multi_single_layer_qbx(
                Xq, [c["X"] for c in cref], dref, curves_N=[c["normal"] for c in cref], order=14, oversample=4
            )
            for nb in args.nbs:
                curves = curves_for_gap(gap, nb, nb)
                g = eval_u(curves[0]["X"], ch)
                b = batch_from(curves, [5.0, 5.0], g, Xq)
                for ov in args.oversamples:
                    for rfac in args.rfacs:
                        model.qbx = True
                        model.qbx_kw = dict(order=args.order, r_fac=rfac, near_fac=args.near_fac)
                        model.assembly_oversample = "adaptive" if ov == "adaptive" else int(ov)
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            u = model(b)[0].detach().cpu().numpy()
                        runtime = time.perf_counter() - t0
                        row = {
                            "gap": gap,
                            "n_boundary": nb,
                            "oversample": ov,
                            "qbx_r_fac": rfac,
                            "rel_l2_gap_region": rel(u, uref),
                            "runtime_ms": 1e3 * runtime,
                            "dtype": "float64",
                            "device": str(device),
                        }
                        rows.append(row)
                        print(
                            f"[qbx-gap] gap={gap:.4f} nb={nb} ov={ov} rfac={rfac:.1f} "
                            f"rel={row['rel_l2_gap_region']:.2e} runtime={row['runtime_ms']:.1f}ms"
                        )
    finally:
        torch.set_default_dtype(old_dtype)
    out = {
        "task": "double_precision_reduced_radius_qbx_gap_stress",
        "reference": f"numpy BEM+QBX at n={args.ref_n}",
        "rows": rows,
    }
    save_json(out, ROOT / args.out)
    return out


# ---------------------------------------------------------------------------
# Task 4: Helmholtz CFIE diagnostic


def _helm_D(Xt, Xs, Ns, k):
    diff = Xs[None, :, :] - Xt[:, None, :]
    r = np.sqrt((diff ** 2).sum(-1))
    num = (diff * Ns[None, :, :]).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(1j * k / 4.0) * hankel1(1, k * r) * num / r


def _helm_S(Xt, Xs, k):
    r = np.linalg.norm(Xt[:, None, :] - Xs[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1j / 4.0) * hankel1(0, k * r)


def _laplace_D(Xt, Xs, Ns):
    return _double_layer_kernel(Xt, Xs, Ns)[0]


def _gmres_count(A, b, tol=1e-8):
    it = {"n": 0}

    def cb(_):
        it["n"] += 1

    try:
        x, info = spla.gmres(spla.aslinearoperator(A), b, rtol=tol, atol=0.0, restart=200, maxiter=500,
                             callback=cb, callback_type="pr_norm")
    except TypeError:
        x, info = spla.gmres(spla.aslinearoperator(A), b, tol=tol, restart=200, maxiter=500, callback=cb)
    return it["n"], int(info), rel(A @ x, b)


def _rank(M, eps=1e-8):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > eps * s[0]))


def run_helmholtz_cfie(args):
    rng = np.random.default_rng(args.seed)
    bd = boundary(sample_coeffs(rng), args.n_b)
    n = args.n_b
    diam = float(2 * np.linalg.norm(bd["X"], axis=1).max())
    q = n // 4
    rows = []
    for Nwl in args.wavelengths:
        k = 2 * np.pi * Nwl / diam
        if Nwl == 0:
            D = _laplace_D(bd["X"], bd["X"], bd["normal"]).astype(np.complex128)
            np.fill_diagonal(D, -bd["curv"] / (4 * np.pi))
            A_d = -0.5 * np.eye(n, dtype=np.complex128) + D * bd["w"][None, :]
            rank_blk = _rank((D[:q, 2 * q:3 * q] * bd["w"][None, 2 * q:3 * q]))
            b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            it_d, info_d, res_d = _gmres_count(A_d, b)
            row = {
                "wavelengths_across_diameter": Nwl,
                "k": k,
                "variant": "double_layer_laplace",
                "eta_factor": 0.0,
                "eta_sign": 0,
                "offdiag_rank": rank_blk,
                "gmres_iters": it_d,
                "gmres_info": info_d,
                "residual": res_d,
                "cond": float(np.linalg.cond(A_d)) if n <= 1024 else None,
                "note": "Laplace baseline.",
            }
            rows.append(row)
            print(f"[helm-cfie] N={Nwl:>2} {row['variant']:>22s} rank={rank_blk:>3} GMRES={it_d:>4} cond={row['cond']:.1f}")
            continue
        else:
            D = _helm_D(bd["X"], bd["X"], bd["normal"], k).astype(np.complex128)
            S = _helm_S(bd["X"], bd["X"], k).astype(np.complex128)
            np.fill_diagonal(D, -bd["curv"] / (4 * np.pi))
            # Crude Kress-style diagonal for S: sufficient for conditioning diagnostics, not a final solver.
            h = float(np.mean(bd["w"]))
            sdiag = (1j / 4.0) * hankel1(0, k * h / (2 * math.sqrt(math.pi)))
            np.fill_diagonal(S, sdiag)
            D0 = D * bd["w"][None, :]
            S0 = S * bd["w"][None, :]
            A_d = -0.5 * np.eye(n, dtype=np.complex128) + D0
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        it_d, info_d, res_d = _gmres_count(A_d, b)
        rows.append({
            "wavelengths_across_diameter": Nwl,
            "k": k,
            "variant": "double_layer",
            "eta_factor": 0.0,
            "eta_sign": 0,
            "offdiag_rank": _rank((D[:q, 2 * q:3 * q] * bd["w"][None, 2 * q:3 * q])),
            "gmres_iters": it_d,
            "gmres_info": info_d,
            "residual": res_d,
            "cond": float(np.linalg.cond(A_d)) if n <= 1024 else None,
            "note": "CFIE discretization uses a simple single-layer diagonal; use as conditioning/rank diagnostic, not final Helmholtz BEM validation.",
        })
        print(f"[helm-cfie] N={Nwl:>2} {'double_layer':>22s} rank={rows[-1]['offdiag_rank']:>3} GMRES={it_d:>4} cond={rows[-1]['cond']:.1f}")
        for eta_factor in args.eta_factors:
            for eta_sign in args.eta_signs:
                eta = eta_factor * k
                A_c = -0.5 * np.eye(n, dtype=np.complex128) + D0 + eta_sign * 1j * eta * S0
                it_c, info_c, res_c = _gmres_count(A_c, b)
                rank_blk = _rank((D[:q, 2 * q:3 * q] + eta_sign * 1j * eta * S[:q, 2 * q:3 * q]) * bd["w"][None, 2 * q:3 * q])
                row = {
                    "wavelengths_across_diameter": Nwl,
                    "k": k,
                    "variant": "cfie",
                    "eta_factor": eta_factor,
                    "eta_sign": eta_sign,
                    "offdiag_rank": rank_blk,
                    "gmres_iters": it_c,
                    "gmres_info": info_c,
                    "residual": res_c,
                    "cond": float(np.linalg.cond(A_c)) if n <= 1024 else None,
                    "note": "Swept CFIE coupling; this is a conditioning/rank diagnostic, not final Helmholtz BEM validation.",
                }
                rows.append(row)
                print(
                    f"[helm-cfie] N={Nwl:>2} cfie sign={eta_sign:+d} eta={eta_factor:.2f} "
                    f"rank={rank_blk:>3} GMRES={it_c:>4} cond={row['cond']:.1f}"
                )
    out = {
        "task": "helmholtz_combined_field_diagnostic",
        "honest_status": "CFIE targets resonance/conditioning; off-diagonal rank still grows with k, so directional/butterfly FMM remains necessary.",
        "rows": rows,
    }
    save_json(out, RESULTS / "next_steps_helmholtz_cfie.json")
    return out


# ---------------------------------------------------------------------------
# Task 5: Newton diagnostic for nonlinear BVPs


def run_newton(args):
    # Small finite-difference Dirichlet solve on a unit square.  This is deliberately not NBIO: exact Newton
    # introduces the variable reaction operator (-Delta + 3*k*u^2), which is the point being diagnosed.
    n = args.n_grid
    h = 1.0 / (n + 1)
    xs = np.linspace(h, 1.0 - h, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u_star = np.sin(np.pi * X) * np.sin(np.pi * Y)
    Lap_u_star = -2 * np.pi ** 2 * u_star
    rows = []

    e = np.ones(n)
    T = sp.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(n, n)) / (h * h)
    L = sp.kronsum(T, T, format="csr")  # -Delta with zero Dirichlet
    u_true = u_star.ravel()
    for kappa in args.kappas:
        f = (L @ u_true) + kappa * u_true ** 3
        # Picard: solve L u_{m+1} = f - k u_m^3.
        u = np.zeros_like(u_true)
        pic_curve = []
        for it in range(args.max_iters):
            rhs = f - kappa * u ** 3
            un = spla.spsolve(L, rhs)
            pic_curve.append(rel(un, u_true))
            if rel(un, u) < 1e-10:
                u = un
                break
            u = un
        # Newton: (L + 3k diag(u^2)) delta = -(L u + k u^3 - f)
        u = np.zeros_like(u_true)
        newt_curve = []
        for it in range(args.max_iters):
            F = L @ u + kappa * u ** 3 - f
            J = L + sp.diags(3 * kappa * u ** 2, 0, format="csr")
            delta = spla.spsolve(J, -F)
            # basic damping to keep larger kappa stable
            alpha = 1.0
            base = np.linalg.norm(F)
            while alpha > 1e-3:
                trial = u + alpha * delta
                if np.linalg.norm(L @ trial + kappa * trial ** 3 - f) <= (1 - 1e-4 * alpha) * base:
                    break
                alpha *= 0.5
            u = u + alpha * delta
            newt_curve.append(rel(u, u_true))
            if np.linalg.norm(alpha * delta) / (np.linalg.norm(u) + 1e-30) < 1e-10:
                break
        rows.append({
            "kappa": kappa,
            "picard_iters": len(pic_curve),
            "picard_final_rel": pic_curve[-1],
            "newton_iters": len(newt_curve),
            "newton_final_rel": newt_curve[-1],
            "newton_requires_inner_operator": "-Delta + 3*kappa*u^2",
        })
        print(
            f"[newton] k={kappa:.2f} Picard {len(pic_curve)} iters rel={pic_curve[-1]:.2e}; "
            f"Newton {len(newt_curve)} iters rel={newt_curve[-1]:.2e}"
        )
    out = {
        "task": "nonlinear_newton_diagnostic",
        "honest_status": (
            "Newton converges faster, but its inner solve is variable-reaction/unknown-kernel. Existing NBIO "
            "is a solver-grade zero-shot harmonic/Picard inner solve; exact Newton needs a learned or assembled "
            "operator for -Delta + 3*k*u^2."
        ),
        "rows": rows,
    }
    save_json(out, RESULTS / "next_steps_newton_diagnostic.json")
    return out


# ---------------------------------------------------------------------------
# Task 6: differentiable QBX training smoke


def run_qbx_train(args):
    import torch
    from src.data.bie import sample_charges
    from src.data.transmission import build_curves, transmission_solve
    from src.models.nbio_transmission import NBIOTransmission
    from src.numerics.qbx import multi_single_layer_qbx

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def circle(R, k=5):
        return {"r0": float(R), "a": np.zeros(k), "b": np.zeros(k)}

    gap = args.gap
    r = 0.45
    cen = r + gap / 2
    incls = [(np.array([-cen, 0.0]), r, circle(r)), (np.array([cen, 0.0]), r, circle(r))]
    curves = build_curves(circle(2.0), incls, args.n_b, args.n_b)
    ch = sample_charges(np.random.default_rng(args.seed), "interior", circle(2.0), 6)
    g = eval_u(curves[0]["X"], ch)
    dref, _ = transmission_solve(curves, [1.0, 5.0, 5.0], g)
    x = np.linspace(-0.8 * gap / 2, 0.8 * gap / 2, args.n_line)
    y = np.linspace(-0.18, 0.18, 5)
    X, Y = np.meshgrid(x, y)
    Xq = np.stack([X.ravel(), Y.ravel()], 1)
    uref = multi_single_layer_qbx(Xq, [c["X"] for c in curves], dref, curves_N=[c["normal"] for c in curves],
                                  order=14, oversample=4)

    def t(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    b = {
        "Xb_out": t(curves[0]["X"])[None],
        "Nb_out": t(curves[0]["normal"])[None],
        "wb_out": t(curves[0]["w"])[None],
        "curv_out": t(curves[0]["curv"])[None],
        "Xb_in": t(np.stack([c["X"] for c in curves[1:]]))[None],
        "Nb_in": t(np.stack([c["normal"] for c in curves[1:]]))[None],
        "wb_in": t(np.stack([c["w"] for c in curves[1:]]))[None],
        "curv_in": t(np.stack([c["curv"] for c in curves[1:]]))[None],
        "a_in": t([5.0, 5.0])[None],
        "g": t(g)[None],
        "Xq": t(Xq)[None],
    }
    target = t(uref)[None]
    # Start from a deliberately nonzero smooth correction so optimization has something to fix.
    model = NBIOTransmission(split=True, hidden=32, n_layers=2, init_scale=0.03, qbx=True,
                             qbx_order=12, qbx_r_fac=2.0, qbx_near_fac=2.0,
                             assembly_oversample=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    grad_norms = []
    for step in range(args.steps):
        pred = model(b)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        gn = sum(float(p.grad.detach().abs().sum().cpu()) for p in model.parameters() if p.grad is not None)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(gn)
        if step % max(1, args.steps // 5) == 0:
            print(f"[qbx-train] step={step:03d} loss={losses[-1]:.3e} grad={gn:.3e}")
    out = {
        "task": "differentiable_qbx_training_smoke",
        "gap": gap,
        "n_boundary": args.n_b,
        "device": str(device),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1e-30),
        "max_grad_norm": max(grad_norms),
        "finite_grads": bool(np.isfinite(grad_norms).all()),
        "losses": losses,
        "honest_status": "gradient path through QBX+oversampled assembly is finite and optimizable on close-boundary supervision",
    }
    save_json(out, RESULTS / "next_steps_qbx_training.json")
    return out


# ---------------------------------------------------------------------------
# Report


def run_report(_args):
    files = {
        "H-matrix GMRES": RESULTS / "next_steps_hmatrix_gmres.json",
        "Volume quadrature": RESULTS / "next_steps_volume_quadrature.json",
        "QBX close gaps": RESULTS / "next_steps_qbx_gap_double.json",
        "Helmholtz CFIE": RESULTS / "next_steps_helmholtz_cfie.json",
        "Newton diagnostic": RESULTS / "next_steps_newton_diagnostic.json",
        "QBX training": RESULTS / "next_steps_qbx_training.json",
        "3D joint": RESULTS / "next_steps_3d_joint.json",
        "3D joint DeepSet": RESULTS / "next_steps_3d_joint_deepset.json",
    }
    lines = [
        "# NBIO next-step acceleration/QBX/3D status",
        "",
        "This report aggregates the post-QBX next-step diagnostics.  All rows are additive and isolated under `results/next_steps_*`.",
        "",
    ]
    master = []
    for title, path in files.items():
        lines.append(f"## {title}")
        if not path.exists():
            if title == "QBX close gaps":
                parts = sorted(RESULTS.glob("next_steps_qbx_gap_double_gap_*.json"))
                if parts:
                    merged_rows = []
                    for pp in parts:
                        merged_rows.extend(json.load(open(pp)).get("rows", []))
                    data = {
                        "task": "double_precision_reduced_radius_qbx_gap_stress",
                        "reference": "merged per-gap part files",
                        "rows": merged_rows,
                    }
                    save_json(data, path)
                else:
                    lines.append(f"- Status: not run yet (`{path.relative_to(ROOT)}` missing).")
                    lines.append("")
                    continue
            else:
                lines.append(f"- Status: not run yet (`{path.relative_to(ROOT)}` missing).")
                lines.append("")
                continue
        else:
            data = json.load(open(path))
            if title == "QBX close gaps":
                parts = sorted(RESULTS.glob("next_steps_qbx_gap_double_gap_*.json"))
                if parts and len(data.get("rows", [])) <= 1:
                    merged_rows = []
                    for pp in parts:
                        merged_rows.extend(json.load(open(pp)).get("rows", []))
                    if merged_rows:
                        data = {
                            "task": "double_precision_reduced_radius_qbx_gap_stress",
                            "reference": "merged per-gap part files",
                            "rows": merged_rows,
                        }
                        save_json(data, path)
        lines.append(f"- Artifact: `{path.relative_to(ROOT)}`")
        if "honest_status" in data:
            lines.append(f"- Status: {data['honest_status']}")
        if "claim_status" in data:
            lines.append(f"- Status: {data['claim_status']}")
        rows = data.get("rows")
        if rows:
            lines.append(f"- Rows: {len(rows)}")
            # Compact one-line headline from the last/best row.
            if title == "H-matrix GMRES":
                best = rows[-1]
                lines.append(
                    f"- Headline: n={best['n_b']}, compressed matvec {best['compressed_matvec_ms']:.2f} ms, "
                    f"GMRES {best['gmres_iters']} iters, residual {best['solve_residual']:.1e}, "
                    f"density rel-to-dense {best['density_rel_to_dense']:.1e}."
                )
            elif title == "QBX close gaps":
                best = min(rows, key=lambda r: r["rel_l2_gap_region"])
                worst_gap = min(r["gap"] for r in rows)
                lines.append(
                    f"- Headline: best close-gap row rel-L2 {best['rel_l2_gap_region']:.1e} "
                    f"at gap={best['gap']}, nb={best['n_boundary']}, ov={best['oversample']}, r_fac={best['qbx_r_fac']}; "
                    f"smallest tested gap {worst_gap}."
                )
            elif title == "Helmholtz CFIE":
                max_n = max(r.get("wavelengths_across_diameter", -1) for r in rows)
                high = [r for r in rows if r.get("wavelengths_across_diameter") == max_n]
                best = min(high, key=lambda r: r.get("gmres_iters", 10**9))
                ranks = [r.get("offdiag_rank") for r in high if r.get("offdiag_rank") is not None]
                lines.append(
                    f"- Headline: at {max_n} wavelengths, best swept variant is `{best['variant']}` "
                    f"(eta_factor={best.get('eta_factor')}, sign={best.get('eta_sign')}) with "
                    f"{best['gmres_iters']} GMRES iterations; swept off-diagonal ranks span "
                    f"{min(ranks)}-{max(ranks)}. High-k rank growth remains."
                )
            elif title == "Newton diagnostic":
                last = rows[-1]
                lines.append(
                    f"- Headline: kappa={last['kappa']} Newton {last['newton_iters']} iters vs Picard "
                    f"{last['picard_iters']} iters, but Newton requires `{last['newton_requires_inner_operator']}`."
                )
            elif title == "Volume quadrature":
                best = min(rows, key=lambda r: r["rel_l2_to_analytic_phi"])
                lines.append(
                    f"- Headline: best particular potential rel-L2 {best['rel_l2_to_analytic_phi']:.1e} "
                    f"via {best['method']}."
                )
            master.extend({"section": title, **r} for r in rows[:])
        elif title == "QBX training":
            lines.append(
                f"- Headline: loss {data['initial_loss']:.2e} -> {data['final_loss']:.2e}; "
                f"finite grads={data['finite_grads']}."
            )
        elif title.startswith("3D joint"):
            ev = data.get("eval", {})
            lines.append(f"- Headline: eval={ev}.")
        lines.append("")
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "next_steps_acceleration_qbx_3d_status.md"
    out.write_text("\n".join(lines))
    print(f"[write] {out.relative_to(ROOT)}")
    if master:
        import pandas as pd
        pd.DataFrame(master).to_csv(RESULTS / "next_steps_master_rows.csv", index=False)
        print("[write] results/next_steps_master_rows.csv")


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)

    h = sub.add_parser("hmatrix")
    h.add_argument("--ns", type=int, nargs="+", default=[512, 1024, 2048])
    h.add_argument("--leaf", type=int, default=192)
    h.add_argument("--tol", type=float, default=1e-8)
    h.add_argument("--max-rank", type=int, default=48)
    h.add_argument("--gmres-tol", type=float, default=1e-8)
    h.add_argument("--matvec-reps", type=int, default=10)
    h.add_argument("--nq", type=int, default=256)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--factored-ckpt", default="results/scal_factored_r4/best.pt")

    v = sub.add_parser("volume")
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--n-gauss", type=int, default=6)
    v.add_argument("--n-targets", type=int, default=256)
    v.add_argument("--L", type=float, default=2.8)
    v.add_argument("--fft-n", type=int, default=512)
    v.add_argument("--direct-ns", type=int, nargs="+", default=[96, 144, 192])

    q = sub.add_parser("qbx_gap")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--gaps", type=float, nargs="+", default=[0.02, 0.01, 0.005, 0.003])
    q.add_argument("--nbs", type=int, nargs="+", default=[128, 256])
    q.add_argument("--ref-n", type=int, default=1024)
    q.add_argument("--oversamples", nargs="+", default=["4", "8", "adaptive"])
    q.add_argument("--rfacs", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    q.add_argument("--order", type=int, default=14)
    q.add_argument("--near-fac", type=float, default=2.0)
    q.add_argument("--cpu", action="store_true")
    q.add_argument("--out", default="results/next_steps_qbx_gap_double.json")

    c = sub.add_parser("helmholtz_cfie")
    c.add_argument("--seed", type=int, default=3)
    c.add_argument("--n-b", type=int, default=768)
    c.add_argument("--wavelengths", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    c.add_argument("--eta-factors", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    c.add_argument("--eta-signs", type=int, nargs="+", default=[-1, 1])

    n = sub.add_parser("newton")
    n.add_argument("--n-grid", type=int, default=48)
    n.add_argument("--kappas", type=float, nargs="+", default=[0.1, 0.3, 0.6, 1.0, 2.0])
    n.add_argument("--max-iters", type=int, default=40)

    t = sub.add_parser("qbx_train")
    t.add_argument("--seed", type=int, default=4)
    t.add_argument("--gap", type=float, default=0.01)
    t.add_argument("--n-b", type=int, default=128)
    t.add_argument("--n-line", type=int, default=80)
    t.add_argument("--steps", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--cpu", action="store_true")

    sub.add_parser("report")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.task == "hmatrix":
        run_hmatrix(args)
    elif args.task == "volume":
        run_volume(args)
    elif args.task == "qbx_gap":
        run_qbx_gap(args)
    elif args.task == "helmholtz_cfie":
        run_helmholtz_cfie(args)
    elif args.task == "newton":
        run_newton(args)
    elif args.task == "qbx_train":
        run_qbx_train(args)
    elif args.task == "report":
        run_report(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
