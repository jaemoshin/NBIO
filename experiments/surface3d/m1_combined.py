#!/usr/bin/env python
"""M1 combined experiment: LEARNED correction transfers zero-shot ACROSS TOPOLOGY at solver-grade.

Train the Yukawa-given-Laplace kernel correction on a family of TORI (genus-1) only; evaluate
zero-shot on (a) held-out tori and (b) DEFORMED SPHERES (genus-0, topology never seen), both with
solver-grade QBX bases (torus: Fourier x Fourier, p=10; sphere: GL x Fourier, p=8 — validated
7.5e-6 / 4.7e-6 known-kernel). The correction features are coordinate-free (r, kernel decays,
normal dot-products), so transfer across topology is the falsifiable claim. Baselines: wrong-kernel
floor (0 params) and stabilized DeepSet trained on the same tori.

QBX bases are constants (no grad) -> assembled in numpy (reusing validated sphere_qbx/torus code),
converted to torch; training loop reuses surface_varkernel (featv2, cosine, clip).
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np
import torch

import surface_varkernel as vk          # feats, Corr, DeepSet, solve_eval, rel, torus_np, yukawa_gt, upsample
import sphere_qbx as sq                 # sphere_geom, upsample_sphere, qbx_laplace_np, _fib

PI = math.pi


def make_case(kind, seed, kappa, dev, featv2=True):
    rng = np.random.default_rng(seed)
    if kind == "torus":
        nt, nph, os_, p = 32, 14, 3, 10
        r_min = rng.uniform(0.28, 0.42); defo = rng.uniform(0.0, 0.18)
        Xc, nrmc, wc = vk.torus_np(nt, nph, 1.0, r_min, defo, rng)
        Xf, _, wf = vk.torus_np(nt * os_, nph * os_, 1.0, r_min, defo, rng)
        U = vk.upsample_matrix(nt, nph, nt * os_, nph * os_)
        beta = 2.0 * (2 * PI / nt)
        # interior queries: near centerline of the tube
        aa = np.linspace(0, 2 * PI, 300, endpoint=False)
        off = rng.uniform(0, 0.35 * r_min, 300); be = rng.uniform(0, 2 * PI, 300)
        ring = 1.0 + off * np.cos(be)
        Xq = np.stack([ring * np.cos(aa), ring * np.sin(aa), off * np.sin(be)], 1)
    elif kind == "sphere":
        nu, nph, os_, p = 20, 24, 3, 8
        amp = rng.uniform(0.15, 0.35)
        pa, pb, pc = 0.10, rng.uniform(0.03, 0.08), rng.uniform(0.02, 0.06)
        Xc, nrmc, wc = sq.sphere_geom(nu, nph, amp, pa, pb, pc)
        Xf, _, wf = sq.sphere_geom(nu * os_, nph * os_, amp, pa, pb, pc)
        U = sq.upsample_sphere(nu, nph, nu * os_, nph * os_)
        beta = 2.0 * (PI / nu)
        Xq = sq._fib(300) * rng.uniform(0.05, 0.45, (300, 1))
    elif kind == "two_sphere":
        # THIRD family (never trained on): two deformed genus-0 components, block QBX.
        from scipy.linalg import block_diag
        nu, nph, os_, p = 16, 20, 3, 8
        d = 1.5
        parts = []
        for sgn in (-1.0, 1.0):
            amp = rng.uniform(0.12, 0.30)
            pa, pb, pc = 0.10, rng.uniform(0.03, 0.08), rng.uniform(0.02, 0.06)
            Xc_i, nrm_i, wc_i = sq.sphere_geom(nu, nph, amp, pa, pb, pc)
            Xf_i, _, wf_i = sq.sphere_geom(nu * os_, nph * os_, amp, pa, pb, pc)
            shift = np.array([sgn * d, 0.0, 0.0])
            parts.append((Xc_i + shift, nrm_i, wc_i, Xf_i + shift, wf_i))
        Xc = np.concatenate([p_[0] for p_ in parts]); nrmc = np.concatenate([p_[1] for p_ in parts])
        wc = np.concatenate([p_[2] for p_ in parts])
        Xf = np.concatenate([p_[3] for p_ in parts]); wf = np.concatenate([p_[4] for p_ in parts])
        U1 = sq.upsample_sphere(nu, nph, nu * os_, nph * os_)
        U = block_diag(U1, U1)
        beta = 2.0 * (PI / nu)
        Xq = sq._fib(300) * rng.uniform(0.05, 0.40, (300, 1)) + np.array([-d, 0.0, 0.0])
    else:
        raise ValueError(kind)
    kap = kappa
    c_lo, c_hi = (3.4, 4.2) if kind == "two_sphere" else (2.0, 3.0)  # exterior to ALL components
    c = sq._fib(8) * rng.uniform(c_lo, c_hi, (8, 1))
    q = rng.normal(size=8); q -= q.mean()
    g = vk.yukawa_gt(Xc, c, q, kap)
    uq = vk.yukawa_gt(Xq, c, q, kap)
    Sbase = sq.qbx_laplace_np(Xc, nrmc, Xf, wf, beta, p) @ U
    D = np.sqrt(((Xq[:, None, :] - Xf[None, :, :]) ** 2).sum(-1)).clip(1e-15)
    Pbase = (1.0 / (4 * PI * D) * wf[None, :]) @ U
    T = lambda a: torch.tensor(a, device=dev)
    Xc_t, nrmc_t, Xq_t = T(Xc), T(nrmc), T(Xq)
    fc = vk.feats(Xc_t, Xc_t, nrmc_t, nrmc_t, v2=featv2)
    fq = vk.feats(Xq_t, Xc_t, torch.zeros_like(Xq_t), nrmc_t, v2=featv2)
    return dict(Sbase=T(Sbase), Pbase=T(Pbase), g=T(g), uq=T(uq), fc=fc, fq=fq, wc=T(wc),
                Xc=Xc_t, nrmc=nrmc_t, Xq=Xq_t, kind=kind, kappa=float(kappa),
                charge_centers=T(c), charge_weights=T(q))


def eval_group(cases, corr):
    with torch.no_grad():
        return float(np.mean([vk.rel(vk.solve_eval(c, corr), c["uq"]).item() for c in cases]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=3.0); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ntrain", type=int, default=16); ap.add_argument("--neval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mix", action="store_true", help="train on tori+spheres; add two_sphere eval (3rd family)")
    ap.add_argument("--feats", choices=["full11", "inv8"], default="full11",
                    help="feature set: pre-registered full-11 (default) or rotation-invariant 8")
    a = ap.parse_args()
    if a.smoke:
        a.steps = 30; a.ntrain = 3; a.neval = 2
    torch.set_default_dtype(torch.float64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} kappa={a.kappa} steps={a.steps} mix={a.mix}", flush=True)
    rng = np.random.default_rng(a.seed)

    if a.mix:
        print("[cache] MIXED train (tori+spheres); eval tori+spheres+TWO-SPHERE (3rd family)...", flush=True)
        half = max(1, a.ntrain // 2)
        train = ([make_case("torus", 100 + i, a.kappa, dev) for i in range(half)]
                 + [make_case("sphere", 150 + i, a.kappa, dev) for i in range(half)])
    else:
        print("[cache] building QBX cases (train=tori; eval=tori+spheres)...", flush=True)
        train = [make_case("torus", 100 + i, a.kappa, dev) for i in range(a.ntrain)]
    ev_torus = [make_case("torus", 900 + i, a.kappa, dev) for i in range(a.neval)]
    ev_sphere = [make_case("sphere", 950 + i, a.kappa, dev) for i in range(a.neval)]
    groups = {"torus": ev_torus, "sphere": ev_sphere}
    if a.mix:
        groups["two_sphere"] = [make_case("two_sphere", 990 + i, a.kappa, dev) for i in range(a.neval)]

    res = {"seed": a.seed, "kappa": a.kappa, "mix": a.mix, "feats": a.feats}
    for name, ev in groups.items():
        res[f"floor_{name}"] = eval_group(ev, None)
    print("[floor]    " + "  ".join(f"{k} {res[f'floor_{k}']:.3e}" for k in groups), flush=True)

    if a.feats == "inv8":
        from t1_secondkind import SliceCorr
        corr = SliceCorr([3, 4, 5, 6, 7, 8, 9, 10], h=64).to(dev)
    else:
        corr = vk.Corr(h=64, din=11).to(dev)
    opt = torch.optim.Adam(corr.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=1e-5)
    for step in range(1, a.steps + 1):
        c = train[rng.integers(0, len(train))]
        loss = vk.rel(vk.solve_eval(c, corr), c["uq"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
        opt.step(); sched.step()
        if step % max(1, a.steps // 5) == 0:
            print(f"  [corr] {step}/{a.steps} train rel {loss.item():.3e}", flush=True)
    corr.eval()
    for name, ev in groups.items():
        res[f"corr_{name}"] = eval_group(ev, corr)
    print("[corr]     " + "  ".join(f"{k} {res[f'corr_{k}']:.3e}" for k in groups), flush=True)

    ds = vk.DeepSet().to(dev)
    opt2 = torch.optim.Adam(ds.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=a.steps, eta_min=1e-5)
    for step in range(1, a.steps + 1):
        c = train[rng.integers(0, len(train))]
        loss = vk.rel(ds(c), c["uq"])
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ds.parameters(), 1.0)
        opt2.step(); sched2.step()
    ds.eval()
    with torch.no_grad():
        for name, ev in groups.items():
            res[f"deepset_{name}"] = float(np.mean([vk.rel(ds(c), c["uq"]).item() for c in ev]))
    print("[deepset]  " + "  ".join(f"{k} {res[f'deepset_{k}']:.3e}" for k in groups), flush=True)

    head = "two_sphere" if a.mix else "sphere"
    print(f"\nHEADLINE ({head}, family never seen): floor {res[f'floor_{head}']:.3e} -> corr "
          f"{res[f'corr_{head}']:.3e}; deepset {res[f'deepset_{head}']:.3e}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
