#!/usr/bin/env python
"""Learning-necessary surface regime: WRONG-KERNEL modified-Helmholtz (Yukawa) on tori.

NBIO-Stage-2 / Yukawa analog in 3D on surfaces, reusing the validated differentiable QBX (Laplace
single-layer). Manufactured GT solves (Delta - kappa^2)u = 0 in the interior (Yukawa point charges
outside): u(x) = sum_i q_i e^{-kappa|x-c_i|}/(4pi|x-c_i|). The model is given ONLY the Laplace QBX
single-layer as its known base kernel (WRONG for Yukawa) -> a learned correction is NECESSARY.

Models (all consume the SAME solver-grade Laplace QBX base; differ only in learning):
  laplace_only  : Laplace single-layer, no correction (0 params) -> wrong-kernel FLOOR.
  laplace_corr  : Laplace QBX base + learned smooth correction (added to solve & eval kernels).
  deepset       : black-box (fails, as before).
Train on a torus family (varying R,r,deformation); eval ZERO-SHOT on held-out torus shapes.
Success (NBIO pattern): laplace_only floors high, laplace_corr closes a large fraction, deepset fails.
Caches the QBX bases (expensive) once, then trains the correction on the cache.
"""
from __future__ import annotations
import argparse, math
import numpy as np
from numpy.fft import rfft, irfft
import torch, torch.nn as nn

PI = math.pi


def torus_np(n_th, n_ph, R, r, defo, rng):
    th = np.arange(n_th) * 2 * PI / n_th
    ph = np.arange(n_ph) * 2 * PI / n_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij"); TH = TH.ravel(); PH = PH.ravel()
    phase = 2 * TH + 0.7
    rr = r * (1.0 + defo * np.cos(phase))                      # mild shape variation
    drr_dth = -2 * r * defo * np.sin(phase)
    ring = R + rr * np.cos(PH)
    X = np.stack([ring * np.cos(TH), ring * np.sin(TH), rr * np.sin(PH)], 1)
    # The standard torus normal/area formula is exact only when rr is constant.
    # Include drr/dtheta so deformed cases use the geometry they actually sample.
    dX_dth = np.stack([
        drr_dth * np.cos(PH) * np.cos(TH) - ring * np.sin(TH),
        drr_dth * np.cos(PH) * np.sin(TH) + ring * np.cos(TH),
        drr_dth * np.sin(PH),
    ], 1)
    dX_dph = np.stack([
        -rr * np.sin(PH) * np.cos(TH),
        -rr * np.sin(PH) * np.sin(TH),
        rr * np.cos(PH),
    ], 1)
    area_vec = np.cross(dX_dth, dX_dph)
    area = np.linalg.norm(area_vec, axis=1)
    nrm = area_vec / area[:, None]
    w = area * (2 * PI / n_th) * (2 * PI / n_ph)
    return X, nrm, w


def upsample_1d(N, M):
    U = np.zeros((M, N))
    for j in range(N):
        e = np.zeros(N); e[j] = 1.0
        E = rfft(e); Ep = np.zeros(M // 2 + 1, dtype=complex)
        k = min(len(E), len(Ep)); Ep[:k] = E[:k]
        U[:, j] = irfft(Ep, M) * (M / N)
    return U


def upsample_matrix(nt, nph, mt, mph):
    return np.kron(upsample_1d(nt, mt), upsample_1d(nph, mph))


def qbx_laplace(Xc, nrmc, Xf, wf, beta, p):
    C = Xc + beta * nrmc
    d = Xf[None] - C[:, None]
    r_y = torch.linalg.vector_norm(d, dim=-1).clamp_min(1e-30)
    cosg = -(nrmc[:, None, :] * d).sum(-1) / r_y
    S = (beta ** 0) / r_y * torch.ones_like(cosg)
    Pm1 = torch.ones_like(cosg); Pl = cosg.clone()
    S = S + beta / r_y ** 2 * Pl
    for l in range(1, p):
        Pn = ((2 * l + 1) * cosg * Pl - l * Pm1) / (l + 1)
        Pm1, Pl = Pl, Pn
        S = S + beta ** (l + 1) / r_y ** (l + 2) * Pl
    return S * wf[None, :] / (4 * PI)


def feats(Xt, Xs, Nt, Ns, v2=False):
    diff = Xs[None, :, :] - Xt[:, None, :]
    r = diff.norm(dim=-1, keepdim=True).clamp_min(1e-9); dh = diff / r
    nt = Nt[:, None, :]; ns = Ns[None, :, :]
    base = [dh, r, torch.exp(-r), (dh * nt).sum(-1, True),
            (dh * ns).sum(-1, True), (nt * ns).sum(-1, True)]                  # 8 dims
    if v2:  # multi-scale radial decays + BOUNDED 1/r surrogate: (1-e^{-r})/(4 pi r) is analytic at r=0
            # (-> 1/4pi) and -> 1/(4 pi r) at far field, so the far correction stays LINEAR in it and
            # extrapolates beyond the training r-range (raw 1/r exploded on near-diagonal pairs).
        base += [torch.exp(-2 * r), torch.exp(-4 * r),
                 (1.0 - torch.exp(-r)) / (4 * math.pi * r)]                     # 11 dims
    return torch.cat(base, -1)


def yukawa_gt(X, c, q, kappa):
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2).clip(1e-12)
    return (np.exp(-kappa * d) / (4 * PI * d)) @ q


def _fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = PI * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def build_case(nt, nph, os_, p, cbeta, kappa, R, r, defo, seed, dev, featv2=False):
    rng = np.random.default_rng(seed)
    Xc, nrmc, wc = torus_np(nt, nph, R, r, defo, rng)
    mt, mph = nt * os_, nph * os_
    Xf, nrmf, wf = torus_np(mt, mph, R, r, defo, rng)
    U = upsample_matrix(nt, nph, mt, mph)
    beta = cbeta * (2 * PI * R / nt)
    c = _fib(8) * rng.uniform(2.2, 3.2, (8, 1)); q = rng.normal(size=8); q -= q.mean()
    g = yukawa_gt(Xc, c, q, kappa)
    aa = np.linspace(0, 2 * PI, 300, endpoint=False)
    off = rng.uniform(0, 0.35 * r, 300); be = rng.uniform(0, 2 * PI, 300)
    ringq = R + off * np.cos(be)
    Xq = np.stack([ringq * np.cos(aa), ringq * np.sin(aa), off * np.sin(be)], 1)
    uq = yukawa_gt(Xq, c, q, kappa)
    T = lambda a: torch.tensor(a, device=dev)
    Xc, nrmc, Xf, nrmf, wf, U, g, Xq, uq = map(T, [Xc, nrmc, Xf, nrmf, wf, U, g, Xq, uq])
    Sbase = qbx_laplace(Xc, nrmc, Xf, wf, beta, p) @ U        # coarse Laplace single-layer (N_c x N_c)
    # eval-kernel base (query x fine sources), Laplace single-layer, then @U -> (N_q x N_c)
    Dqf = torch.cdist(Xq, Xf).clamp_min(1e-15)
    Pbase = (1.0 / (4 * PI * Dqf) * wf[None, :]) @ U          # N_q x N_c
    fc = feats(Xc, Xc, nrmc, nrmc, v2=featv2)                 # N_c x N_c x 8|10
    fq = feats(Xq, Xc, torch.zeros_like(Xq), nrmc, v2=featv2) # N_q x N_c x 8|10
    return dict(Sbase=Sbase, Pbase=Pbase, g=g, uq=uq, fc=fc, fq=fq, wc=T(wc),
                Xc=Xc, nrmc=nrmc, Xq=Xq)


class Corr(nn.Module):
    def __init__(self, h=32, din=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):  # (...,din) -> (...)
        return self.net(f).squeeze(-1)


class DeepSet(nn.Module):
    def __init__(self, h=128, pc=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, pc))
        self.dec = nn.Sequential(nn.Linear(3 + pc, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, case):
        g = case["g"]
        gs = g.std().clamp_min(1e-12)                          # per-case input normalization
        nodes = torch.cat([case["Xc"], (g / gs)[:, None]], -1)
        code = self.enc(nodes).mean(0)
        Q = case["Xq"].shape[0]
        return self.dec(torch.cat([case["Xq"], code[None].expand(Q, -1)], -1)).squeeze(-1) * gs


def rel(u, uq):
    return torch.linalg.vector_norm(u - uq) / torch.linalg.vector_norm(uq).clamp_min(1e-30)


def solve_eval(case, corr):
    S = case["Sbase"]
    P = case["Pbase"]
    if corr is not None:
        S = S + corr(case["fc"]) * case["wc"][None, :]
        P = P + corr(case["fq"]) * case["wc"][None, :]
    S = S + 1e-10 * torch.eye(S.shape[0], device=S.device)
    sigma = torch.linalg.solve(S, case["g"])
    return P @ sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=24); ap.add_argument("--nph", type=int, default=12)
    ap.add_argument("--os", type=int, default=2); ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=3.0); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ntrain", type=int, default=12); ap.add_argument("--neval", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--featv2", action="store_true"); ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--cosine", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps = 20; a.ntrain = 3; a.neval = 2
    torch.set_default_dtype(torch.float64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} kappa={a.kappa} nt={a.nt} os={a.os} p={a.p}", flush=True)
    rng = np.random.default_rng(a.seed)

    def sample_shape(s):
        rr = np.random.default_rng(s)
        return dict(R=1.0, r=rr.uniform(0.28, 0.42), defo=rr.uniform(0.0, 0.18))

    print("[cache] building QBX bases...", flush=True)
    train = [build_case(a.nt, a.nph, a.os, a.p, 2.0, a.kappa, seed=100 + i, dev=dev, featv2=a.featv2,
                        **sample_shape(100 + i)) for i in range(a.ntrain)]
    ev = [build_case(a.nt, a.nph, a.os, a.p, 2.0, a.kappa, seed=900 + i, dev=dev, featv2=a.featv2,
                     **sample_shape(900 + i)) for i in range(a.neval)]

    # laplace_only (0 params): wrong-kernel floor.
    lo = float(np.mean([rel(solve_eval(c, None), c["uq"]).item() for c in ev]))
    print(f"[laplace_only]  eval zero-shot rel-L2 = {lo:.3e}  (wrong-kernel FLOOR)", flush=True)

    # laplace_corr: train the correction.
    corr = Corr(h=a.hidden, din=11 if a.featv2 else 8).to(dev)
    opt = torch.optim.Adam(corr.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=1e-5) if a.cosine else None
    for step in range(1, a.steps + 1):
        c = train[rng.integers(0, len(train))]
        loss = rel(solve_eval(c, corr), c["uq"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
        opt.step()
        if sched: sched.step()
        if step % max(1, a.steps // 5) == 0:
            print(f"  [corr] step {step}/{a.steps} train rel {loss.item():.3e}", flush=True)
    corr.eval()
    with torch.no_grad():
        lc = float(np.mean([rel(solve_eval(c, corr), c["uq"]).item() for c in ev]))
    print(f"[laplace_corr]  eval zero-shot rel-L2 = {lc:.3e}  (learned correction)", flush=True)

    # deepset black-box (stabilized: input norm, grad clip, cosine) — strongest fair version.
    ds = DeepSet().to(dev); opt2 = torch.optim.Adam(ds.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=a.steps, eta_min=1e-5) if a.cosine else None
    for step in range(1, a.steps + 1):
        c = train[rng.integers(0, len(train))]
        loss = rel(ds(c), c["uq"])
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ds.parameters(), 1.0)
        opt2.step()
        if sched2: sched2.step()
    ds.eval()
    with torch.no_grad():
        de = float(np.mean([rel(ds(c), c["uq"]).item() for c in ev]))
    print(f"[deepset]       eval zero-shot rel-L2 = {de:.3e}  (black-box)", flush=True)

    print(f"\nSUMMARY  laplace_only(floor)={lo:.3e}  laplace_corr={lc:.3e}  deepset={de:.3e}", flush=True)
    print(f"correction closed {100*(1-lc/lo):.1f}% of the wrong-kernel gap; "
          f"structured beats black-box {de/lc:.1f}x", flush=True)
    print("SMOKE_OK" if (lc < lo and lc < de) else "CHECK", flush=True)
    if a.out:
        import os, json
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(seed=a.seed, kappa=a.kappa, laplace_only=lo, laplace_corr=lc, deepset=de),
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
