#!/usr/bin/env python
"""Generate the transmission (piecewise-constant conductivity) dataset. GT = validated multi-region
Nystrom BEM. Fixed N=2 inclusions for the core (batchable); zero-shot shards vary #inclusions, contrast,
and boundary resolution. Reports BEM self-convergence (the achievable ceiling)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import sample_coeffs, sample_charges, eval_u, radius_at
from src.data.transmission import sample_inclusions, build_curves, transmission_solve


def _incl_signed_dist(pt, center, coeffs):
    v = pt - center; ang = np.arctan2(v[1], v[0])
    return np.linalg.norm(v) - float(radius_at(coeffs, np.array([ang]))[0])


def sample_tm_queries(rng, outer_c, incls, n_q, margin=0.06):
    r_out_min = float(radius_at(outer_c, np.linspace(0, 2 * np.pi, 256)).min())
    pts = []
    while len(pts) < n_q:
        ang = rng.uniform(0, 2 * np.pi, 4 * n_q)
        rad = np.sqrt(rng.uniform(0, 1, 4 * n_q)) * 0.93 * r_out_min
        cand = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        for p in cand:
            ang_p = np.arctan2(p[1], p[0])
            if np.linalg.norm(p) > 0.93 * float(radius_at(outer_c, np.array([ang_p]))[0]):
                continue
            if all(abs(_incl_signed_dist(p, c, cc)) > margin for (c, _, cc) in incls):
                pts.append(p)
                if len(pts) >= n_q:
                    break
    return np.array(pts[:n_q])


def make_tm_sample(rng, N, nbo, nbi, con_lo, con_hi, ref_res=None):
    while True:
        outer_c = sample_coeffs(rng, 5)
        incls = sample_inclusions(rng, N, outer_c)
        if len(incls) == N:
            break
    a_in = np.exp(rng.uniform(np.log(con_lo), np.log(con_hi), N))   # contrasts sigma_k = a_k/a0, a0=1
    ch = sample_charges(rng, "interior", outer_c, 6)               # random smooth harmonic boundary data
    curves = build_curves(outer_c, incls, nbo, nbi)
    g = eval_u(curves[0]["X"], ch)
    Xq = sample_tm_queries(rng, outer_c, incls, 256)
    _, ev = transmission_solve(curves, [1.0] + a_in.tolist(), g)
    uq = ev(Xq)
    if ref_res is not None:   # high-res reference (resolution test): recompute GT at ref_res
        cr = build_curves(outer_c, incls, ref_res, max(48, ref_res // 2))
        gr = eval_u(cr[0]["X"], ch); _, evr = transmission_solve(cr, [1.0] + a_in.tolist(), gr)
        uq = evr(Xq)
    out = {
        "Xb_out": curves[0]["X"], "Nb_out": curves[0]["normal"], "wb_out": curves[0]["w"],
        "curv_out": curves[0]["curv"], "g": g,
        "Xb_in": np.stack([c["X"] for c in curves[1:]]), "Nb_in": np.stack([c["normal"] for c in curves[1:]]),
        "wb_in": np.stack([c["w"] for c in curves[1:]]), "curv_in": np.stack([c["curv"] for c in curves[1:]]),
        "a_in": a_in, "Xq": Xq, "uq": uq,
    }
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def gen(n, N, nbo, nbi, con, seed, ref_res=None):
    rng = np.random.default_rng(seed)
    rows = [make_tm_sample(rng, N, nbo, nbi, con[0], con[1], ref_res) for _ in range(n)]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0].keys()}


def bem_selfconv(N=2, nbo=192, nbi=96, seed=99, n=20):
    """BEM ceiling estimate: rel diff between BEM@(nbo,nbi) and BEM@(2nbo,2nbi) on the same geometries."""
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        while True:
            oc = sample_coeffs(rng, 5); inc = sample_inclusions(rng, N, oc)
            if len(inc) == N: break
        a_in = np.exp(rng.uniform(np.log(0.25), np.log(4), N)); ch = sample_charges(rng, "interior", oc, 6)
        Xq = sample_tm_queries(rng, oc, inc, 200)
        us = {}
        for f in (1, 2):
            cv = build_curves(oc, inc, nbo * f, nbi * f); g = eval_u(cv[0]["X"], ch)
            _, ev = transmission_solve(cv, [1.0] + a_in.tolist(), g); us[f] = ev(Xq)
        diffs.append(np.linalg.norm(us[1] - us[2]) / np.linalg.norm(us[2]))
    return float(np.mean(diffs))


def main():
    out = "data_cache"; NBO, NBI = 192, 96; CON = (0.25, 4.0)
    print(f"BEM self-convergence ceiling @(nbo={NBO},nbi={NBI}): {bem_selfconv(2, NBO, NBI):.2e}")
    for split, n, seed in [("train", 1000, 41), ("val", 200, 42), ("test", 200, 43)]:
        torch.save(gen(n, 2, NBO, NBI, CON, seed), os.path.join(out, f"tm_{split}.pt"))
        print(f"  tm_{split}: N=2 nbo={NBO} -> saved")
    for N, n, seed in [(1, 150, 51), (3, 150, 52), (4, 120, 53)]:
        torch.save(gen(n, N, NBO, NBI, CON, seed), os.path.join(out, f"tm_test_N{N}.pt"))
        print(f"  tm_test_N{N}: zero-shot #inclusions -> saved")
    torch.save(gen(150, 2, NBO, NBI, (6.0, 12.0), 61), os.path.join(out, "tm_test_hicon.pt"))
    print("  tm_test_hicon: zero-shot high contrast (sigma 6-12) -> saved")
    for nbo in (96, 192, 384):
        torch.save(gen(100, 2, nbo, max(48, nbo // 2), CON, 71, ref_res=512), os.path.join(out, f"tm_testres{nbo}.pt"))
        print(f"  tm_testres{nbo}: GT=BEM@512 ref -> saved")
    print("[gen_transmission] DONE")


if __name__ == "__main__":
    main()
