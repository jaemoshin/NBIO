#!/usr/bin/env python
"""Generate hard transmission stress shards for main-conference defensibility.

Shards:
  tm_test_close   : two near-touching inclusions with small interfacial gaps.
  tm_test_extreme : fixed N=2 but much higher contrast than training.

Both use the same tensor schema as the Stage-1 transmission data, so existing NBIO/DeepSet/Transformer
models can be evaluated without retraining.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import sample_coeffs, sample_charges, eval_u, radius_at
from src.data.transmission import build_curves, transmission_solve
from scripts.gen_transmission import sample_tm_queries, gen as gen_standard


def _close_inclusions(rng, outer_c):
    r_out_min = float(radius_at(outer_c, np.linspace(0, 2 * np.pi, 256)).min())
    rad = rng.uniform(0.17, 0.22) * r_out_min
    gap = rng.uniform(0.035, 0.07) * r_out_min
    sep = 2 * rad + gap
    base_ang = rng.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(base_ang), np.sin(base_ang)])
    mid_rad = rng.uniform(0.0, 0.23 * r_out_min)
    mid_ang = rng.uniform(0, 2 * np.pi)
    mid = mid_rad * np.array([np.cos(mid_ang), np.sin(mid_ang)])
    centers = [mid - 0.5 * sep * axis, mid + 0.5 * sep * axis]
    incls = []
    for c in centers:
        coeffs = sample_coeffs(rng, n_modes=3, amp=0.12, r0=rad)
        incls.append((c, rad, coeffs))
    return incls


def make_close_sample(rng, nbo=192, nbi=96, contrast=(0.25, 4.0), ref_res=512):
    for _ in range(200):
        outer_c = sample_coeffs(rng, 5)
        incls = _close_inclusions(rng, outer_c)
        r_out_min = float(radius_at(outer_c, np.linspace(0, 2 * np.pi, 256)).min())
        if all(np.linalg.norm(c) + 1.35 * rad < r_out_min for c, rad, _ in incls):
            break
    a_in = np.exp(rng.uniform(np.log(contrast[0]), np.log(contrast[1]), 2))
    ch = sample_charges(rng, "interior", outer_c, 6)
    curves = build_curves(outer_c, incls, nbo, nbi)
    g = eval_u(curves[0]["X"], ch)
    Xq = sample_tm_queries(rng, outer_c, incls, 256, margin=0.035)
    cr = build_curves(outer_c, incls, ref_res, max(48, ref_res // 2))
    gr = eval_u(cr[0]["X"], ch)
    _, evr = transmission_solve(cr, [1.0] + a_in.tolist(), gr)
    uq = evr(Xq)
    out = {
        "Xb_out": curves[0]["X"], "Nb_out": curves[0]["normal"], "wb_out": curves[0]["w"],
        "curv_out": curves[0]["curv"], "g": g,
        "Xb_in": np.stack([c["X"] for c in curves[1:]]), "Nb_in": np.stack([c["normal"] for c in curves[1:]]),
        "wb_in": np.stack([c["w"] for c in curves[1:]]), "curv_in": np.stack([c["curv"] for c in curves[1:]]),
        "a_in": a_in, "Xq": Xq, "uq": uq,
    }
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def gen_close(n, seed):
    rng = np.random.default_rng(seed)
    rows = [make_close_sample(rng) for _ in range(n)]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0].keys()}


def main():
    out = "data_cache"
    os.makedirs(out, exist_ok=True)
    torch.save(gen_close(120, 121), os.path.join(out, "tm_test_close.pt"))
    print("  tm_test_close: near-touching N=2 inclusions -> saved")
    torch.save(gen_standard(120, 2, 192, 96, (20.0, 50.0), 122, ref_res=512),
               os.path.join(out, "tm_test_extreme.pt"))
    print("  tm_test_extreme: zero-shot extreme contrast sigma 20-50 -> saved")
    print("[gen_transmission_stress] DONE")


if __name__ == "__main__":
    main()
