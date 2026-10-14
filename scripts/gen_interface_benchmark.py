#!/usr/bin/env python
"""Head-to-head benchmark: parametric elliptic interface problem with jumps (IONet/XI-style, source-free).
Family varies outer + interface geometry, contrast, jump data, and boundary data; GT is a manufactured
piecewise-harmonic solution (exact). Stores a dataset for black-box baselines AND evaluates the NBIO-interface
STRUCTURE (the classical interface solver as a zero-parameter operator) across the family."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import boundary, sample_coeffs, eval_u, radius_at
from src.data.interface import interface_solve, eval_grad_n


def charges_at(rng, n, rlo, rhi):
    ang = rng.uniform(0, 2 * np.pi, n); rad = rng.uniform(rlo, rhi, n)
    q = rng.normal(size=n); q -= q.mean()
    return {"c": np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), "q": q}


def _signed_dist_incl(pts, center, ci):
    """>0 outside the (offset) inclusion, <0 inside."""
    v = pts - center[None, :]; ang = np.arctan2(v[:, 1], v[:, 0])
    return np.linalg.norm(v, axis=1) - radius_at(ci, ang).ravel()


def make_sample(rng, nbo=256, nbi=192, n_q=256, margin=0.15):
    co = sample_coeffs(rng, 4, amp=0.15, r0=1.6)
    rad = rng.uniform(0.28, 0.36); center = rng.uniform(-0.12, 0.12, 2)
    ci = sample_coeffs(rng, 3, amp=0.10, r0=rad)
    a_plus, a_minus = 1.0, float(np.exp(rng.uniform(np.log(2), np.log(8))))   # contrast 2..8
    ch_m = charges_at(rng, 6, rad * 1.6, rad * 2.4 + 0.4)
    ch_p = charges_at(rng, 6, rad * 0.15, rad * 0.55)
    G1 = boundary(ci, nbi); G1 = {**G1, "X": G1["X"] + center[None, :]}
    G0 = boundary(co, nbo)
    alpha = eval_u(G1["X"], ch_p) - eval_u(G1["X"], ch_m)
    beta = a_plus * eval_grad_n(G1["X"], G1["normal"], ch_p) - a_minus * eval_grad_n(G1["X"], G1["normal"], ch_m)
    g = eval_u(G0["X"], ch_p)
    # queries: region-correct with margin from BOTH interfaces (accounts for the OFFSET inclusion)
    nq2 = n_q // 2
    def collect(region):
        pts = []
        while len(pts) < nq2:
            cand = rng.uniform(-1.6, 1.6, (4 * nq2, 2))
            ang = np.arctan2(cand[:, 1], cand[:, 0])
            inside_out = np.linalg.norm(cand, axis=1) < (1 - margin) * radius_at(co, ang).ravel()
            sd = _signed_dist_incl(cand, center, ci)
            keep = inside_out & ((sd < -margin) if region == "minus" else (sd > margin))
            pts.extend(cand[keep].tolist())
        return np.array(pts[:nq2])
    Xm, Xp = collect("minus"), collect("plus")
    Xq = np.concatenate([Xm, Xp]); reg = np.array(["minus"] * nq2 + ["plus"] * nq2)
    uq = np.concatenate([eval_u(Xm, ch_m), eval_u(Xp, ch_p)])
    return dict(co=co, ci=ci, center=center, rad=rad, a_plus=a_plus, a_minus=a_minus,
                G0=G0, G1=G1, alpha=alpha, beta=beta, g=g, Xq=Xq, reg=reg, uq=uq,
                nbo=nbo, nbi=nbi)


def nbio_structure_rel(s):
    ev, _ = interface_solve({"r0": s["co"]["r0"], "a": s["co"]["a"], "b": s["co"]["b"]},
                            (s["center"], s["rad"], {"r0": s["ci"]["r0"], "a": s["ci"]["a"], "b": s["ci"]["b"]}),
                            s["a_plus"], s["a_minus"], s["g"], s["alpha"], s["beta"], s["nbo"], s["nbi"])
    pred = np.empty_like(s["uq"])
    m = s["reg"] == "minus"
    pred[m] = ev(s["Xq"][m], "minus"); pred[~m] = ev(s["Xq"][~m], "plus")
    return float(np.linalg.norm(pred - s["uq"]) / np.linalg.norm(s["uq"]))


def to_shard(samples):
    """Flatten to fixed-size tensors for black-box baselines: boundary tokens + jump/g + query coords."""
    keys = {}
    keys["Xb_out"] = np.stack([s["G0"]["X"] for s in samples]).astype(np.float32)
    keys["g"] = np.stack([s["g"] for s in samples]).astype(np.float32)
    keys["Xb_in"] = np.stack([s["G1"]["X"] for s in samples]).astype(np.float32)
    keys["Nb_in"] = np.stack([s["G1"]["normal"] for s in samples]).astype(np.float32)
    keys["alpha"] = np.stack([s["alpha"] for s in samples]).astype(np.float32)
    keys["beta"] = np.stack([s["beta"] for s in samples]).astype(np.float32)
    keys["logc"] = np.array([[np.log(s["a_minus"] / s["a_plus"])] for s in samples], np.float32)
    keys["Xq"] = np.stack([s["Xq"] for s in samples]).astype(np.float32)
    keys["uq"] = np.stack([s["uq"] for s in samples]).astype(np.float32)
    return {k: torch.from_numpy(v) for k, v in keys.items()}


def main():
    out = "data_cache"; os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2024)
    print("=== NBIO-interface STRUCTURE on the parametric interface family (zero-shot, no training) ===")
    test = [make_sample(rng) for _ in range(60)]
    errs = [nbio_structure_rel(s) for s in test]
    print(f"  NBIO-interface structure: mean rel-L2 {np.mean(errs):.3e}  median {np.median(errs):.3e}  "
          f"p90 {np.quantile(errs, 0.9):.3e}  (N=60)")
    # datasets for baselines
    rng = np.random.default_rng(11)
    torch.save(to_shard([make_sample(rng) for _ in range(800)]), os.path.join(out, "iface_train.pt"))
    torch.save(to_shard([make_sample(rng) for _ in range(150)]), os.path.join(out, "iface_val.pt"))
    torch.save(to_shard(test), os.path.join(out, "iface_test.pt"))
    import json
    json.dump({"model": "nbio_interface_structure", "n_params": 0,
               "eval": {"iface_test": float(np.mean(errs)), "median": float(np.median(errs))}},
              open("results/nbio_interface_structure_eval.json", "w"), indent=2)
    os.makedirs("results", exist_ok=True)
    print("  saved iface_{train,val,test}.pt")
    print("[gen_interface_benchmark] DONE")


if __name__ == "__main__":
    main()
