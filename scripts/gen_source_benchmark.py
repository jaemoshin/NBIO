#!/usr/bin/env python
"""Full inhomogeneous interface head-to-head: -div(a grad u)=f with jumps [u],[a u_n] (IONet/XI/BENO-style,
WITH volume source). Family varies geometry, contrast, source (Gaussians), jumps, boundary data; GT is the
manufactured u = phi/a_k + u_h. Evaluates NBIO-interface-source STRUCTURE and stores a source-aware dataset
(f sampled at tokens + queries) for black-box baselines."""
from __future__ import annotations

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.data.bie import boundary, sample_coeffs, eval_u, radius_at
from src.data.interface import eval_grad_n
from src.data.volume_source import gaussians, phi_and_f, inhomogeneous_interface_solve


def charges_at(rng, n, rlo, rhi):
    ang = rng.uniform(0, 2 * np.pi, n); rad = rng.uniform(rlo, rhi, n)
    q = rng.normal(size=n); q -= q.mean()
    return {"c": np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), "q": q}


def signed_dist(pts, center, ci):
    v = pts - center[None, :]; ang = np.arctan2(v[:, 1], v[:, 0])
    return np.linalg.norm(v, axis=1) - radius_at(ci, ang).ravel()


def make_sample(rng, nbo=256, nbi=192, n_q=256, margin=0.15):
    co = sample_coeffs(rng, 4, amp=0.15, r0=1.6)
    rad = rng.uniform(0.28, 0.36); center = rng.uniform(-0.12, 0.12, 2)
    ci = sample_coeffs(rng, 3, amp=0.10, r0=rad)
    a_plus, a_minus = 1.0, float(np.exp(rng.uniform(np.log(2), np.log(8))))
    cg, Ag, sg = gaussians(rng, 5)
    f = lambda X: phi_and_f(X, cg, Ag, sg)[1]; phi = lambda X: phi_and_f(X, cg, Ag, sg)[0]
    ch_m = charges_at(rng, 5, rad * 1.6, rad * 2.4 + 0.4); ch_p = charges_at(rng, 5, rad * 0.15, rad * 0.55)
    G1 = boundary(ci, nbi); G1 = {**G1, "X": G1["X"] + center[None, :]}; G0 = boundary(co, nbo)
    alpha = (phi(G1["X"]) / a_plus + eval_u(G1["X"], ch_p)) - (phi(G1["X"]) / a_minus + eval_u(G1["X"], ch_m))
    beta = a_plus * eval_grad_n(G1["X"], G1["normal"], ch_p) - a_minus * eval_grad_n(G1["X"], G1["normal"], ch_m)
    g = phi(G0["X"]) / a_plus + eval_u(G0["X"], ch_p)
    # region-correct queries with margin
    nq2 = n_q // 2
    def collect(reg):
        pts = []
        while len(pts) < nq2:
            cand = rng.uniform(-1.6, 1.6, (4 * nq2, 2)); ang = np.arctan2(cand[:, 1], cand[:, 0])
            ins = np.linalg.norm(cand, axis=1) < (1 - margin) * radius_at(co, ang).ravel()
            sd = signed_dist(cand, center, ci)
            keep = ins & ((sd < -margin) if reg == "minus" else (sd > margin))
            pts.extend(cand[keep].tolist())
        return np.array(pts[:nq2])
    Xm, Xp = collect("minus"), collect("plus")
    Xq = np.concatenate([Xm, Xp]); reg = np.array(["minus"] * nq2 + ["plus"] * nq2)
    uq = np.concatenate([phi(Xm) / a_minus + eval_u(Xm, ch_m), phi(Xp) / a_plus + eval_u(Xp, ch_p)])
    return dict(co=co, ci=ci, center=center, rad=rad, a_plus=a_plus, a_minus=a_minus, f=f,
                G0=G0, G1=G1, alpha=alpha, beta=beta, g=g, Xq=Xq, reg=reg, uq=uq, nbo=nbo, nbi=nbi)


def nbio_structure_rel(s):
    ev = inhomogeneous_interface_solve({"r0": s["co"]["r0"], "a": s["co"]["a"], "b": s["co"]["b"]},
                                       (s["center"], s["rad"], {"r0": s["ci"]["r0"], "a": s["ci"]["a"], "b": s["ci"]["b"]}),
                                       s["a_plus"], s["a_minus"], s["f"], s["g"], s["alpha"], s["beta"], s["nbo"], s["nbi"])
    pred = np.empty_like(s["uq"]); m = s["reg"] == "minus"
    pred[m] = ev(s["Xq"][m], "minus"); pred[~m] = ev(s["Xq"][~m], "plus")
    return float(np.linalg.norm(pred - s["uq"]) / np.linalg.norm(s["uq"]))


def to_shard(samples):
    t = lambda L: torch.from_numpy(np.stack(L).astype(np.float32))
    return {"Xb_out": t([s["G0"]["X"] for s in samples]), "g": t([s["g"] for s in samples]),
            "f_out": t([s["f"](s["G0"]["X"]) for s in samples]),
            "Xb_in": t([s["G1"]["X"] for s in samples]), "Nb_in": t([s["G1"]["normal"] for s in samples]),
            "alpha": t([s["alpha"] for s in samples]), "beta": t([s["beta"] for s in samples]),
            "f_in": t([s["f"](s["G1"]["X"]) for s in samples]),
            "logc": t([[np.log(s["a_minus"] / s["a_plus"])] for s in samples]),
            "Xq": t([s["Xq"] for s in samples]), "fq": t([s["f"](s["Xq"]) for s in samples]),
            "uq": t([s["uq"] for s in samples])}


def main():
    os.makedirs("data_cache", exist_ok=True); os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(4040)
    test = [make_sample(rng) for _ in range(60)]
    errs = [nbio_structure_rel(s) for s in test]
    print(f"NBIO-interface-SOURCE structure: mean {np.mean(errs):.3e} median {np.median(errs):.3e} "
          f"p90 {np.quantile(errs, 0.9):.3e} (N=60)")
    json.dump({"model": "nbio_interface_source_structure", "n_params": 0,
               "eval": {"src_test": float(np.mean(errs)), "median": float(np.median(errs))}},
              open("results/nbio_interface_source_structure_eval.json", "w"), indent=2)
    rng = np.random.default_rng(55)
    torch.save(to_shard([make_sample(rng) for _ in range(800)]), "data_cache/src_train.pt")
    torch.save(to_shard([make_sample(rng) for _ in range(150)]), "data_cache/src_val.pt")
    torch.save(to_shard(test), "data_cache/src_test.pt")
    print("saved src_{train,val,test}.pt  [gen_source_benchmark] DONE")


if __name__ == "__main__":
    main()
