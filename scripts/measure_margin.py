#!/usr/bin/env python
"""Full audit of the discrete perturbation margin ||A_{0,h}^{-1} T_{theta,h}||_2 for the trained
wrong-kernel (Laplace-prior) NBIO, using the COMPLETE learned perturbation T_theta = gamma*K_0 + K_corr.

Population: 3 seeds x 4 geometry families x n_b in {64,128,256} x 64 geometries each.
Also reports the LOOSE product bound ||A_0^{-1}||_2 * ||T_theta||_2 (an upper bound on the margin, to
reconcile with the earlier ~2.6 diagnostic) and cond(A_theta).

A_0  = jump I + W K_0 (gate coeff 1, no corr);  A_th = jump I + W((1+gamma)K_0 + K_corr).
margin = ||A_0^{-1} T_theta||_2 ;  product bound = ||A_0^{-1}||_2 ||T_theta||_2 >= margin.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model                       # noqa: E402
from src.data.loader import BIEDataset                   # noqa: E402

torch.set_default_dtype(torch.float64)
DATA = ROOT / "data_cache" / "tmlr_revision"
CKPTS = {0: ROOT / "results/tmlr_revision/nbio_yuk_split_s0/best.pt",
         1: ROOT / "results/nbio_yuk_split_s1/best.pt",
         2: ROOT / "results/nbio_yuk_split_s2/best.pt"}
FAMILIES = ["in_dist", "rough", "deform", "highmode"]
NBS = [64, 128, 256]


class _ZeroSmooth(torch.nn.Module):
    def __init__(self, real):
        super().__init__(); self.real = real
    def forward(self, *a, **k):
        return torch.zeros_like(self.real(*a, **k))


def assemble(model, b, gate_one=False, zero_corr=False):
    bio = model.bio
    g0 = bio.log_gate.data.clone(); sm0 = bio.smooth
    if gate_one:
        bio.log_gate.data.zero_()
    if zero_corr:
        bio.smooth = _ZeroSmooth(sm0)
    try:
        return bio.assemble(b["Xb"], b["Nb"], b["wb"], b["curv"], model.jump)
    finally:
        bio.log_gate.data.copy_(g0); bio.smooth = sm0


def load_batch(shard):
    ds = BIEDataset(str(shard))
    keys = ["Xb", "Nb", "wb", "curv"]
    return {k: torch.stack([ds[i][k] for i in range(len(ds))]).double() for k in keys}


def main():
    all_margin, all_prod, all_cond, all_frac = [], [], [], []
    per_nb = {nb: [] for nb in NBS}
    print(f"{'seed':>4s} {'family':10s} {'n_b':>4s} {'n':>3s} {'margin max':>10s} "
          f"{'prod-bnd max':>12s} {'cond max':>9s}")
    with torch.no_grad():
        for seed, ckpt in CKPTS.items():
            if not ckpt.exists():
                print(f"seed {seed}: checkpoint missing ({ckpt})"); continue
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = build_model(ck["config"]).double()
            model.load_state_dict(ck["state_dict"]); model.eval()
            gamma = float(torch.tanh(model.bio.log_gate.data))
            print(f"# seed {seed}: gamma = {gamma:+.4f}")
            for fam in FAMILIES:
                for nb in NBS:
                    shard = DATA / f"yukawa_k4_{fam}_m120_nb{nb}.pt"
                    if not shard.exists():
                        continue
                    b = load_batch(shard)
                    A_th = assemble(model, b)
                    A_0 = assemble(model, b, gate_one=True, zero_corr=True)
                    A_g = assemble(model, b, gate_one=False, zero_corr=True)   # gate on, corr off
                    T = A_th - A_0
                    T_gam = A_g - A_0                                          # gamma*K_0 part only
                    frac = (torch.linalg.matrix_norm(T_gam, ord="fro")
                            / torch.linalg.matrix_norm(T, ord="fro").clamp_min(1e-30))
                    all_frac.append(frac.cpu().numpy())
                    margin = torch.linalg.matrix_norm(torch.linalg.solve(A_0, T), ord=2)
                    A0inv = torch.linalg.matrix_norm(torch.linalg.inv(A_0), ord=2)
                    Tn = torch.linalg.matrix_norm(T, ord=2)
                    prod = A0inv * Tn
                    cond = (torch.linalg.matrix_norm(A_th, ord=2)
                            * torch.linalg.matrix_norm(torch.linalg.inv(A_th), ord=2))
                    m = margin.cpu().numpy()
                    all_margin.append(m); all_prod.append(prod.cpu().numpy())
                    all_cond.append(cond.cpu().numpy()); per_nb[nb].append(m)
                    print(f"{seed:>4d} {fam:10s} {nb:>4d} {len(m):>3d} {m.max():10.3f} "
                          f"{prod.cpu().numpy().max():12.3f} {cond.cpu().numpy().max():9.3f}")
    M = np.concatenate(all_margin); P = np.concatenate(all_prod); C = np.concatenate(all_cond)
    print("\n=== OVERALL (3 seeds x 4 families x n_b in {64,128,256}, 64 geoms each) ===")
    print(f"  cases: {M.size}")
    print(f"  direct margin ||A_0^-1 T||_2:   mean {M.mean():.3f}  median {np.median(M):.3f}  "
          f"p90 {np.percentile(M,90):.3f}  MAX {M.max():.3f}")
    print(f"  loose product bound ||A_0^-1||*||T||: mean {P.mean():.3f}  MAX {P.max():.3f}  "
          f"(>= margin; explains a larger earlier bound)")
    print(f"  cond(A_theta): mean {C.mean():.3f}  max {C.max():.3f}")
    F=np.concatenate(all_frac); print(f"  gamma*K0 fraction ||gK0||/||T||: mean {F.mean():.3%}  max {F.max():.3%}  (all seeds)")
    for nb in NBS:
        if per_nb[nb]:
            mm = np.concatenate(per_nb[nb]); print(f"  n_b={nb}: margin max {mm.max():.3f}")
    print(f"\n  >>> max direct margin over the ENTIRE audit = {M.max():.3f} (< 1)")


if __name__ == "__main__":
    main()
