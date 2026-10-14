#!/usr/bin/env python
"""Modified-Helmholtz (Yukawa) interior splits: Delta u - kappa^2 u = 0. The 'wrong-kernel' stress test:
NBIO will be given only the LAPLACE known kernel, so its smooth correction must learn the Yukawa
difference. We also store `laplace_bem_uq` = classical Laplace-BEM applied to the Yukawa boundary data
(the FIXED wrong kernel) -> shows the classical fixed kernel fails where the learned correction succeeds.
GT (`uq`) and the correct Yukawa-BEM (`bem_uq`) are exact/spectral."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.bie import BIEConfig, make_sample, bem_solve, validate_bem

KAPPA = 4.0


def gen(n, cfg, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        s = make_sample(rng, cfg)
        # wrong fixed kernel: classical LAPLACE BEM on the Yukawa boundary data g
        bd = {"X": s["Xb"], "normal": s["Nb"], "w": s["wb"], "curv": s["curv"]}
        try:
            _, ev = bem_solve(bd, s["g"], cfg.region, kappa=0.0)
            s["laplace_bem_uq"] = ev(s["Xq"]).astype(np.float32)
        except np.linalg.LinAlgError:
            s["laplace_bem_uq"] = np.full_like(s["uq"], np.nan)
        rows.append(s)
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0].keys()}


def main():
    out = "data_cache"
    e = np.mean([validate_bem(seed=s, n_b=256, region="interior", kappa=KAPPA) for s in range(5)])
    print(f"Yukawa(kappa={KAPPA}) BEM vs exact (n_b=256): rel-err {e:.2e}")
    assert e < 1e-3, f"Yukawa BEM not validating ({e:.2e})"
    cfg = BIEConfig(region="interior", n_b=128, n_q=256, kappa=KAPPA)
    for split, n, seed in [("train", 1200, 31), ("val", 200, 32), ("test", 200, 33)]:
        d = gen(n, cfg, seed)
        torch.save(d, os.path.join(out, f"yukawa_{split}.pt"))
        def rel(key):
            v = d[key]; fin = torch.isfinite(v).all(dim=1)
            return (torch.linalg.vector_norm(v[fin] - d["uq"][fin], dim=1)
                    / (torch.linalg.vector_norm(d["uq"][fin], dim=1) + 1e-9)).mean().item()
        print(f"  yukawa_{split}: N={n}  Yukawa-BEM rel-L2={rel('bem_uq'):.2e}  "
              f"WRONG Laplace-BEM rel-L2={rel('laplace_bem_uq'):.2e}")
    print("[gen_yukawa] DONE")


if __name__ == "__main__":
    main()
