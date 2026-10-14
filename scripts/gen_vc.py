#!/usr/bin/env python
"""Stage 2 data: variable-coefficient div(a grad u)=f on the torus. Manufactured-solution validation gate,
then generate train/val/test + zero-shot (higher contrast, finer resolution) shards. GT = validated FD."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.vc import make_sample, validate_fd

BASE_N = 96
CONTRAST = 3.0


def gen(n_samples, n, contrast, seed):
    rng = np.random.default_rng(seed)
    rows = [make_sample(rng, n, contrast) for _ in range(n_samples)]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0].keys()}


def main():
    out = "data_cache"
    print("=== FD GT manufactured-solution validation (O(h^2)) ===")
    errs = {n: np.mean([validate_fd(s, n, CONTRAST) for s in range(3)]) for n in (48, 96, 192)}
    for n, e in errs.items():
        print(f"  validate_fd n={n}: rel-err vs manufactured u* = {e:.2e}")
    assert errs[192] < 5e-3, f"FD GT not validating ({errs[192]:.2e})"
    assert errs[48] > errs[192], "no O(h^2) convergence trend"
    print(f"  GT ceiling @n={BASE_N}: ~{errs[BASE_N]:.2e}")
    for split, n_s, seed in [("train", 800, 81), ("val", 150, 82), ("test", 150, 83)]:
        torch.save(gen(n_s, BASE_N, CONTRAST, seed), os.path.join(out, f"vc_{split}.pt"))
        print(f"  vc_{split}: N={n_s} n={BASE_N} contrast={CONTRAST} -> saved")
    torch.save(gen(150, BASE_N, 6.0, 91), os.path.join(out, "vc_test_hicon.pt"))
    print("  vc_test_hicon: zero-shot contrast=6 -> saved")
    for n in (64, 128):
        torch.save(gen(100, n, CONTRAST, 95), os.path.join(out, f"vc_testres{n}.pt"))
        print(f"  vc_testres{n}: zero-shot resolution -> saved")
    print("[gen_vc] DONE")


if __name__ == "__main__":
    main()
