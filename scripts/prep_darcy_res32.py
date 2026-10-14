#!/usr/bin/env python
"""Higher-resolution (res 32) variant of the released Darcy benchmark, for a head-to-head closer to the
canonical FNO Darcy regime than the coarse res-16 quickstart split. Uses the OFFICIAL released res-32
tensors (Zenodo 12784353, darcy_{train,test}_32.pt; x=coefficient a, y=solution u). Same provenance &
caveats as prep_darcy_benchmark.py. Builds res-32 train/val/test + a res-64 zero-shot super-resolution OOD
shard, in TMDataset format {a,f,u,ulap}. ulap = const-coeff (-abar*Lap, zero-Dirichlet, sine-basis) prior."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.prep_darcy_benchmark import build_shard, load_released, save  # reuse the validated builders

DC = "data_cache"


def main():
    print("=== Darcy res-32 benchmark prep (official released tensors) ===")
    a_tr, u_tr = load_released("train", 32)        # 5000 released train
    a_te, u_te = load_released("test", 32)         # 1000 released test
    n_train, n_val = 1000, 200                     # modest split for the 12GB budget
    perm = np.random.default_rng(0).permutation(a_tr.shape[0])
    tr_idx = perm[:n_train]; va_idx = perm[n_train:n_train + n_val]
    save(build_shard(a_tr[tr_idx], u_tr[tr_idx]), "darcy32_train.pt")
    save(build_shard(a_tr[va_idx], u_tr[va_idx]), "darcy32_val.pt")
    save(build_shard(a_te[:200], u_te[:200]), "darcy32_test.pt")
    # zero-shot super-resolution OOD: res 64
    a64, u64 = load_released("test", 64)
    save(build_shard(a64[:200], u64[:200]), "darcy32_testres64.pt")
    print("[prep_darcy_res32] DONE")


if __name__ == "__main__":
    main()
