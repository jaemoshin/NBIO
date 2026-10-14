#!/usr/bin/env python
"""Convert the OFFICIAL released Darcy-flow benchmark tensors (neuraloperator, Zenodo record 12784353,
files darcy_{16,32,64,128}.tgz -> darcy_{train,test}_{res}.pt with keys x=coefficient a (thresholded GRF),
y=solution u) into this repo's TMDataset shard format so the existing fno / green_fno models train on them
UNCHANGED.

PROVENANCE (honest):
  * a (=x) and u (=y) are the REAL released tensors of the canonical FNO Darcy benchmark
    (Li et al. 2021, -div(a grad u)=f on the unit square, zero-Dirichlet BC, a a thresholded Gaussian
    random field, f=1; finite-difference reference solution). Downloaded from the official neuraloperator
    Zenodo mirror, NOT regenerated.
  * The hosted u tensors were produced at native res 421 and down-sampled / output-normalised by the
    neuraloperator pipeline (the stored u is therefore NOT bit-exactly the textbook FD solve at the stored
    grid, and its boundary trace is O(1e-1) rather than exactly 0). We use (a, u) AS RELEASED for an
    identical-data head-to-head: both FNO and green_fno see the same (a, u) pairs, so the comparison is fair
    and the provenance is the recognised benchmark.
  * f is set to the benchmark constant 1.
  * ulap = (-abar * Laplacian)^-1 f via FFT = the constant-coefficient Green's prior (the "wrong fixed
    kernel") that green_fno corrects. It is computed FROM the released coefficient a; it is an input feature,
    NOT ground truth.

Output shards (data_cache/):
  darcy_train.pt / darcy_val.pt / darcy_test.pt   -- res 16 (canonical FNO-small split, 1000 train)
  darcy_testres32.pt / darcy_testres64.pt / darcy_testres128.pt  -- zero-shot super-resolution OOD
Each shard: dict of stacked float32 tensors {a:(N,H,W), f:(N,H,W), u:(N,H,W), ulap:(N,H,W)}.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DC = "data_cache"
F_CONST = 1.0


def fft_const_prior(a, f_const=F_CONST):
    """ulap = (-abar Laplacian)^-1 f on the unit square with ZERO-DIRICHLET BC via the sine (DST-I)
    eigenbasis -- the constant-coefficient Green's prior for the canonical Darcy BC (periodic FFT is
    degenerate for a constant f, so we use the Dirichlet sine transform). a:(H,W) -> abar=mean(a); this is
    the fixed-kernel structure-informed prior green_fno corrects, an input feature (NOT ground truth)."""
    from scipy.fft import dstn, idstn
    H, W = a.shape
    abar = float(a.mean()) if a.mean() > 1e-6 else 1.0
    f = np.full((H, W), f_const, dtype=np.float64)
    h = 1.0 / (H - 1)                     # unit square, node grid; interior eigenvalues of -Laplacian (5-pt)
    fi = f[1:-1, 1:-1]                    # interior (Dirichlet zero on boundary)
    n = fi.shape[0]
    fh = dstn(fi, type=1, norm="ortho")
    i = np.arange(1, n + 1)
    lx = (2 - 2 * np.cos(np.pi * i / (n + 1))) / h ** 2
    LX, LY = np.meshgrid(lx, lx, indexing="ij")
    uh = fh / (abar * (LX + LY))
    ui = idstn(uh, type=1, norm="ortho")
    u = np.zeros((H, W)); u[1:-1, 1:-1] = ui
    return u.astype(np.float32)


def load_released(split, res):
    p = os.path.join(DC, f"darcy_{split}_{res}.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    a = d["x"].float().numpy()           # (N,H,W) coefficient (bool@16 -> {0,1}; float elsewhere {0,1})
    u = d["y"].float().numpy()           # (N,H,W) released solution
    return a, u


def build_shard(a, u):
    N = a.shape[0]
    f = np.full_like(a, F_CONST)
    ulap = np.stack([fft_const_prior(a[i]) for i in range(N)]).astype(np.float32)
    return {"a": torch.from_numpy(a.astype(np.float32)),
            "f": torch.from_numpy(f.astype(np.float32)),
            "u": torch.from_numpy(u.astype(np.float32)),
            "ulap": torch.from_numpy(ulap)}


def save(d, name):
    torch.save(d, os.path.join(DC, name))
    a = d["a"]; u = d["u"]; ul = d["ulap"]
    print(f"  {name:22s} N={a.shape[0]:5d} res={a.shape[1]}  "
          f"a[{float(a.min()):.2f},{float(a.max()):.2f}] u[{float(u.min()):.2f},{float(u.max()):.2f}] "
          f"ulap[{float(ul.min()):.3f},{float(ul.max()):.3f}]")


def main():
    print("=== Darcy benchmark prep (official released tensors -> TMDataset shards) ===")
    # In-distribution res 16 (canonical FNO-small). Released: 1000 train, 50 test. Carve a val from train.
    a16_tr, u16_tr = load_released("train", 16)
    a16_te, u16_te = load_released("test", 16)
    n_val = 100
    perm = np.random.default_rng(0).permutation(a16_tr.shape[0])
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    save(build_shard(a16_tr[tr_idx], u16_tr[tr_idx]), "darcy_train.pt")
    save(build_shard(a16_tr[va_idx], u16_tr[va_idx]), "darcy_val.pt")
    save(build_shard(a16_te, u16_te), "darcy_test.pt")
    # Zero-shot super-resolution OOD: evaluate the 16-trained model on finer grids (canonical Darcy generalisation).
    for res in (32, 64, 128):
        try:
            a, u = load_released("test", res)
        except FileNotFoundError:
            print(f"  (skip res {res}: file missing)")
            continue
        cap = min(200, a.shape[0])       # keep OOD eval shards modest for the 12GB budget
        save(build_shard(a[:cap], u[:cap]), f"darcy_testres{res}.pt")
    print("[prep_darcy_benchmark] DONE")


if __name__ == "__main__":
    main()
