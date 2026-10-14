#!/usr/bin/env python
"""Sanity for the transmission models: NBIO-tm @init ~ BEM (M1), well-conditioned, variable #inclusions;
nosplit ~ exact BEM; deepset_tm forward/backward."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.tm_loader import TMDataset
from src.data.loader import to_device
from src.models import build_model
from src.training.trainer import rel_l2
from src.utils.common import configure_torch

DATA = "data_cache"


def batch(shard, bs=8, dev="cpu"):
    ds = TMDataset(os.path.join(DATA, shard))
    return to_device(next(iter(DataLoader(ds, batch_size=bs))), torch.device(dev))


def main():
    configure_torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = batch("tm_test.pt", 8, dev)
    nbio = build_model({"model": {"name": "nbio_tm", "kwargs": {"hidden": 64, "n_layers": 3, "split": True}}}).to(dev)
    nbio.eval()
    with torch.no_grad():
        r0 = rel_l2(nbio(b), b["uq"]).item()
    cond = nbio.cond_number(b)
    print(f"[sanity] NBIO-tm @init rel-L2={r0:.2e} (should be ~BEM)  cond(A)={cond:.1f}")
    nosplit = build_model({"model": {"name": "nbio_tm", "kwargs": {"split": False}}}).to(dev).eval()
    with torch.no_grad():
        rns = rel_l2(nosplit(b), b["uq"]).item()
    print(f"[sanity] NBIO-tm nosplit (pure known BEM) @init rel-L2={rns:.2e} (should be ~machine/float32)")
    assert rns < 1e-2, f"torch known-kernel BEM disagrees with numpy GT ({rns:.2e})"
    # variable #inclusions
    for shard in ("tm_test_N1.pt", "tm_test_N3.pt", "tm_test_N4.pt"):
        bb = batch(shard, 6, dev)
        with torch.no_grad():
            rr = rel_l2(nbio(bb), bb["uq"]).item()
        print(f"[sanity] NBIO-tm @init {shard:16s} (N={bb['a_in'].shape[1]}) rel-L2={rr:.2e}")
    # baseline forward/backward
    m = build_model({"model": {"name": "deepset_tm", "kwargs": {"hidden": 64, "p": 64}}}).to(dev)
    m.train(); loss = rel_l2(m(b), b["uq"]); loss.backward()
    gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert gn > 0
    print(f"[sanity] deepset_tm OK params={sum(p.numel() for p in m.parameters())} init-loss {loss.item():.3e}")
    # NBIO-tm backward
    nbio.train(); loss = rel_l2(nbio(b), b["uq"]); loss.backward()
    gn = sum(p.grad.abs().sum().item() for p in nbio.parameters() if p.grad is not None)
    assert gn > 0 and all(torch.isfinite(p.grad).all() for p in nbio.parameters() if p.grad is not None)
    print(f"[sanity] NBIO-tm backward OK (grad-norm {gn:.2e})  params={sum(p.numel() for p in nbio.parameters())}")
    print("[sanity_tm] ALL PASSED")


if __name__ == "__main__":
    main()
