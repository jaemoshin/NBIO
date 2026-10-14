#!/usr/bin/env python
"""Sanity: all models forward/backward + key NBIO structural checks (M1 init~BEM, well-conditioned 2nd-kind
solve, variable-n_b zero-shot)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.loader import BIEDataset, to_device
from src.models import build_model
from src.training.trainer import rel_l2
from src.utils.common import configure_torch

DATA = "data_cache"


def batch_from(shard, bs=8, device="cpu"):
    ds = BIEDataset(os.path.join(DATA, shard))
    return to_device(next(iter(DataLoader(ds, batch_size=bs))), torch.device(device))


def main():
    configure_torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = batch_from("interior_test.pt", 8, dev)

    # ---- NBIO: M1 init ~ classical BEM (gate=1, smooth~0 at init -> pure double-layer Nystrom) ----
    nbio = build_model({"model": {"name": "nbio", "kwargs": {"hidden": 64, "n_layers": 3, "split": True}}}).to(dev)
    nbio.eval()
    with torch.no_grad():
        r0 = rel_l2(nbio(b), b["uq"]).item()
    cond = nbio.condition_number(b)
    print(f"[sanity] NBIO @init rel-L2={r0:.2e} (should be ~BEM/classical, <1e-2)  cond(A)={cond:.1f}")
    assert r0 < 5e-2, f"NBIO init not near classical BEM (rel-L2 {r0:.2e}) -> M1 known-kernel broken"
    assert cond < 50, f"assembled 2nd-kind system not well-conditioned (cond {cond:.1f})"

    # ---- variable boundary resolution (zero-shot n_b): same operator, different n_b ----
    for shard in ("interior_testres64.pt", "interior_testres256.pt"):
        bb = batch_from(shard, 8, dev)
        with torch.no_grad():
            rr = rel_l2(nbio(bb), bb["uq"]).item()
        print(f"[sanity] NBIO @init on {shard:24s} (n_b={bb['Xb'].shape[1]}) rel-L2={rr:.2e} finite={torch.isfinite(torch.tensor(rr))}")

    # ---- all models: forward shape + finite + backward ----
    specs = {
        "nbio": {"hidden": 32, "n_layers": 2, "split": True},
        "nbio_nosplit": {"hidden": 32, "n_layers": 2, "split": False},
        "nbio_neumann": {"hidden": 32, "n_layers": 2, "split": True, "solve": "neumann", "n_iter": 15},
        "deeponet": {"n_sensors": 128, "p": 64, "hidden": 128, "n_layers": 3},
        "deepset": {"hidden": 64, "p": 64},
    }
    for name, kw in specs.items():
        reg = "nbio" if name.startswith("nbio") else name
        m = build_model({"model": {"name": reg, "kwargs": kw}}).to(dev)
        m.train()
        pred = m(b)
        assert pred.shape == b["uq"].shape and torch.isfinite(pred).all(), f"{name} bad forward"
        loss = rel_l2(pred, b["uq"]); loss.backward()
        gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        assert gn > 0, f"{name} no grad"
        print(f"[sanity] {name:14s} OK params={sum(p.numel() for p in m.parameters()):7d} init-loss {loss.item():.3e}")
    print("[sanity] ALL PASSED")


if __name__ == "__main__":
    main()
