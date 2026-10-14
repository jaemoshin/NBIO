#!/usr/bin/env python
"""Self-contained smoke tests for core NBIO dataset/model paths.

Generates tiny manufactured samples in memory, then runs CPU/GPU-safe forwards
and solves. This is not a training/evaluation benchmark.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.gen_transmission import make_tm_sample
from src.data.bie import BIEConfig, make_sample
from src.data.loader import to_device
from src.models import build_model


def stack(samples):
    return {key: torch.from_numpy(np.stack([sample[key] for sample in samples])) for key in samples[0]}


def one(b, model_cfg, target_key, device):
    b = to_device(b, device)
    m = build_model(model_cfg).to(device).eval()
    with torch.no_grad():
        y = m(b)
    assert y.shape == b[target_key].shape, (y.shape, b[target_key].shape)
    assert torch.isfinite(y).all()
    return float(torch.linalg.vector_norm(y - b[target_key]) / torch.linalg.vector_norm(b[target_key]).clamp_min(1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)
    bie_rng = np.random.default_rng(7)
    bie_cfg = BIEConfig(n_b=32, n_q=24, with_bem=False)
    bie_batch = stack([make_sample(bie_rng, bie_cfg) for _ in range(2)])
    tm_rng = np.random.default_rng(11)
    tm_batch = stack([make_tm_sample(tm_rng, N=2, nbo=32, nbi=16, con_lo=0.5, con_hi=2.0)
                      for _ in range(2)])
    out = {}
    out["nbio_laplace"] = one(bie_batch,
                              {"model": {"name": "nbio", "kwargs": {"hidden": 16, "n_layers": 1, "split": True}}},
                              "uq", device)
    out["nbio_tm_structure"] = one(tm_batch,
                                   {"model": {"name": "nbio_tm", "kwargs": {"split": False}}},
                                   "uq", device)
    out["btransformer"] = one(bie_batch,
                              {"model": {"name": "btransformer", "kwargs": {"width": 32, "n_layers": 1, "heads": 4}}},
                              "uq", device)
    print({"device": str(device), "smoke_rel_l2": out})
    print("[smoke_nbio_paths] OK")


if __name__ == "__main__":
    main()
