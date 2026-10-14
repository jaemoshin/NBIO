#!/usr/bin/env python
"""Train a KNOWN-kernel Yukawa NBIO (src/models/nbio_yukawa.NBIOYukawa) at a FIXED kappa on the
yukawa_{train,val,test}.pt shards produced by scripts/gen_yukawa.py.

This is a NEW training entrypoint (the global build_model / scripts/train.py route the stock Laplace-kernel
NBIO and cannot reach the new class without editing the registry, which is intentionally avoided). The model
INITIALIZES at the classical Yukawa Nystrom solver (known Yukawa double layer, gate=1, zero-init correction)
and learns a small smooth correction -> solver grade, exactly as the Laplace NBIO does for Laplace.

Reuses the repo trainer (relative-L2 regression) and loader unchanged. Saves best.pt + history.json + eval.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.common import set_seed, get_device, configure_torch, save_json
from src.data.loader import BIEDataset
from src.models.nbio_yukawa import NBIOYukawa
from src.training.trainer import train, evaluate_rel_l2
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--out-dir", default="results/yukawa_nbio")
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    configure_torch()
    device = get_device(True)

    train_set = BIEDataset(os.path.join(args.data_dir, "yukawa_train.pt"))
    val_set = BIEDataset(os.path.join(args.data_dir, "yukawa_val.pt"))
    test_set = BIEDataset(os.path.join(args.data_dir, "yukawa_test.pt"))

    model = NBIOYukawa(kappa=args.kappa, hidden=args.hidden, n_layers=args.n_layers,
                       region="interior", split=True, solve="direct")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_yukawa_nbio] kappa={args.kappa} device={device} n_params={n_params}")

    cfg = {"train": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
                     "weight_decay": 0.0, "grad_clip": 1.0},
           "model": {"name": "nbio_yukawa", "kappa": args.kappa}}
    os.makedirs(args.out_dir, exist_ok=True)
    hist = train(model, train_set, val_set, cfg, device, args.out_dir)
    print(f"[train_yukawa_nbio] done best_val_rel_l2={hist['best_val_rel_l2']:.4e}")

    # reload best and evaluate vs exact GT and vs Yukawa-BEM on the test shard
    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tl = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_rel = evaluate_rel_l2(model, tl, device)

    # BEM reference and wrong-Laplace-BEM already stored in the shard
    d = test_set.d
    def shard_rel(key):
        v = d[key]
        fin = torch.isfinite(v).all(dim=1)
        return (torch.linalg.vector_norm(v[fin] - d["uq"][fin], dim=1)
                / (torch.linalg.vector_norm(d["uq"][fin], dim=1) + 1e-9)).mean().item()

    res = {"kappa": args.kappa, "n_params": n_params,
           "best_val_rel_l2": hist["best_val_rel_l2"],
           "test_nbio_rel_l2": test_rel,
           "test_yukawa_bem_rel_l2": shard_rel("bem_uq"),
           "test_wrong_laplace_bem_rel_l2": shard_rel("laplace_bem_uq")}
    save_json(res, os.path.join(args.out_dir, "eval.json"))
    print("[train_yukawa_nbio] TEST:")
    for k, v in res.items():
        print(f"    {k:32s}: {v}")
    print("[train_yukawa_nbio] DONE")


if __name__ == "__main__":
    main()
