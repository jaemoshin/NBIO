#!/usr/bin/env python
"""Train an NBIO-transmission / baseline on the piecewise-constant conductivity dataset; zero-shot eval
over #inclusions, contrast, and boundary resolution."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.utils.common import set_seed, get_device, configure_torch, load_yaml, deep_update, save_json
from src.data.tm_loader import TMDataset
from src.data.loader import to_device
from src.models import build_model
from src.training.trainer import train, rel_l2

EVAL_SHARDS = ["tm_test", "tm_test_N1", "tm_test_N3", "tm_test_N4", "tm_test_hicon",
               "tm_testres96", "tm_testres192", "tm_testres384",
               "tm_test_close", "tm_test_extreme"]


@torch.no_grad()
def shard_rel(model, path, device, bs=32):
    if not os.path.exists(path):
        return float("nan")
    ds = TMDataset(path); tot = 0.0; n = 0
    for b in DataLoader(ds, batch_size=bs):
        b = to_device(b, device); k = b["uq"].shape[0]
        tot += rel_l2(model(b), b["uq"]).item() * k; n += k
    return tot / max(n, 1)


def parse_overrides(pairs):
    import yaml
    out = {}
    for p in pairs or []:
        key, _, v = p.partition("="); node = out
        ks = key.split(".")
        for kk in ks[:-1]:
            node = node.setdefault(kk, {})
        node[ks[-1]] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_yaml("configs/base.yaml"); cfg = deep_update(cfg, parse_overrides(args.set))
    seed = int(cfg.get("seed", 0)); set_seed(seed); configure_torch()
    device = get_device(cfg.get("device", "cuda") == "cuda")
    tr = TMDataset(os.path.join(args.data_dir, "tm_train.pt"))
    va = TMDataset(os.path.join(args.data_dir, "tm_val.pt"))
    model = build_model(cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_tm] model={cfg['model']['name']} seed={seed} trainable_params={n_train}")
    if n_train == 0:                         # nosplit structure = classical solver, nothing to train
        model.to(device).eval(); hist = {"n_params": 0, "best_val_rel_l2": float("nan")}
    else:
        hist = train(model, tr, va, cfg, device, args.out_dir)
        print(f"[train_tm] done best_val={hist['best_val_rel_l2']:.4e}")
    ev = {s: shard_rel(model, os.path.join(args.data_dir, f"{s}.pt"), device) for s in EVAL_SHARDS}
    save_json({"model": cfg["model"]["name"], "seed": seed, "n_params": hist["n_params"],
               "best_val_rel_l2": hist["best_val_rel_l2"], "eval": ev},
              os.path.join(args.out_dir, "eval.json"))
    for s, v in ev.items():
        print(f"    {s:16s}: rel_l2 {v:.4e}")


if __name__ == "__main__":
    main()
