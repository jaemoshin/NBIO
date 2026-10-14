#!/usr/bin/env python
"""Refresh a saved transmission run's eval.json over the current EVAL_SHARDS.

Useful after adding new stress shards: it reuses results/<run>/best.pt and writes an updated eval.json.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.common import get_device, configure_torch, save_json
from src.models import build_model
from scripts.train_tm import shard_rel, EVAL_SHARDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", default="data_cache")
    args = ap.parse_args()
    configure_torch()
    device = get_device(True)
    ckpt_path = os.path.join("results", args.run, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ev = {s: shard_rel(model, os.path.join(args.data_dir, f"{s}.pt"), device) for s in EVAL_SHARDS}
    hist = {}
    hpath = os.path.join("results", args.run, "history.json")
    if os.path.exists(hpath):
        import json
        hist = json.load(open(hpath))
    save_json({"model": cfg["model"]["name"], "seed": cfg.get("seed", "?"),
               "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "best_val_rel_l2": hist.get("best_val_rel_l2", float("nan")), "eval": ev},
              os.path.join("results", args.run, "eval.json"))
    for s, v in ev.items():
        print(f"    {s:16s}: rel_l2 {v:.4e}")
    print(f"[eval_tm_run] refreshed results/{args.run}/eval.json")


if __name__ == "__main__":
    main()
