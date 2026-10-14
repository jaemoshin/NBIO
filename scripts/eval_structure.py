#!/usr/bin/env python
"""Evaluate the NBIO-tm STRUCTURE (split=False, 0 trainable params = the classical multi-region solver in
torch) across all zero-shot shards. This is the headline Stage-1 result: structure alone -> solver grade at
any #inclusions / contrast / resolution."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.common import get_device, configure_torch
from src.models import build_model
from scripts.train_tm import shard_rel, EVAL_SHARDS


def main():
    configure_torch(); device = get_device(True)
    model = build_model({"model": {"name": "nbio_tm", "kwargs": {"split": False}}}).to(device).eval()
    ev = {s: shard_rel(model, os.path.join("data_cache", f"{s}.pt"), device, bs=16) for s in EVAL_SHARDS}
    os.makedirs("results/tm_structure", exist_ok=True)
    json.dump({"model": "nbio_tm_structure", "n_params": 0, "eval": ev},
              open("results/tm_structure/eval.json", "w"), indent=2)
    for s, v in ev.items():
        print(f"    {s:16s}: rel_l2 {v:.4e}")
    print("[eval_structure] DONE")


if __name__ == "__main__":
    main()
