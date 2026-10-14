#!/usr/bin/env python
"""Train an NBIO / baseline operator on the interior Laplace-BVP dataset, then run the full eval."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import set_seed, get_device, configure_torch, load_yaml, deep_update, save_json
from src.data.loader import BIEDataset
from src.models import build_model
from src.training.trainer import train
from src.evaluation.evaluate import full_eval


def parse_overrides(pairs):
    import yaml
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("="); node = out
        ks = k.split(".")
        for kk in ks[:-1]:
            node = node.setdefault(kk, {})
        node[ks[-1]] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--task", default="interior", choices=["interior", "exterior", "yukawa"])
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_yaml(args.base_config)
    cfg = deep_update(cfg, load_yaml(args.config))
    cfg = deep_update(cfg, parse_overrides(args.set))
    seed = int(cfg.get("seed", 0)); set_seed(seed); configure_torch()
    device = get_device(cfg.get("device", "cuda") == "cuda")
    train_set = BIEDataset(os.path.join(args.data_dir, f"{args.task}_train.pt"))
    val_set = BIEDataset(os.path.join(args.data_dir, f"{args.task}_val.pt"))
    model = build_model(cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[train] task={args.task} model={cfg['model']['name']} seed={seed} device={device} "
          f"params={sum(p.numel() for p in model.parameters())}")
    hist = train(model, train_set, val_set, cfg, device, args.out_dir)
    print(f"[train] done. best_val_rel_l2={hist['best_val_rel_l2']:.4e} n_params={hist['n_params']}")
    ev = full_eval(model, args.data_dir, device, task=args.task)
    res = {"model": cfg["model"]["name"], "task": args.task, "seed": seed, "n_params": hist["n_params"],
           "best_val_rel_l2": hist["best_val_rel_l2"], "eval": ev}
    save_json(res, os.path.join(args.out_dir, "eval.json"))
    for k, v in ev.items():
        if isinstance(v, dict):
            print(f"    {k:16s}: rel_l2 {v['rel_l2']:.4e}  (BEM {v.get('bem_rel_l2', float('nan')):.2e}, n_b={v.get('n_b')})")
        else:
            print(f"    {k:16s}: {v:.4f}")


if __name__ == "__main__":
    main()
