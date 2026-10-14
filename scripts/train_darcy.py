#!/usr/bin/env python
"""Train a model (fno black-box / green_fno = const-coeff Dirichlet Green's prior + learned correction) on
the OFFICIAL released Darcy-flow benchmark shards (data_cache/darcy_*.pt, built by prep_darcy_benchmark.py).
Trains at res 16 (canonical FNO-small), then zero-shot super-resolution eval at res 32/64/128.

Self-contained (does not depend on configs/base.yaml's nbio default). rel-L2 on the grid; reports the
const-coeff prior ('wrong fixed kernel') error for reference. Identical data/splits for both models."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.utils.common import set_seed, get_device, configure_torch, save_json, AverageMeter
from src.data.tm_loader import TMDataset
from src.data.loader import to_device
from src.models import build_model

SHARD_SETS = {
    "darcy": ["darcy_test", "darcy_testres32", "darcy_testres64", "darcy_testres128"],
    "darcy32": ["darcy32_test", "darcy32_testres64"],
}


def rel_l2_grid(pred, target):
    p = pred.reshape(pred.shape[0], -1); t = target.reshape(target.shape[0], -1)
    return (torch.linalg.vector_norm(p - t, dim=1) / torch.linalg.vector_norm(t, dim=1).clamp_min(1e-9)).mean()


@torch.no_grad()
def shard_eval(model, path, device):
    if not os.path.exists(path):
        return {"rel_l2": float("nan"), "prior_rel_l2": float("nan")}
    ds = TMDataset(path); m = AverageMeter(); mp = AverageMeter()
    for b in DataLoader(ds, batch_size=8):
        b = to_device(b, device); k = b["u"].shape[0]
        m.update(rel_l2_grid(model(b), b["u"]).item(), k)
        mp.update(rel_l2_grid(b["ulap"], b["u"]).item(), k)   # const-coeff Dirichlet prior (wrong fixed kernel)
    return {"rel_l2": m.avg, "prior_rel_l2": mp.avg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--prefix", default="darcy", choices=list(SHARD_SETS.keys()),
                    help="shard prefix: 'darcy' (res16 train) or 'darcy32' (res32 train)")
    ap.add_argument("--model", required=True, choices=["fno", "green_fno"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed); configure_torch()
    device = get_device(True)
    use_prior = args.model == "green_fno"
    kwargs = {"in_ch": 3 if use_prior else 4, "width": args.width, "modes": args.modes,
              "n_layers": args.n_layers, "use_prior": use_prior}
    cfg = {"model": {"name": args.model, "kwargs": kwargs}}
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_darcy] model={args.model} seed={args.seed} params={n_params} device={device}")

    tr = TMDataset(os.path.join(args.data_dir, f"{args.prefix}_train.pt"))
    va = TMDataset(os.path.join(args.data_dir, f"{args.prefix}_val.pt"))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            b = to_device(b, device); opt.zero_grad()
            loss = rel_l2_grid(model(b), b["u"]); loss.backward(); opt.step()
        sched.step()
        model.eval(); vm = AverageMeter()
        with torch.no_grad():
            for b in vl:
                b = to_device(b, device); vm.update(rel_l2_grid(model(b), b["u"]).item(), b["u"].shape[0])
        if vm.avg < best:
            best = vm.avg
            torch.save({"state_dict": model.state_dict(), "config": cfg}, os.path.join(args.out_dir, "best.pt"))
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  val_rel_l2 {vm.avg:.4e}  best {best:.4e}")

    # reload best for eval
    ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    ev = {s: shard_eval(model, os.path.join(args.data_dir, f"{s}.pt"), device) for s in SHARD_SETS[args.prefix]}
    save_json({"model": args.model, "seed": args.seed, "best_val_rel_l2": best, "n_params": n_params,
               "epochs": args.epochs, "prefix": args.prefix, "eval": ev}, os.path.join(args.out_dir, "eval.json"))
    print(f"[train_darcy] done best_val={best:.4e}")
    for s, v in ev.items():
        print(f"    {s:18s}: model {v['rel_l2']:.4e}  | const-prior {v['prior_rel_l2']:.4e}")


if __name__ == "__main__":
    main()
