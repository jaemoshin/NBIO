#!/usr/bin/env python
"""Train a black-box interface baseline on the parametric interface-with-jumps benchmark (same data NBIO-
interface structure was evaluated on) -> head-to-head."""
from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from src.utils.common import set_seed, get_device, configure_torch, save_json, AverageMeter
from src.data.tm_loader import TMDataset
from src.data.loader import to_device
from src.models import build_model


def rel_grid(p, t):
    return (torch.linalg.vector_norm(p - t, dim=1) / torch.linalg.vector_norm(t, dim=1).clamp_min(1e-12)).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--out", required=True); a = ap.parse_args()
    set_seed(a.seed); configure_torch(); dev = get_device(True)
    kw = {"iface_transformer": {"d": 128, "heads": 4, "layers": 3},
          "iface_deepset": {"hidden": 256, "p": 128}}[a.model]
    model = build_model({"model": {"name": a.model, "kwargs": kw}}).to(dev)
    tr = TMDataset("data_cache/iface_train.pt"); va = TMDataset("data_cache/iface_val.pt"); te = TMDataset("data_cache/iface_test.pt")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    tl = DataLoader(tr, batch_size=16, shuffle=True, drop_last=True)
    print(f"[iface] {a.model} params={sum(p.numel() for p in model.parameters())}")
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        for b in tl:
            b = to_device(b, dev); loss = rel_grid(model(b), b["uq"]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); vm = AverageMeter()
        with torch.no_grad():
            for b in DataLoader(va, batch_size=16):
                b = to_device(b, dev); vm.update(rel_grid(model(b), b["uq"]).item(), b["uq"].shape[0])
        best = min(best, vm.avg)
        if ep % 30 == 0: print(f"  ep{ep} val {vm.avg:.3e}")
    model.eval(); tm = AverageMeter()
    with torch.no_grad():
        for b in DataLoader(te, batch_size=16):
            b = to_device(b, dev); tm.update(rel_grid(model(b), b["uq"]).item(), b["uq"].shape[0])
    print(f"[iface] {a.model} TEST rel-L2 {tm.avg:.3e} (best val {best:.3e})")
    os.makedirs(a.out, exist_ok=True)
    save_json({"model": a.model, "n_params": sum(p.numel() for p in model.parameters()),
               "eval": {"iface_test": tm.avg, "best_val": best}}, os.path.join(a.out, "eval.json"))


if __name__ == "__main__":
    main()
