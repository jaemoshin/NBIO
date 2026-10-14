#!/usr/bin/env python
"""Train a Stage-2 model (fno black-box / green_fno = const-coeff prior + learned correction) on the
variable-coefficient dataset; zero-shot eval over contrast and resolution. Reports the const-coeff prior
('wrong fixed kernel') error for reference."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.utils.common import set_seed, get_device, configure_torch, load_yaml, deep_update, save_json, AverageMeter
from src.data.tm_loader import TMDataset
from src.data.loader import to_device
from src.models import build_model

SHARDS = ["vc_test", "vc_test_hicon", "vc_testres64", "vc_testres128"]


def rel_l2_grid(pred, target):
    p = pred.reshape(pred.shape[0], -1); t = target.reshape(target.shape[0], -1)
    return (torch.linalg.vector_norm(p - t, dim=1) / torch.linalg.vector_norm(t, dim=1).clamp_min(1e-9)).mean()


@torch.no_grad()
def shard_eval(model, path, device):
    if not os.path.exists(path):
        return {"rel_l2": float("nan"), "prior_rel_l2": float("nan")}
    ds = TMDataset(path); m = AverageMeter(); mp = AverageMeter()
    for b in DataLoader(ds, batch_size=16):
        b = to_device(b, device); k = b["u"].shape[0]
        m.update(rel_l2_grid(model(b), b["u"]).item(), k)
        mp.update(rel_l2_grid(b["ulap"], b["u"]).item(), k)   # const-coeff prior (wrong fixed kernel)
    return {"rel_l2": m.avg, "prior_rel_l2": mp.avg}


def parse_overrides(pairs):
    import yaml
    out = {}
    for p in pairs or []:
        key, _, v = p.partition("="); node = out
        for kk in key.split(".")[:-1]:
            node = node.setdefault(kk, {})
        node[key.split(".")[-1]] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_yaml("configs/base.yaml"); cfg = deep_update(cfg, parse_overrides(args.set))
    seed = int(cfg.get("seed", 0)); set_seed(seed); configure_torch()
    device = get_device(True)
    tr = TMDataset(os.path.join(args.data_dir, "vc_train.pt")); va = TMDataset(os.path.join(args.data_dir, "vc_val.pt"))
    model = build_model(cfg).to(device)
    os.makedirs(args.out_dir, exist_ok=True)
    epochs = int(cfg["train"]["epochs"]); bs = int(cfg["train"]["batch_size"])
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=bs)
    print(f"[train_vc] model={cfg['model']['name']} params={sum(p.numel() for p in model.parameters())}")
    best = 1e9
    for ep in range(epochs):
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
            best = vm.avg; torch.save({"state_dict": model.state_dict(), "config": cfg}, os.path.join(args.out_dir, "best.pt"))
    print(f"[train_vc] done best_val={best:.4e}")
    ev = {s: shard_eval(model, os.path.join(args.data_dir, f"{s}.pt"), device) for s in SHARDS}
    save_json({"model": cfg["model"]["name"], "seed": seed, "best_val_rel_l2": best,
               "n_params": sum(p.numel() for p in model.parameters()), "eval": ev}, os.path.join(args.out_dir, "eval.json"))
    for s, v in ev.items():
        print(f"    {s:16s}: model {v['rel_l2']:.4e}  | const-prior {v['prior_rel_l2']:.4e}")


if __name__ == "__main__":
    main()
