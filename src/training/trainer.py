"""Trainer: relative-L2 (scale-invariant) regression of u at query points."""
from __future__ import annotations

import csv
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from ..data.loader import to_device
from ..utils.common import AverageMeter, count_params, save_json


def rel_l2(pred, target):
    num = torch.linalg.vector_norm(pred - target, dim=-1)
    den = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-9)
    return (num / den).mean()


@torch.no_grad()
def evaluate_rel_l2(model, loader, device):
    model.eval(); m = AverageMeter()
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch)
        m.update(rel_l2(pred, batch["uq"]).item(), batch["uq"].shape[0])
    return m.avg


def train(model, train_set, val_set, cfg, device, out_dir) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    tc = cfg["train"]
    model = model.to(device)
    tl = DataLoader(train_set, batch_size=int(tc["batch_size"]), shuffle=True, drop_last=True, num_workers=0)
    vl = DataLoader(val_set, batch_size=int(tc["batch_size"]), shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc.get("weight_decay", 0.0)))
    epochs = int(tc["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    clip = float(tc.get("grad_clip", 1.0))
    hist = {"n_params": count_params(model), "epochs": []}
    csv_path = os.path.join(out_dir, "training_curve.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_rel_l2", "val_rel_l2", "lr"])
    best = 1e9
    for ep in range(epochs):
        model.train(); lm = AverageMeter()
        for batch in tl:
            batch = to_device(batch, device)
            opt.zero_grad()
            loss = rel_l2(model(batch), batch["uq"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); lm.update(loss.item(), batch["uq"].shape[0])
        sched.step()
        vr = evaluate_rel_l2(model, vl, device)
        hist["epochs"].append({"epoch": ep, "train_rel_l2": lm.avg, "val_rel_l2": vr})
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, lm.avg, vr, sched.get_last_lr()[0]])
        if vr < best:
            best = vr
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
    hist["best_val_rel_l2"] = best
    save_json(hist, os.path.join(out_dir, "history.json"))
    checkpoint = torch.load(
        os.path.join(out_dir, "best.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    return hist
