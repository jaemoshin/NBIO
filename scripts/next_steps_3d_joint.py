#!/usr/bin/env python
"""Interior+exterior joint training for the minimal 3D surface NBIO.

The existing 3D run trained on interior Dirichlet data and then evaluated exterior zero-shot, where the
learned correction helped interior but degraded exterior.  This script trains the same small NBIO3D model
on mixed interior/exterior batches to test whether the 3D operator can learn a shared correction without
sacrificing either side.

Outputs only ``results/next_steps_3d_joint.json`` and an optional checkpoint under
``results/next_steps_3d_joint/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.nbio3d import NBIO3D, DeepSet3D, batch, rel
from src.utils.common import configure_torch, set_seed

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = ROOT / "results"


def cat_batches(items):
    keys = items[0].keys()
    return {k: torch.cat([b[k] for b in items], dim=0) for k in keys}


@torch.no_grad()
def eval_split(model, device, n, amp, exterior, seed0=1000, batches=4, B=4):
    model.eval()
    vals = []
    for i in range(batches):
        b = batch(np.random.default_rng(seed0 + i), B, n, amp=amp, exterior=exterior, device=device)
        vals.append(float(rel(model(b), b["uq"]).detach().cpu()))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["nbio3d", "nbio3d_structure", "deepset3d"], default="nbio3d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-half", type=int, default=4, help="interior samples and exterior samples per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/next_steps_3d_joint")
    args = ap.parse_args()

    configure_torch()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    if args.model == "nbio3d":
        model = NBIO3D(split=True).to(device)
    elif args.model == "nbio3d_structure":
        model = NBIO3D(split=False).to(device)
    else:
        model = DeepSet3D().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr) if n_params else None
    history = []
    print(f"[3d-joint] model={args.model} params={n_params} device={device} n={args.n} epochs={args.epochs}")

    for ep in range(args.epochs if n_params else 0):
        model.train()
        bi = batch(rng, args.batch_half, args.n, amp=1.0, exterior=False, device=device)
        be = batch(rng, args.batch_half, args.n, amp=1.0, exterior=True, device=device)
        b = cat_batches([bi, be])
        pred = model(b)
        loss = rel(pred, b["uq"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
        if ep % max(1, args.epochs // 5) == 0:
            print(f"[3d-joint] ep={ep:04d} mixed rel={history[-1]:.3e}")

    evals = {}
    for tag, kw in {
        "interior_indist": dict(n=args.n, amp=1.0, exterior=False),
        "exterior_indist": dict(n=args.n, amp=1.0, exterior=True),
        "interior_strong_deform": dict(n=args.n, amp=1.8, exterior=False),
        "exterior_strong_deform": dict(n=args.n, amp=1.8, exterior=True),
        "interior_res720": dict(n=720, amp=1.0, exterior=False),
        "exterior_res720": dict(n=720, amp=1.0, exterior=True),
    }.items():
        mean, std = eval_split(model, device=device, **kw)
        evals[tag] = {"mean_rel_l2": mean, "std_rel_l2": std}
        print(f"[3d-joint] {tag:24s} {mean:.3e} +/- {std:.1e}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "history": history},
               out_dir / "best.pt")
    out = {
        "task": "3d_surface_interior_exterior_joint_training",
        "model": args.model,
        "seed": args.seed,
        "n_params": n_params,
        "device": str(device),
        "n_surface_train": args.n,
        "epochs": args.epochs,
        "history": history,
        "eval": evals,
        "honest_status": (
            "Tests whether the learned 3D correction can cover both interior and exterior data. Compare to "
            "existing interior-only results in results/nbio3d/eval.json and structure-only results in "
            "results/nbio3d_structure/eval.json."
        ),
    }
    json_name = f"{out_dir.name}.json"
    with open(RESULTS / json_name, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] results/{json_name}")


if __name__ == "__main__":
    main()
