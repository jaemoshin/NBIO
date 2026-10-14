#!/usr/bin/env python
"""Runtime/scaling benchmark for paper tables.

Measures neural forward time per sample on GPU for representative checkpoints and CPU BEM solve+evaluate
time on small subsets. Writes results/runtime_summary.{json,md}.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import BIEDataset, to_device
from src.data.tm_loader import TMDataset
from src.data.bie import bem_solve
from src.data.transmission import transmission_solve
from src.models import build_model


def _load_model(run, device):
    ckpt = torch.load(os.path.join("results", run, "best.pt"), map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"]


@torch.no_grad()
def neural_ms(model, ds, device, batch_size=16, warmup=3, iters=10):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    b = to_device(next(iter(loader)), device)
    for _ in range(warmup):
        model(b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / (iters * next(iter(b.values())).shape[0])


def bem_laplace_ms(shard, n=8):
    ds = BIEDataset(shard)
    t0 = time.perf_counter()
    for i in range(min(n, len(ds))):
        b = ds[i]
        bd = {"X": b["Xb"].numpy(), "normal": b["Nb"].numpy(), "w": b["wb"].numpy(),
              "curv": b["curv"].numpy(), "speed": b["speed"].numpy()}
        _, ev = bem_solve(bd, b["g"].numpy(), "interior")
        ev(b["Xq"].numpy())
    return 1000.0 * (time.perf_counter() - t0) / min(n, len(ds))


def bem_tm_ms(shard, n=4):
    ds = TMDataset(shard)
    t0 = time.perf_counter()
    for i in range(min(n, len(ds))):
        b = ds[i]
        curves = [{"X": b["Xb_out"].numpy(), "normal": b["Nb_out"].numpy(), "w": b["wb_out"].numpy(),
                   "curv": b["curv_out"].numpy(), "speed": (b["wb_out"].numpy() / (2 * np.pi / b["wb_out"].shape[0]))}]
        Xi, Ni, wi, ci = b["Xb_in"].numpy(), b["Nb_in"].numpy(), b["wb_in"].numpy(), b["curv_in"].numpy()
        for k in range(Xi.shape[0]):
            curves.append({"X": Xi[k], "normal": Ni[k], "w": wi[k], "curv": ci[k],
                           "speed": wi[k] / (2 * np.pi / wi[k].shape[0])})
        _, ev = transmission_solve(curves, [1.0] + b["a_in"].numpy().tolist(), b["g"].numpy())
        ev(b["Xq"].numpy())
    return 1000.0 * (time.perf_counter() - t0) / min(n, len(ds))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    specs = [
        ("Laplace NBIO", "nbio_s0", BIEDataset, "data_cache/interior_test.pt"),
        ("Laplace DeepONet", "deeponet_s0", BIEDataset, "data_cache/interior_test.pt"),
        ("Laplace DeepSet", "deepset_s0", BIEDataset, "data_cache/interior_test.pt"),
        ("Transmission NBIO", "tm_nbio_s0", TMDataset, "data_cache/tm_test.pt"),
        ("Transmission DeepSet", "tm_deepset_s0", TMDataset, "data_cache/tm_test.pt"),
    ]
    for name, run, cls, shard in specs:
        if not os.path.exists(os.path.join("results", run, "best.pt")):
            continue
        model, _ = _load_model(run, device)
        rows.append({"method": name, "run": run, "device": str(device),
                     "ms_per_sample": neural_ms(model, cls(shard), device),
                     "params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    if os.path.exists("results/tm_transformer_s0/best.pt"):
        model, _ = _load_model("tm_transformer_s0", device)
        rows.append({"method": "Transmission Transformer", "run": "tm_transformer_s0", "device": str(device),
                     "ms_per_sample": neural_ms(model, TMDataset("data_cache/tm_test.pt"), device),
                     "params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    rows.append({"method": "Laplace BEM solve+eval", "run": "classical", "device": "cpu",
                 "ms_per_sample": bem_laplace_ms("data_cache/interior_test.pt"), "params": 0})
    rows.append({"method": "Transmission BEM solve+eval", "run": "classical", "device": "cpu",
                 "ms_per_sample": bem_tm_ms("data_cache/tm_test.pt"), "params": 0})
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/runtime_summary.json", "w"), indent=2)
    lines = ["# Runtime and scaling summary\n",
             "Forward/inference wall-clock. Neural models are batched on the available GPU; classical BEM rows",
             "include per-sample dense solve plus query evaluation on CPU, so they are reference timings rather",
             "than a hardware-normalized solver benchmark.\n",
             "| method | params | device | ms / sample |",
             "|---|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['params']} | {r['device']} | {r['ms_per_sample']:.3f} |")
    open("results/runtime_summary.md", "w").write("\n".join(lines) + "\n")
    print(open("results/runtime_summary.md").read())


if __name__ == "__main__":
    main()
