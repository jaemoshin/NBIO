"""Evaluation: interior accuracy, zero-shot geometry (test set = unseen shapes), zero-shot boundary
resolution (same geometry @ n_b in {64,128,256}), exterior capability, and runtime. Reports the classical
BEM reference per shard as the accuracy ceiling."""
from __future__ import annotations

import os
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..data.loader import BIEDataset, to_device
from ..training.trainer import rel_l2
from ..utils.common import Timer


@torch.no_grad()
def shard_rel_l2(model, shard_path, device, batch_size=32) -> Dict[str, float]:
    if not os.path.exists(shard_path):
        return {"rel_l2": float("nan"), "bem_rel_l2": float("nan"), "n_b": -1}
    ds = BIEDataset(shard_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    errs, bems, n = [], [], 0
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch)
        bs = batch["uq"].shape[0]
        errs.append(rel_l2(pred, batch["uq"]).item() * bs); n += bs
        if "bem_uq" in batch:
            b = batch["bem_uq"]
            finite = torch.isfinite(b).all(dim=1)
            if finite.any():
                bems.append(rel_l2(b[finite], batch["uq"][finite]).item() * int(finite.sum()))
    nb = int(ds.d["Xb"].shape[1])
    return {"rel_l2": sum(errs) / max(n, 1),
            "bem_rel_l2": (sum(bems) / max(n, 1)) if bems else float("nan"),
            "n_b": nb, "n": n}


@torch.no_grad()
def measure_runtime(model, shard_path, device, n_iters=10):
    ds = BIEDataset(shard_path)
    batch = to_device(next(iter(DataLoader(ds, batch_size=16))), device)
    for _ in range(2):
        model(batch)
    with Timer() as t:
        for _ in range(n_iters):
            model(batch)
    return 1000.0 * t.dt / (n_iters * batch["uq"].shape[0])


_TASK_SHARDS = {
    "interior": [("interior_test", "interior_test.pt"), ("testres64", "interior_testres64.pt"),
                 ("testres128", "interior_testres128.pt"), ("testres256", "interior_testres256.pt")],
    "exterior": [("exterior_test", "exterior_test.pt")],
    "yukawa":   [("yukawa_test", "yukawa_test.pt")],
}


def full_eval(model, data_dir, device, task="interior") -> Dict[str, Dict]:
    out = {}
    shards = _TASK_SHARDS[task]
    for name, fn in shards:
        out[name] = shard_rel_l2(model, os.path.join(data_dir, fn), device)
    try:
        out["runtime_ms"] = measure_runtime(model, os.path.join(data_dir, shards[0][1]), device)
    except Exception:
        out["runtime_ms"] = float("nan")
    return out
