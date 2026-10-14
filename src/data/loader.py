"""Load the BIE shards and serve per-sample dicts. rel-L2 is scale-invariant so no normalization needed."""
from __future__ import annotations

import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset

_KEYS = ["Xb", "Nb", "wb", "curv", "speed", "g", "Xq", "uq"]
_OPT = ["bem_uq", "mu", "kappa", "laplace_bem_uq", "query_margin"]


class BIEDataset(Dataset):
    def __init__(self, shard_path: str):
        self.d = torch.load(shard_path, map_location="cpu", weights_only=False)
        self.n = self.d["uq"].shape[0]
        self.keys = [k for k in (_KEYS + _OPT) if k in self.d]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.d[k][i] for k in self.keys}


def to_device(batch: Dict[str, torch.Tensor], device):
    return {k: v.to(device) for k, v in batch.items()}
