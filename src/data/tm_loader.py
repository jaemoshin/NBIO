"""Loader for transmission shards (dict of stacked tensors; serves per-sample dicts of all keys)."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TMDataset(Dataset):
    def __init__(self, shard_path: str):
        self.d = torch.load(shard_path, map_location="cpu", weights_only=False)
        self.keys = list(self.d.keys())
        self.n = next(iter(self.d.values())).shape[0]   # generic (works for tm 'uq' and vc 'u' shards)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.d[k][i] for k in self.keys}
