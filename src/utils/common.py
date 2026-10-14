"""Shared utilities (seed, device, torch config, io, meters)."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def configure_torch():
    # Disable cuDNN for Pascal-node portability (cuDNN "no engine" errors); deterministic-ish.
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True


def get_device(want_cuda=True):
    return torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_yaml(obj, path):
    import yaml
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


class AverageMeter:
    def __init__(self): self.sum = 0.0; self.n = 0
    def update(self, v, k=1): self.sum += float(v) * k; self.n += k
    @property
    def avg(self): return self.sum / max(self.n, 1)


class Timer:
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *a): self.dt = time.perf_counter() - self.t0
