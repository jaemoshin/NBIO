#!/usr/bin/env python3
"""CPU forward/backward and endpoint smoke for direct quadratic radial NBIO."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.utils.common import count_params


def main() -> None:
    config = {"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_quadratic", "hidden": 32, "n_layers": 3,
    }}}
    model = build_model(config).double()
    assert count_params(model) == 2241
    theta = torch.arange(32, dtype=torch.float64) * (2 * torch.pi / 32)
    points = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).unsqueeze(0)
    batch = {
        "Xb": points, "Nb": points.clone(),
        "wb": torch.full((1, 32), 2 * torch.pi / 32, dtype=torch.float64),
        "curv": torch.ones(1, 32, dtype=torch.float64),
        "g": points[..., 0], "Xq": 0.5 * points[:, :8],
        "uq": 0.5 * points[:, :8, 0],
        "kappa": torch.tensor([2.0], dtype=torch.float64),
    }
    prediction = model(batch)
    (prediction - batch["uq"]).square().mean().backward()
    assert torch.isfinite(prediction).all()
    s0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    value = model.m_law(s0)
    derivative = torch.autograd.grad(value.sum(), s0)[0]
    assert torch.equal(value, torch.ones_like(value))
    assert derivative.abs().max() < 1e-12
    print("direct quadratic radial CPU smoke passed")


if __name__ == "__main__":
    main()
