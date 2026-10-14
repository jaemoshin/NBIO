#!/usr/bin/env python3
"""CPU smoke for endpoint regularity and both new training paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model


def batch() -> dict[str, torch.Tensor]:
    theta = torch.linspace(0, 2 * torch.pi, 33)[:-1]
    points = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).unsqueeze(0)
    normals = points.clone()
    weights = torch.full((1, 32), 2 * torch.pi / 32)
    return {
        "Xb": points,
        "Nb": normals,
        "wb": weights,
        "curv": torch.ones(1, 32),
        "g": points[..., 0],
        "Xq": 0.5 * points[:, :8],
        "uq": 0.5 * points[:, :8, 0],
        "kappa": torch.tensor([2.0]),
    }


def main() -> None:
    configs = (
        {"model": {"name": "nbio_screening", "kwargs": {
            "variant": "radial_endpoint", "hidden": 32, "n_layers": 3,
        }}},
        {"model": {"name": "nbio_kappa", "kwargs": {
            "hidden": 30, "n_layers": 3, "prior_mode": "none",
            "feature_mode": "smooth", "residual_order": 2.0,
        }}},
    )
    for config in configs:
        model = build_model(config)
        prediction = model(batch())
        loss = (prediction - batch()["uq"]).square().mean()
        loss.backward()
        assert torch.isfinite(prediction).all()
        assert any(parameter.grad is not None for parameter in model.parameters())
    endpoint = build_model(configs[0]).double()
    s0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(endpoint.m_law(s0).sum(), s0)[0]
    assert derivative.abs().max() < 1e-12
    print("endpoint and matched no-prior CPU smoke passed")


if __name__ == "__main__":
    main()
