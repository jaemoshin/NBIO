#!/usr/bin/env python
"""CPU smoke and invariant checks for the prospectively fixed models."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402


def circle_batch():
    n_boundary, n_query = 24, 12
    theta = torch.arange(n_boundary) * (2 * torch.pi / n_boundary)
    boundary = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).repeat(2, 1, 1)
    return {
        "Xb": boundary,
        "Nb": boundary.clone(),
        "wb": torch.full((2, n_boundary), 2 * torch.pi / n_boundary),
        "curv": torch.ones(2, n_boundary),
        "g": torch.cos(theta).repeat(2, 1),
        "Xq": 0.5 * boundary[:, :n_query],
        "uq": 0.5 * torch.cos(theta[:n_query]).repeat(2, 1),
        "kappa": torch.tensor([0.15, 6.0]),
    }


def main():
    batch = circle_batch()
    specifications = (
        ("mixture", {"variant": "mixture", "n_atoms": 8}, 16),
        ("radial_mlp", {"variant": "radial_mlp", "hidden": 32, "n_layers": 3}, 2241),
        ("radial_separate", {"variant": "radial_separate", "hidden": 32, "n_layers": 3}, 2241),
    )
    for name, kwargs, expected_parameters in specifications:
        model = build_model({"model": {"name": "nbio_screening", "kwargs": kwargs}})
        output = model(batch)
        assert output.shape == batch["uq"].shape and torch.isfinite(output).all(), name
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters()), name
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
        matrix = model.assemble(batch)
        assert torch.isfinite(matrix).all()
        if name == "radial_separate":
            scaled = torch.linspace(0, 20, 4096).reshape(1, 1, -1)
            radius = scaled / 2.0
            law = model.m_law(scaled, kappa=torch.tensor([2.0]), radius=radius).reshape(-1)
        else:
            law = model.m_law(torch.linspace(0, 20, 4096))
        torch.testing.assert_close(law[0], torch.tensor(1.0))
        if name == "mixture":
            assert (law > 0).all() and (law <= 1).all()
            assert (law[1:] <= law[:-1] + 1e-7).all()
    print("confirmatory screening models: CPU smoke passed")


if __name__ == "__main__":
    main()
