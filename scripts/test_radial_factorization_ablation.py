#!/usr/bin/env python
"""CPU smoke checks for all models in the matched factorization ablation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_radial_factorization_ablation import METHODS, model_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import count_params  # noqa: E402


def batch():
    n_boundary = 24
    theta = torch.arange(n_boundary) * (2.0 * torch.pi / n_boundary)
    boundary = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).repeat(2, 1, 1)
    return {
        "Xb": boundary,
        "Nb": boundary.clone(),
        "wb": torch.full((2, n_boundary), 2.0 * torch.pi / n_boundary),
        "curv": torch.ones(2, n_boundary),
        "g": torch.cos(theta).repeat(2, 1),
        "Xq": 0.5 * boundary[:, :12],
        "uq": 0.5 * torch.cos(theta[:12]).repeat(2, 1),
        "kappa": torch.tensor([1.2, 3.8]),
    }


def main() -> None:
    sample = batch()
    expected = {"generic_matched": 2221, "radial_separate": 2241, "radial_normal": 2241}
    for method in METHODS:
        model = build_model(model_config(method))
        assert count_params(model) == expected[method]
        output = model(sample)
        assert output.shape == sample["uq"].shape
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
        if method == "radial_normal":
            model.zero_grad(set_to_none=True)
            model.qbx = True
            qbx_output = model(sample)
            assert qbx_output.shape == sample["uq"].shape
            assert torch.isfinite(qbx_output).all()
            qbx_output.square().mean().backward()
            assert all(parameter.grad is not None for parameter in model.parameters())
    print("radial factorization ablation CPU smoke passed")


if __name__ == "__main__":
    main()
