#!/usr/bin/env python
"""CPU smoke checks for the kappa-conditioned model and analytic prior schedule."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402


def circle_batch(batch_size=2, n_b=16, n_q=12):
    theta = torch.arange(n_b) * (2 * torch.pi / n_b)
    boundary = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    Xb = boundary.unsqueeze(0).repeat(batch_size, 1, 1)
    batch = {
        "Xb": Xb,
        "Nb": Xb.clone(),
        "wb": torch.full((batch_size, n_b), 2 * torch.pi / n_b),
        "curv": torch.ones(batch_size, n_b),
        "g": torch.cos(theta).unsqueeze(0).repeat(batch_size, 1),
        "Xq": 0.5 * Xb[:, :n_q],
        "uq": 0.5 * torch.cos(theta[:n_q]).unsqueeze(0).repeat(batch_size, 1),
        "kappa": torch.tensor([0.0, 4.0]),
    }
    return batch


def main():
    torch.manual_seed(4)
    config = {"model": {"name": "nbio_kappa", "kwargs": {
        "hidden": 16, "n_layers": 2, "prior_mode": "local",
        "local_tau": 0.32, "local_power": 4.0,
    }}}
    model = build_model(config)
    batch = circle_batch()
    output = model(batch)
    assert output.shape == batch["uq"].shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    weight = model.bio._prior_weight(batch["Xb"], batch["Xb"], batch["kappa"])
    torch.testing.assert_close(weight[0], torch.ones_like(weight[0]))
    scale = model.bio._correction_scale(batch["kappa"], weight)
    assert scale[0].eq(0).all() and scale[1].eq(1).all()
    off_diagonal = ~torch.eye(weight.shape[-1], dtype=torch.bool)
    assert weight[1][off_diagonal].mean() < 0.2
    assert torch.diagonal(weight[1]).eq(1).all()

    potential_config = {"model": {"name": "nbio_kappa", "kwargs": {
        "hidden": 8, "n_layers": 2, "prior_mode": "local",
        "local_scope": "potential",
    }}}
    potential_model = build_model(potential_config)
    assembly_weight = potential_model.bio._prior_weight(
        batch["Xb"], batch["Xb"], batch["kappa"], interaction="assembly"
    )
    assert assembly_weight[0].eq(1).all()
    assert assembly_weight[1].max() < 1e-6

    full_config = {"model": {"name": "nbio_kappa", "kwargs": {
        "hidden": 8, "n_layers": 2, "prior_mode": "full",
    }}}
    full_model = build_model(full_config)
    kappa_zero = {key: value[:1] for key, value in batch.items()}
    torch.testing.assert_close(model(kappa_zero), full_model(kappa_zero))
    torch.testing.assert_close(potential_model(kappa_zero), full_model(kappa_zero))

    adaptive_config = {"model": {"name": "nbio_kappa", "kwargs": {
        "hidden": 8, "n_layers": 2, "prior_mode": "adaptive",
    }}}
    adaptive_model = build_model(adaptive_config)
    adaptive_weight = adaptive_model.bio._prior_weight(
        batch["Xb"], batch["Xb"], batch["kappa"]
    )
    assert adaptive_weight[0].eq(1).all()
    assert (adaptive_weight[1] >= weight[1]).all()
    torch.testing.assert_close(adaptive_model(kappa_zero), full_model(kappa_zero))
    adaptive_model(batch).square().mean().backward()
    assert torch.isfinite(adaptive_model.bio.log_gate_rate.grad)

    saturating_config = {"model": {"name": "nbio_kappa", "kwargs": {
        "hidden": 8, "n_layers": 2, "prior_mode": "adaptive",
        "residual_scale": "saturating",
    }}}
    saturating_model = build_model(saturating_config)
    saturating_scale = saturating_model.bio._correction_scale(batch["kappa"], weight)
    assert saturating_scale[0].eq(0).all()
    assert 0 < saturating_scale[1].item() < 1
    torch.testing.assert_close(saturating_model(kappa_zero), full_model(kappa_zero))
    print("kappa-conditioned CPU smoke passed")


if __name__ == "__main__":
    main()
