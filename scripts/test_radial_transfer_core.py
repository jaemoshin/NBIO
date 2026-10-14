#!/usr/bin/env python3
"""CPU smoke tests for radial transfer core model paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.utils.common import count_params


def batch(dtype=torch.float32):
    theta = torch.linspace(0, 2 * torch.pi, 17, dtype=dtype)[:-1]
    boundary = torch.stack((theta.cos(), theta.sin()), dim=-1).unsqueeze(0)
    normals = boundary.clone()
    queries = 0.5 * boundary[:, :8]
    return {
        "Xb": boundary,
        "Nb": normals,
        "wb": torch.full((1, 16), 2 * torch.pi / 16, dtype=dtype),
        "curv": torch.ones(1, 16, dtype=dtype),
        "g": boundary[..., 0],
        "Xq": queries,
        "uq": queries[..., 0],
        "kappa": torch.tensor([2.0], dtype=dtype),
    }


def test_asymptotic(device: torch.device) -> None:
    model = build_model({"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_asymptotic",
        "hidden": 32,
        "n_layers": 3,
        "region": "interior",
        "solve": "direct",
    }}}).double().to(device)
    assert count_params(model) == 2274
    sample = {
        key: value.to(device) for key, value in batch(torch.float64).items()
    }
    prediction = model(sample)
    assert prediction.shape == sample["uq"].shape
    prediction.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    endpoint = torch.zeros(
        1, dtype=torch.float64, device=device, requires_grad=True,
    )
    value = model.m_law(endpoint)
    derivative = torch.autograd.grad(value.sum(), endpoint)[0]
    torch.testing.assert_close(value, torch.ones_like(value), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        derivative, torch.zeros_like(derivative), atol=1e-12, rtol=0,
    )


def test_transformer(device: torch.device) -> None:
    for normalized in (False, True):
        model = build_model({"model": {"name": "kappa_btransformer", "kwargs": {
            "width": 32,
            "n_layers": 2,
            "heads": 4,
            "token_layers": 2,
            "kappa_scale": 4.0,
            "normalize_load": normalized,
            "include_weights": normalized,
        }}}).to(device)
        sample = {key: value.to(device) for key, value in batch().items()}
        prediction = model(sample)
        assert prediction.shape == sample["uq"].shape
        (prediction - sample["uq"]).square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if normalized:
            scaled = dict(sample)
            scaled["g"] = 3.0 * scaled["g"]
            torch.testing.assert_close(
                model(scaled), 3.0 * model(sample), atol=2e-5, rtol=2e-5,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    arguments = parser.parse_args()
    if arguments.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but CUDA is unavailable")
    selected_device = torch.device("cuda" if arguments.cuda else "cpu")
    test_asymptotic(selected_device)
    test_transformer(selected_device)
    print(f"radial transfer core {selected_device.type} smoke: PASS")
