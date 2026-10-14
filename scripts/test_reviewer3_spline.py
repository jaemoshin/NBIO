#!/usr/bin/env python
"""CPU smoke tests for the reviewer-3 spline representation control."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.nbio_screening_law import ScreeningLawNBIO  # noqa: E402
from src.utils.common import count_params  # noqa: E402


def main() -> None:
    model = ScreeningLawNBIO(
        variant="radial_spline", n_spline_coefficients=2241, spline_s_max=16.0,
    )
    if count_params(model) != 2241:
        raise RuntimeError(f"parameter mismatch: {count_params(model)}")
    samples = torch.tensor([0.0, 1e-6, 0.1, 1.0, 8.0, 16.0, 20.0])
    multiplier = model.m_law(samples)
    torch.testing.assert_close(multiplier, torch.ones_like(multiplier))
    loss = multiplier.square().sum()
    loss.backward()
    gradient = model.spline_coefficients.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().any():
        raise RuntimeError("invalid coefficient gradient")

    with torch.no_grad():
        model.spline_coefficients.copy_(torch.linspace(-0.2, 0.3, 2241))
    values = model.spline_function(torch.linspace(0.0, 16.0, 1001))
    if not torch.isfinite(values).all():
        raise RuntimeError("nonfinite spline evaluation")
    endpoint = model.spline_function(torch.tensor([16.0, 20.0]))
    torch.testing.assert_close(endpoint[0], endpoint[1])
    print("reviewer-3 spline CPU smoke passed")


if __name__ == "__main__":
    main()
