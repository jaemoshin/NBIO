#!/usr/bin/env python
"""CPU smoke and diagonal-continuity check for smooth conditioned features."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.nbio_kappa_conditioned import (  # noqa: E402
    KappaConditionedBIO,
    KappaConditionedKernel,
    KappaConditionedNBIO,
)


def main() -> None:
    torch.manual_seed(17)
    dtype = torch.float64
    kernel = KappaConditionedKernel(
        hidden=16, n_layers=2, feature_mode="smooth"
    ).to(dtype=dtype)
    xt = torch.zeros((1, 1, 2), dtype=dtype)
    nt = torch.tensor([[[1.0, 0.0]]], dtype=dtype)
    ns = torch.tensor([[[1.0, 0.0]]], dtype=dtype)
    kappa = torch.tensor([2.0], dtype=dtype)
    diagonal = kernel(xt, xt, nt, ns, kappa)
    differences = []
    for epsilon in (1e-2, 1e-4, 1e-6):
        plus = torch.tensor([[[epsilon, 0.0]]], dtype=dtype)
        minus = torch.tensor([[[-epsilon, 0.0]]], dtype=dtype)
        values = torch.stack((
            kernel(xt, plus, nt, ns, kappa).squeeze(),
            kernel(xt, minus, nt, ns, kappa).squeeze(),
        ))
        differences.append(float((values - diagonal.squeeze()).abs().max()))
    assert differences[2] < differences[1] < differences[0]
    assert differences[2] < 1e-5

    # Fair endpoint contract: only a prior-bearing residual is gated to zero.
    reference = torch.ones((2, 3, 4), dtype=dtype)
    endpoint = torch.zeros(2, dtype=dtype)
    full_bio = KappaConditionedBIO(prior_mode="full").to(dtype=dtype)
    free_bio = KappaConditionedBIO(prior_mode="none").to(dtype=dtype)
    assert torch.equal(full_bio._correction_scale(endpoint, reference), torch.zeros((2, 1, 1), dtype=dtype))
    assert torch.equal(free_bio._correction_scale(endpoint, reference), torch.ones((2, 1, 1), dtype=dtype))

    n_b = 16
    theta = torch.arange(n_b, dtype=torch.float32) * (2.0 * math.pi / n_b)
    xb = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1).unsqueeze(0)
    batch = {
        "Xb": xb,
        "Nb": xb.clone(),
        "wb": torch.full((1, n_b), 2.0 * math.pi / n_b),
        "curv": torch.ones((1, n_b)),
        "g": torch.cos(theta).unsqueeze(0),
        "Xq": torch.tensor([[[0.1, 0.1], [0.2, -0.1]]]),
        "kappa": torch.tensor([2.0]),
    }
    model = KappaConditionedNBIO(
        hidden=16, n_layers=2, prior_mode="full", feature_mode="smooth"
    )
    output = model(batch)
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    print("PASS", differences, output.tolist())


if __name__ == "__main__":
    main()
