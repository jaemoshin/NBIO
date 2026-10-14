#!/usr/bin/env python
"""CPU-light algebra checks for the 3D radial surface factorization."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "surface3d"))

import radial_surface_law as radial


def main():
    torch.set_default_dtype(torch.float64)
    radius = torch.tensor([0.0, 1e-10, 0.1, 0.7, 2.0])
    kappa = 3.0
    scaled = kappa * radius
    q = radial.exact_q(scaled)
    assert torch.isfinite(q).all()
    assert torch.allclose(q[0], torch.tensor(-1.0))

    positive = radius > 0
    correction = kappa * q[positive] / (4 * math.pi)
    direct = torch.expm1(
        -kappa * radius[positive],
    ) / (4 * math.pi * radius[positive])
    assert torch.allclose(correction, direct, atol=1e-12, rtol=1e-10)

    torch.manual_seed(4)
    law = radial.RadialLaw("product")
    scale = 2.7
    original = kappa * law(radius[1:], kappa) / (4 * math.pi)
    scaled_geometry = radius[1:] * scale
    scaled_kappa = kappa / scale
    transformed = scaled_kappa * law(
        scaled_geometry, scaled_kappa,
    ) / (4 * math.pi)
    assert torch.allclose(transformed, original / scale, atol=1e-12, rtol=1e-12)

    mixture = radial.RadialMixture()
    q_mix = mixture(radius, kappa)
    multiplier = 1 + scaled * q_mix
    assert torch.isfinite(q_mix).all()
    assert torch.allclose(multiplier[0], torch.tensor(1.0))
    assert torch.all(multiplier >= 0)
    assert torch.all(multiplier[1:] <= multiplier[:-1])

    print("RADIAL_SURFACE_ALGEBRA_OK")


if __name__ == "__main__":
    main()
