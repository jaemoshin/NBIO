"""Torch implementation of the exact 2D screened double-layer BIE."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.layers import double_layer_kernel


def screened_multiplier(scaled_distance: torch.Tensor) -> torch.Tensor:
    safe = scaled_distance.clamp_min(1e-7)
    value = safe * torch.special.modified_bessel_k1(safe)
    return torch.where(scaled_distance < 1e-6, torch.ones_like(value), value)


class ExactScreenedBIE(nn.Module):
    def __init__(self, region: str = "interior"):
        super().__init__()
        self.jump = -0.5 if region == "interior" else 0.5

    def kernel(self, targets, sources, source_normals, kappa, curv_s=None,
               self_kernel=False):
        known = double_layer_kernel(
            targets, sources, source_normals, curv_s=curv_s,
            self_kernel=self_kernel,
        )
        radius = (
            sources[:, None] - targets[:, :, None]
        ).square().sum(-1).clamp_min(1e-18).sqrt()
        scaled = kappa.reshape(-1, 1, 1).to(radius) * radius
        multiplier = screened_multiplier(scaled)
        if self_kernel:
            diagonal = torch.arange(radius.shape[-1], device=radius.device)
            multiplier = multiplier.clone()
            multiplier[:, diagonal, diagonal] = 1.0
        return multiplier * known

    def assemble(self, batch):
        boundary = batch["Xb"]
        kernel = self.kernel(
            boundary, boundary, batch["Nb"], batch["kappa"],
            curv_s=batch["curv"], self_kernel=True,
        )
        identity = torch.eye(
            boundary.shape[1], device=boundary.device, dtype=boundary.dtype,
        ).unsqueeze(0)
        return self.jump * identity + kernel * batch["wb"][:, None]

    def potential(self, batch):
        return self.kernel(
            batch["Xq"], batch["Xb"], batch["Nb"], batch["kappa"],
        ) * batch["wb"][:, None]

    def forward(self, batch):
        matrix = self.assemble(batch)
        density = torch.linalg.solve(matrix, batch["g"].unsqueeze(-1))
        return torch.bmm(self.potential(batch), density).squeeze(-1)


def laplace_matrix(batch: dict[str, torch.Tensor], jump: float = -0.5):
    boundary = batch["Xb"]
    kernel = double_layer_kernel(
        boundary, boundary, batch["Nb"], curv_s=batch["curv"],
        self_kernel=True,
    )
    identity = torch.eye(
        boundary.shape[1], device=boundary.device, dtype=boundary.dtype,
    ).unsqueeze(0)
    return jump * identity + kernel * batch["wb"][:, None]
