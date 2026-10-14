"""Radially factored modified-Helmholtz boundary integral operators."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FixedPointSolve, double_layer_kernel
from ..numerics.qbx_torch import double_layer_qbx


class ScreeningLawNBIO(nn.Module):
    """Laplace double-layer normal form multiplied by a learned law of kappa*r."""

    def __init__(self, variant="mixture", n_atoms=8, hidden=32, n_layers=3,
                 region="interior", solve="direct", n_iter=20, omega=1.0,
                 kappa_scale=4.0, qbx=False, qbx_order=12, qbx_r_fac=4.0,
                 qbx_near_fac=2.0, qbx_oversample=4,
                 n_spline_coefficients=2241, spline_s_max=16.0):
        super().__init__()
        if variant not in {
            "mixture", "radial_mlp", "radial_endpoint", "radial_quadratic",
            "radial_asymptotic", "radial_separate", "radial_spline",
        }:
            raise ValueError(
                "variant must be 'mixture', 'radial_mlp', 'radial_endpoint', "
                "'radial_quadratic', 'radial_asymptotic', or 'radial_separate'"
            )
        if n_atoms < 1:
            raise ValueError("n_atoms must be positive")
        self.variant = variant
        self.n_atoms = int(n_atoms)
        self.kappa_scale = float(kappa_scale)
        if variant == "mixture":
            self.weight_logits = nn.Parameter(torch.zeros(n_atoms))
            rates = torch.linspace(-3.0, 1.7, n_atoms).exp()
            self.raw_rates = nn.Parameter(torch.log(torch.expm1(rates)))
        elif variant == "radial_spline":
            if n_spline_coefficients < 4:
                raise ValueError("a cubic spline needs at least four coefficients")
            if spline_s_max <= 0:
                raise ValueError("spline_s_max must be positive")
            self.spline_coefficients = nn.Parameter(
                torch.zeros(int(n_spline_coefficients))
            )
            self.spline_s_max = float(spline_s_max)
            self.register_buffer(
                "spline_knots",
                self._open_uniform_knots(int(n_spline_coefficients), degree=3),
            )
        else:
            layers = [nn.Linear(2, hidden), nn.GELU()]
            for _ in range(n_layers - 1):
                layers.extend((nn.Linear(hidden, hidden), nn.GELU()))
            output_channels = 2 if variant == "radial_asymptotic" else 1
            layers.append(nn.Linear(hidden, output_channels))
            self.radial_net = nn.Sequential(*layers)
            with torch.no_grad():
                self.radial_net[-1].weight.mul_(0.05)
                self.radial_net[-1].bias.zero_()
        self.solver = FixedPointSolve(solve, n_iter, omega)
        self.jump = -0.5 if region == "interior" else 0.5
        self.qbx = bool(qbx)
        self.qbx_kw = {
            "order": int(qbx_order),
            "r_fac": float(qbx_r_fac),
            "near_fac": float(qbx_near_fac),
            "oversample": int(qbx_oversample),
        }

    @staticmethod
    def _open_uniform_knots(n_coefficients: int, degree: int) -> torch.Tensor:
        n_spans = n_coefficients - degree
        interior = torch.arange(1, n_spans, dtype=torch.float64) / n_spans
        return torch.cat((
            torch.zeros(degree + 1, dtype=torch.float64),
            interior,
            torch.ones(degree + 1, dtype=torch.float64),
        ))

    def spline_function(self, scaled_distance: torch.Tensor) -> torch.Tensor:
        """Evaluate the fixed open-uniform cubic basis with de Boor basis recursion."""
        degree = 3
        n_coefficients = self.spline_coefficients.numel()
        n_spans = n_coefficients - degree
        t = (scaled_distance / self.spline_s_max).clamp(0.0, 1.0)
        span = torch.floor(t * n_spans).to(torch.long) + degree
        span = span.clamp(max=n_coefficients - 1)
        knots = self.spline_knots.to(device=t.device, dtype=t.dtype)

        basis = [torch.ones_like(t)]
        left = [None]
        right = [None]
        for order in range(1, degree + 1):
            left.append(t - knots[span + 1 - order])
            right.append(knots[span + order] - t)
            saved = torch.zeros_like(t)
            next_basis = []
            for index in range(order):
                denominator = right[index + 1] + left[order - index]
                ratio = basis[index] / denominator.clamp_min(
                    torch.finfo(t.dtype).eps
                )
                next_basis.append(saved + right[index + 1] * ratio)
                saved = left[order - index] * ratio
            next_basis.append(saved)
            basis = next_basis
        basis_values = torch.stack(basis, dim=-1)
        coefficient_indices = span.unsqueeze(-1) - degree + torch.arange(
            degree + 1, device=span.device,
        )
        coefficients = self.spline_coefficients[coefficient_indices]
        return (basis_values * coefficients).sum(dim=-1)

    def m_law(self, scaled_distance: torch.Tensor, *, kappa=None, radius=None) -> torch.Tensor:
        """Evaluate m(kappa*r); the mixture is positive, monotone, and normalized."""
        if self.variant == "mixture":
            rates = torch.nn.functional.softplus(self.raw_rates)
            weights = torch.softmax(self.weight_logits, dim=0)
            scaled = scaled_distance.unsqueeze(-1) * rates
            return ((1.0 + scaled) * torch.exp(-scaled) * weights).sum(dim=-1)
        if self.variant == "radial_spline":
            return 1.0 + scaled_distance * self.spline_function(scaled_distance)
        if self.variant == "radial_separate":
            if kappa is None or radius is None:
                raise ValueError("radial_separate requires kappa and radius")
            normalized_kappa = kappa.reshape(-1, 1, 1).to(radius) / self.kappa_scale
            normalized_kappa = normalized_kappa.expand_as(radius)
            features = torch.stack((normalized_kappa, radius), dim=-1)
        else:
            features = torch.stack((scaled_distance, torch.log1p(scaled_distance)), dim=-1)
        correction = self.radial_net(features)
        if self.variant == "radial_asymptotic":
            coefficient_log, coefficient_regular = correction.unbind(dim=-1)
            safe_distance = scaled_distance.clamp_min(
                torch.finfo(scaled_distance.dtype).tiny
            )
            log_basis = torch.where(
                scaled_distance > 0,
                scaled_distance.square() * safe_distance.log(),
                torch.zeros_like(scaled_distance),
            )
            return (
                1.0
                + log_basis * coefficient_log
                + scaled_distance.square() * coefficient_regular
            )
        correction = correction.squeeze(-1)
        if self.variant == "radial_endpoint":
            # f_theta(0, 0) is a scalar. Evaluate it once and broadcast instead
            # of materializing one identical MLP path per source-target pair.
            f0 = self.radial_net(features.new_zeros(2)).squeeze(-1)
            correction = correction - f0
        if self.variant == "radial_quadratic":
            return 1.0 + scaled_distance.square() * correction
        return 1.0 + scaled_distance * correction

    def kernel(self, targets, sources, source_normals, kappa, curv_s=None,
               self_kernel=False):
        known = double_layer_kernel(
            targets, sources, source_normals, curv_s=curv_s,
            self_kernel=self_kernel,
        )
        radius = (sources[:, None] - targets[:, :, None]).square().sum(-1)
        radius = radius.clamp_min(1e-18).sqrt()
        scaled_distance = kappa.reshape(-1, 1, 1).to(radius) * radius
        multiplier = self.m_law(
            scaled_distance, kappa=kappa, radius=radius,
        )
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
        kernel = self.kernel(
            batch["Xq"], batch["Xb"], batch["Nb"], batch["kappa"],
        )
        return kernel * batch["wb"][:, None]

    def split_qbx_potential(self, batch, density):
        """Evaluate the Laplace singular term by QBX and the regular radial remainder directly."""
        singular = double_layer_qbx(
            batch["Xq"], batch["Xb"], density,
            Nb=batch["Nb"], wb=batch["wb"], **self.qbx_kw,
        )
        radius = (
            batch["Xb"][:, None] - batch["Xq"][:, :, None]
        ).square().sum(-1).clamp_min(1e-18).sqrt()
        scaled_distance = batch["kappa"].reshape(-1, 1, 1).to(radius) * radius
        multiplier = self.m_law(
            scaled_distance, kappa=batch["kappa"], radius=radius,
        )
        laplace = double_layer_kernel(
            batch["Xq"], batch["Xb"], batch["Nb"],
        )
        remainder = laplace * (multiplier - 1.0) * batch["wb"][:, None]
        return singular + torch.bmm(remainder, density.unsqueeze(-1)).squeeze(-1)

    def evaluate_density(self, batch, density):
        if self.qbx:
            return self.split_qbx_potential(batch, density)
        return torch.bmm(self.potential(batch), density.unsqueeze(-1)).squeeze(-1)

    def forward(self, batch):
        if "kappa" not in batch:
            raise KeyError("ScreeningLawNBIO requires batch['kappa']")
        matrix = self.assemble(batch)
        density = self.solver(matrix, batch["g"], self.jump)
        return self.evaluate_density(batch, density)
