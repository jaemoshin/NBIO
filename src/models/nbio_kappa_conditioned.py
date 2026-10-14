"""Parameter-conditioned NBIO for a family of modified-Helmholtz kernels."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FixedPointSolve, double_layer_kernel


class KappaConditionedKernel(nn.Module):
    """Pairwise correction conditioned on the PDE parameter kappa."""

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0,
                 kappa_scale=4.0, init_scale=0.05, feature_mode="legacy"):
        super().__init__()
        if feature_mode not in {"legacy", "smooth"}:
            raise ValueError("feature_mode must be 'legacy' or 'smooth'")
        self.feat_scale = float(feat_scale)
        self.kappa_scale = float(kappa_scale)
        self.feature_mode = feature_mode
        layers = [nn.Linear(10, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend((nn.Linear(hidden, hidden), nn.GELU()))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(init_scale)
            self.net[-1].bias.zero_()

    def forward(self, Xt, Xs, Nt, Ns, kappa):
        diff = Xs[:, None, :, :] - Xt[:, :, None, :]
        target_normal = Nt[:, :, None, :].expand_as(diff)
        source_normal = Ns[:, None, :, :].expand_as(diff)
        parameter = kappa.reshape(-1, 1, 1, 1).to(diff) / self.kappa_scale
        parameter = parameter.expand(*diff.shape[:-1], 1)
        if self.feature_mode == "smooth":
            scaled_diff = diff * self.feat_scale
            squared_distance = scaled_diff.square().sum(dim=-1, keepdim=True)
            features = torch.cat((
                scaled_diff,
                squared_distance,
                torch.exp(-squared_distance),
                (scaled_diff * target_normal).sum(-1, keepdim=True),
                (scaled_diff * source_normal).sum(-1, keepdim=True),
                (target_normal * source_normal).sum(-1, keepdim=True),
                target_normal,
                parameter,
            ), dim=-1)
        else:
            distance = diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            direction = diff / distance
            features = torch.cat((
                direction,
                distance * self.feat_scale,
                torch.exp(-distance * self.feat_scale),
                (direction * target_normal).sum(-1, keepdim=True),
                (direction * source_normal).sum(-1, keepdim=True),
                (target_normal * source_normal).sum(-1, keepdim=True),
                target_normal,
                parameter,
            ), dim=-1)
        return self.net(features).squeeze(-1)


class KappaConditionedBIO(nn.Module):
    """Full, absent, or asymptotically localized Laplace prior across kappa."""

    VALID_PRIORS = {"full", "none", "local", "adaptive"}
    VALID_LOCAL_SCOPES = {"both", "potential"}
    VALID_RESIDUAL_SCALES = {"normalized", "saturating"}

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0,
                 kappa_scale=4.0, prior_mode="local", local_tau=0.32,
                 local_power=4.0, residual_order=2.0, local_scope="both",
                 residual_scale="normalized", feature_mode="legacy"):
        super().__init__()
        if prior_mode not in self.VALID_PRIORS:
            raise ValueError(f"prior_mode must be one of {sorted(self.VALID_PRIORS)}")
        if local_scope not in self.VALID_LOCAL_SCOPES:
            raise ValueError(
                f"local_scope must be one of {sorted(self.VALID_LOCAL_SCOPES)}"
            )
        if residual_scale not in self.VALID_RESIDUAL_SCALES:
            raise ValueError(
                f"residual_scale must be one of {sorted(self.VALID_RESIDUAL_SCALES)}"
            )
        if local_tau <= 0 or local_power <= 0 or residual_order <= 0:
            raise ValueError("local_tau, local_power, and residual_order must be positive")
        self.correction = KappaConditionedKernel(
            hidden, n_layers, feat_scale, kappa_scale=kappa_scale,
            feature_mode=feature_mode,
        )
        self.prior_mode = prior_mode
        self.kappa_scale = float(kappa_scale)
        self.local_tau = float(local_tau)
        self.local_power = float(local_power)
        self.residual_order = float(residual_order)
        self.local_scope = local_scope
        self.residual_scale = residual_scale
        if prior_mode == "adaptive":
            initial_rate = torch.tensor(0.5)
            self.log_gate_rate = nn.Parameter(torch.log(torch.expm1(initial_rate)))
        else:
            self.register_parameter("log_gate_rate", None)

    def _correction_scale(self, kappa, reference):
        shape = (reference.shape[0], 1, 1)
        if self.prior_mode == "none":
            return torch.ones(shape, device=reference.device, dtype=reference.dtype)
        parameter = kappa.reshape(-1, 1, 1).to(reference)
        if self.residual_scale == "saturating":
            powered = torch.pow(parameter, self.residual_order)
            return powered / (1.0 + powered)
        normalized = parameter / self.kappa_scale
        return torch.pow(normalized, self.residual_order)

    def _prior_weight(self, Xt, Xs, kappa, interaction="potential"):
        shape = (Xt.shape[0], Xt.shape[1], Xs.shape[1])
        if self.prior_mode == "none":
            return torch.zeros(shape, device=Xt.device, dtype=Xt.dtype)
        if self.prior_mode == "full":
            return torch.ones(shape, device=Xt.device, dtype=Xt.dtype)
        if interaction == "assembly" and self.local_scope == "potential":
            parameter = kappa.reshape(-1, 1, 1).to(Xt) / self.local_tau
            gate = torch.exp(-torch.pow(parameter, self.local_power))
            return gate.expand(shape)
        distance = (Xs[:, None, :, :] - Xt[:, :, None, :]).norm(dim=-1)
        scaled = kappa.reshape(-1, 1, 1).to(distance) * distance / self.local_tau
        local = torch.exp(-torch.pow(scaled, self.local_power))
        if self.prior_mode == "adaptive":
            rate = torch.nn.functional.softplus(self.log_gate_rate).to(distance)
            parameter = kappa.reshape(-1, 1, 1).to(distance)
            gate = torch.exp(-rate * parameter.square())
            return gate + (1.0 - gate) * local
        return local

    def _kernel(self, Xt, Xs, Nt, Ns, kappa, curv_s=None, self_kernel=False,
                interaction="potential"):
        correction = self.correction(Xt, Xs, Nt, Ns, kappa)
        correction = self._correction_scale(kappa, correction) * correction
        if self.prior_mode == "none":
            return correction
        known = double_layer_kernel(
            Xt, Xs, Ns, curv_s=curv_s, self_kernel=self_kernel
        )
        return self._prior_weight(Xt, Xs, kappa, interaction) * known + correction

    def assemble(self, Xb, Nb, wb, curv, kappa, jump):
        kernel = self._kernel(
            Xb, Xb, Nb, Nb, kappa, curv_s=curv, self_kernel=True,
            interaction="assembly",
        )
        n = Xb.shape[1]
        identity = torch.eye(n, device=Xb.device, dtype=Xb.dtype).unsqueeze(0)
        return jump * identity + kernel * wb[:, None, :]

    def potential(self, Xq, Xb, Nb, wb, kappa):
        target_normal = torch.zeros_like(Xq)
        kernel = self._kernel(
            Xq, Xb, target_normal, Nb, kappa, interaction="potential"
        )
        return kernel * wb[:, None, :]


class KappaConditionedNBIO(nn.Module):
    """One NBIO checkpoint for a kappa-parameterized elliptic operator family."""

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0,
                 region="interior", solve="direct", n_iter=20, omega=1.0,
                 kappa_scale=4.0, prior_mode="local", local_tau=0.32,
                 local_power=4.0, residual_order=2.0, local_scope="both",
                 residual_scale="normalized", feature_mode="legacy"):
        super().__init__()
        self.bio = KappaConditionedBIO(
            hidden=hidden,
            n_layers=n_layers,
            feat_scale=feat_scale,
            kappa_scale=kappa_scale,
            prior_mode=prior_mode,
            local_tau=local_tau,
            local_power=local_power,
            residual_order=residual_order,
            local_scope=local_scope,
            residual_scale=residual_scale,
            feature_mode=feature_mode,
        )
        self.solver = FixedPointSolve(solve, n_iter, omega)
        self.jump = -0.5 if region == "interior" else 0.5

    def forward(self, batch):
        if "kappa" not in batch:
            raise KeyError("KappaConditionedNBIO requires batch['kappa']")
        matrix = self.bio.assemble(
            batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
            batch["kappa"], self.jump,
        )
        density = self.solver(matrix, batch["g"], self.jump)
        potential = self.bio.potential(
            batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"], batch["kappa"]
        )
        return torch.bmm(potential, density.unsqueeze(-1)).squeeze(-1)
