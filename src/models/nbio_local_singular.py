"""NBIO variant that retains only trusted local singular structure.

The full Laplace and modified-Helmholtz kernels can differ substantially away
from the diagonal while sharing the same leading local singularity.  This
module multiplies the analytic Laplace double layer by a smooth radial window
and learns the remaining kernel.  It is intentionally separate from the
original NBIO implementation so existing checkpoints and results are
unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FixedPointSolve, KernelHypernet, double_layer_kernel
from ..numerics.qbx_torch import double_layer_qbx


def local_prior_weight(Xt, Xs, cutoff: float, power: float = 4.0):
    """Smooth weight equal to one on the diagonal and decaying away from it."""
    if cutoff <= 0:
        raise ValueError("prior_cutoff must be positive")
    if power <= 0:
        raise ValueError("prior_power must be positive")
    distance = (Xs[:, None, :, :] - Xt[:, :, None, :]).norm(dim=-1)
    return torch.exp(-torch.pow(distance / cutoff, power))


class LocalizedSingularBIO(nn.Module):
    """Boundary integral operator with a distance-localized analytic prior."""

    VALID_SCOPES = {"both", "potential", "assembly"}

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0,
                 prior_cutoff=0.2, prior_power=4.0, prior_scope="both"):
        super().__init__()
        if prior_scope not in self.VALID_SCOPES:
            raise ValueError(f"prior_scope must be one of {sorted(self.VALID_SCOPES)}")
        self.smooth = KernelHypernet(hidden, n_layers, feat_scale)
        self.prior_cutoff = float(prior_cutoff)
        self.prior_power = float(prior_power)
        self.prior_scope = prior_scope

    def _localized_known(self, Xt, Xs, Ns, curv_s=None, self_kernel=False):
        known = double_layer_kernel(
            Xt, Xs, Ns, curv_s=curv_s, self_kernel=self_kernel
        )
        weight = local_prior_weight(
            Xt, Xs, cutoff=self.prior_cutoff, power=self.prior_power
        )
        return weight * known

    def assemble(self, Xb, Nb, wb, curv, jump):
        kernel = self.smooth(Xb, Xb, Nb, Nb)
        if self.prior_scope in {"both", "assembly"}:
            kernel = kernel + self._localized_known(
                Xb, Xb, Nb, curv_s=curv, self_kernel=True
            )
        n = Xb.shape[1]
        identity = torch.eye(n, device=Xb.device, dtype=Xb.dtype).unsqueeze(0)
        return jump * identity + kernel * wb[:, None, :]

    def potential(self, Xq, Xb, Nb, wb):
        Nq = torch.zeros_like(Xq)
        kernel = self.smooth(Xq, Xb, Nq, Nb)
        if self.prior_scope in {"both", "potential"}:
            kernel = kernel + self._localized_known(Xq, Xb, Nb)
        return kernel * wb[:, None, :]


class LocalizedSingularNBIO(nn.Module):
    """NBIO with a trusted local singular prior and learned global remainder."""

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0,
                 region="interior", solve="direct", n_iter=20, omega=1.0,
                 prior_cutoff=0.2, prior_power=4.0, prior_scope="both",
                 qbx=False, qbx_order=10, qbx_r_fac=4.0, qbx_near_fac=2.0,
                 qbx_oversample=4):
        super().__init__()
        self.bio = LocalizedSingularBIO(
            hidden=hidden,
            n_layers=n_layers,
            feat_scale=feat_scale,
            prior_cutoff=prior_cutoff,
            prior_power=prior_power,
            prior_scope=prior_scope,
        )
        self.solver = FixedPointSolve(solve, n_iter, omega)
        self.jump = -0.5 if region == "interior" else 0.5
        self.qbx = bool(qbx)
        self.qbx_kw = {
            "order": int(qbx_order),
            "r_fac": float(qbx_r_fac),
            "near_fac": float(qbx_near_fac),
            "oversample": int(qbx_oversample),
        }

    def forward(self, batch):
        Xb, Nb, wb, curv, g, Xq = (
            batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
            batch["g"], batch["Xq"],
        )
        matrix = self.bio.assemble(Xb, Nb, wb, curv, self.jump)
        density = self.solver(matrix, g, self.jump)
        if self.qbx and self.bio.prior_scope in {"both", "potential"}:
            # Evaluate chi(r) K0 as K0_QBX - (1-chi(r)) K0_direct.  The
            # subtracted term is regular at r=0 because 1-chi(r)=O(r^p).
            Nq = torch.zeros_like(Xq)
            smooth = self.bio.smooth(Xq, Xb, Nq, Nb) * wb[:, None, :]
            smooth_value = torch.bmm(smooth, density.unsqueeze(-1)).squeeze(-1)
            full_known = double_layer_qbx(
                Xq, Xb, density, Nb=Nb, wb=wb, **self.qbx_kw
            )
            known_direct = double_layer_kernel(Xq, Xb, Nb)
            local_weight = local_prior_weight(
                Xq, Xb, self.bio.prior_cutoff, self.bio.prior_power
            )
            regular_far = (1.0 - local_weight) * known_direct * wb[:, None, :]
            far_value = torch.bmm(regular_far, density.unsqueeze(-1)).squeeze(-1)
            return smooth_value + full_known - far_value
        potential = self.bio.potential(Xq, Xb, Nb, wb)
        return torch.bmm(potential, density.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def condition_number(self, batch):
        matrix = self.bio.assemble(
            batch["Xb"], batch["Nb"], batch["wb"], batch["curv"], self.jump
        )
        return torch.linalg.cond(matrix).mean().item()
