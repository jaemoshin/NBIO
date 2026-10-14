"""Neural Boundary Integral Operator (NBIO) -- the proposed method.

Maps (boundary geometry + Dirichlet data g) -> solution u at arbitrary query points, by:
  M1  assembling a 2nd-kind boundary system A = (jump I + W K_theta), K_theta = known double-layer +
      learned smooth correction;
  M2  solving A mu = g for the boundary density mu (differentiable direct / Neumann fixed point);
  M3  evaluating the off-surface potential u(x_q) = P_theta(x_q, .) mu -- continuous in x_q and in the
      boundary arc-length, hence zero-shot across geometries, boundary resolution, and (interior/exterior).
Ablations: split=False (drop the known kernel), solve='neumann' (vs direct).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import BoundaryIntegralOperator, FixedPointSolve, double_layer_kernel
from ..numerics.qbx_torch import double_layer_qbx


class NBIO(nn.Module):
    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0, region="interior",
                 split=True, solve="direct", n_iter=20, omega=1.0,
                 qbx=False, qbx_order=10, qbx_r_fac=4.0, qbx_near_fac=2.0, kappa_cond=False):
        super().__init__()
        self.bio = BoundaryIntegralOperator(hidden, n_layers, feat_scale, learn_known_gate=split,
                                            kappa_cond=kappa_cond)
        self.split = split
        self.kappa_cond = kappa_cond
        self.solver = FixedPointSolve(solve, n_iter, omega)
        self.jump = -0.5 if region == "interior" else 0.5
        # qbx=True: evaluate the KNOWN double layer by differentiable QBX (close-eval), learned smooth
        # correction stays direct (no singularity) -> neural-operator properties preserved.
        self.qbx = qbx; self.qbx_kw = dict(order=qbx_order, r_fac=qbx_r_fac, near_fac=qbx_near_fac)

    def _assemble_nosplit(self, Xb, Nb, wb, kappa=None):
        k = self.bio.smooth(Xb, Xb, Nb, Nb, kappa)
        D0 = k * wb[:, None, :]
        n = Xb.shape[1]
        return self.jump * torch.eye(n, device=Xb.device).unsqueeze(0) + D0

    def forward(self, batch):
        Xb, Nb, wb, curv, g, Xq = (batch["Xb"], batch["Nb"], batch["wb"],
                                   batch["curv"], batch["g"], batch["Xq"])
        kappa = batch.get("kappa") if self.kappa_cond else None
        A = (self.bio.assemble(Xb, Nb, wb, curv, self.jump, kappa) if self.split
             else self._assemble_nosplit(Xb, Nb, wb, kappa))
        mu = self.solver(A, g, self.jump)
        Nq = torch.zeros(Xq.shape[0], Xq.shape[1], 2, device=Xq.device)
        if self.qbx:
            # known double layer via differentiable QBX (close-eval-safe) + learned smooth correction direct
            u = double_layer_qbx(Xq, Xb, mu, Nb=Nb, wb=wb, **self.qbx_kw)
            if self.split:
                u = self.bio._gate() * u
                Psm = self.bio.smooth(Xq, Xb, Nq, Nb, kappa) * wb[:, None, :]
                u = u + torch.bmm(Psm, mu.unsqueeze(-1)).squeeze(-1)
            return u
        if self.split:
            P = self.bio.potential(Xq, Xb, Nb, wb, kappa)
        else:
            P = self.bio.smooth(Xq, Xb, Nq, Nb, kappa) * wb[:, None, :]
        u = torch.bmm(P, mu.unsqueeze(-1)).squeeze(-1)
        return u

    @torch.no_grad()
    def condition_number(self, batch):
        A = self.bio.assemble(batch["Xb"], batch["Nb"], batch["wb"], batch["curv"], self.jump)
        return torch.linalg.cond(A).mean().item()
