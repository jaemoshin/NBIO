"""Method 4: low-rank / separable NBIO smooth correction.

NBIO's accuracy comes from a *smooth* learned correction added to the known double-layer kernel
(layers.KernelHypernet, a pairwise hypernet over 9 geometric features -> a dense (B,T,S) matrix). That
dense correction makes the boundary system matrix-vector product O(n^2).

This module tests whether the learned correction can instead be SEPARABLE / low-rank:

    K_smooth[i,j] = phi(node_i) . psi(node_j)              (sum over r latent channels)

where phi, psi are small per-node MLPs mapping per-node features (x, y, nx, ny, curv) -> R^r. With
factors PHI (B,T,r) and PSI (B,S,r), the correction matvec is

    (PHI @ (PSI^T @ v))                                   -> O(n . r)   instead of O(n^2).

Everything else (the known double-layer kernel, the arc weights, the differentiable 2nd-kind solve) is
kept exactly as in NBIO -- we import double_layer_kernel and FixedPointSolve and reuse them unchanged.
Only the LEARNED correction is replaced by the factored form.

NOTE: a strictly separable kernel phi(x).psi(y) is itself a rank-r kernel; assembling it densely costs
O(n^2 . r) but the *matvec* (the operationally relevant cost inside the solve / potential eval) is
O(n . r). For the differentiable direct solve we still need the dense matrix A, but the cheap-matvec
property is what enables iterative (Neumann) solves and fast potential evaluation; the microbenchmark in
scripts/scal_factored.py measures the matvec cost directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FixedPointSolve, double_layer_kernel


class _NodeMLP(nn.Module):
    """Per-node feature MLP: (B, N, din) -> (B, N, r). Small, GELU, with a tiny output init so the
    correction starts near zero (model ~= classical solver at init, like KernelHypernet)."""

    def __init__(self, din, hidden, n_layers, rank, init_scale=0.05):
        super().__init__()
        layers = [nn.Linear(din, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, rank)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(init_scale)
            self.net[-1].bias.zero_()

    def forward(self, feats):
        return self.net(feats)


class FactoredNBIO(nn.Module):
    """NBIO with a SEPARABLE/low-rank learned smooth correction K_smooth[i,j] = phi(node_i).psi(node_j).

    Per-node features are [x, y, nx, ny, curv] (5 dims). Query points have no normal/curvature; following
    NBIO.potential, we feed zeros for those node features on the query side. The known double-layer kernel
    (with curvature self-term) and the differentiable solve are imported from layers and used unchanged.

    The factored correction matvec is exposed via `correction_matvec` (O(n.r)); the dense pairwise NBIO
    correction is what it is benchmarked against in scripts/scal_factored.py.
    """

    NODE_FEAT_DIM = 5  # x, y, nx, ny, curv

    def __init__(self, rank=8, hidden=64, n_layers=3, feat_scale=1.0, region="interior",
                 split=True, solve="direct", n_iter=20, omega=1.0, init_scale=0.05):
        super().__init__()
        self.rank = int(rank)
        self.feat_scale = feat_scale
        self.split = split
        self.phi = _NodeMLP(self.NODE_FEAT_DIM, hidden, n_layers, self.rank, init_scale)
        self.psi = _NodeMLP(self.NODE_FEAT_DIM, hidden, n_layers, self.rank, init_scale)
        # learned scalar gate on the known double-layer part (same parameterization as NBIO/BIO)
        self.log_gate = nn.Parameter(torch.zeros(())) if split else None
        self.solver = FixedPointSolve(solve, n_iter, omega)
        self.jump = -0.5 if region == "interior" else 0.5

    # ---- gate ---------------------------------------------------------------
    def _gate(self):
        return (1.0 + torch.tanh(self.log_gate)) if self.log_gate is not None else 1.0

    # ---- per-node features --------------------------------------------------
    def _node_feats(self, X, N=None, curv=None):
        """(B,N,2) positions [+ (B,N,2) normals + (B,N) curvature] -> (B,N,5) features.

        For query nodes (no normal / curvature available) pass N=None, curv=None -> zeros, mirroring
        NBIO.potential which uses a zero normal for off-surface query points."""
        B, n = X.shape[0], X.shape[1]
        pos = X * self.feat_scale
        if N is None:
            N = torch.zeros(B, n, 2, device=X.device, dtype=X.dtype)
        if curv is None:
            c = torch.zeros(B, n, 1, device=X.device, dtype=X.dtype)
        else:
            c = curv.unsqueeze(-1)
        return torch.cat([pos, N, c], dim=-1)  # (B,N,5)

    def factors_phi(self, X, N=None, curv=None):
        """Left factor PHI (B,T,r) for the target nodes."""
        return self.phi(self._node_feats(X, N, curv))

    def factors_psi(self, X, N=None, curv=None):
        """Right factor PSI (B,S,r) for the source nodes."""
        return self.psi(self._node_feats(X, N, curv))

    # ---- factored correction ------------------------------------------------
    def correction_matrix(self, PHI, PSI):
        """Dense (B,T,S) correction K_smooth = PHI @ PSI^T (only needed for the direct dense solve)."""
        return torch.bmm(PHI, PSI.transpose(1, 2))

    @staticmethod
    def correction_matvec(PHI, PSI, v):
        """O(n.r) matvec of the separable correction: (PHI @ (PSI^T @ v)).

        PHI (B,T,r), PSI (B,S,r), v (B,S) -> (B,T). Used by the microbenchmark and (optionally) by an
        iterative solve / potential evaluation where the dense matrix is never formed."""
        tmp = torch.bmm(PSI.transpose(1, 2), v.unsqueeze(-1))   # (B,r,1)  = PSI^T v
        return torch.bmm(PHI, tmp).squeeze(-1)                  # (B,T)    = PHI (PSI^T v)

    # ---- assembly (mirrors BoundaryIntegralOperator / NBIO) -----------------
    def _assemble(self, Xb, Nb, wb, curv):
        PHI = self.factors_phi(Xb, Nb, curv)                    # (B,nb,r)
        PSI = self.factors_psi(Xb, Nb, curv)                    # (B,nb,r)
        k_smooth = self.correction_matrix(PHI, PSI)             # (B,nb,nb)
        if self.split:
            k_known = double_layer_kernel(Xb, Xb, Nb, curv_s=curv, self_kernel=True)
            k = self._gate() * k_known + k_smooth
        else:
            k = k_smooth
        D0 = k * wb[:, None, :]                                  # arc-weighted
        n = Xb.shape[1]
        A = self.jump * torch.eye(n, device=Xb.device).unsqueeze(0) + D0
        return A

    def _potential(self, Xq, Xb, Nb, wb, curv, mu):
        """u(x_q) = sum_j P[q,j] mu_j, with the factored smooth correction and (optionally) known part.

        Query (target) nodes have no normal/curvature -> zero node features on the query side, mirroring
        NBIO.potential. Boundary (source) nodes keep their normals and curvature."""
        PHI_q = self.factors_phi(Xq, None, None)                # (B,Q,r)   query targets
        PSI_b = self.factors_psi(Xb, Nb, curv)                  # (B,nb,r)  boundary sources
        smooth = self.correction_matvec(PHI_q, PSI_b, mu * wb)  # (B,Q)  factored, O(Q.r + nb.r)
        if self.split:
            k_known = double_layer_kernel(Xq, Xb, Nb)           # (B,Q,nb)
            Pk = self._gate() * k_known * wb[:, None, :]
            known = torch.bmm(Pk, mu.unsqueeze(-1)).squeeze(-1)
            return known + smooth
        return smooth

    # ---- forward ------------------------------------------------------------
    def forward(self, batch):
        Xb, Nb, wb, curv, g, Xq = (batch["Xb"], batch["Nb"], batch["wb"],
                                   batch["curv"], batch["g"], batch["Xq"])
        A = self._assemble(Xb, Nb, wb, curv)
        mu = self.solver(A, g, self.jump)
        return self._potential(Xq, Xb, Nb, wb, curv, mu)

    @torch.no_grad()
    def condition_number(self, batch):
        A = self._assemble(batch["Xb"], batch["Nb"], batch["wb"], batch["curv"])
        return torch.linalg.cond(A).mean().item()
