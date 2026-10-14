"""Core layers for the Neural Boundary Integral Operator (NBIO).

M1  analytically-split kernel:   K = K_singular(known double-layer Green's part) + K_smooth(hypernetwork).
M2  differentiable 2nd-kind solve: (jump*I + W*K) mu = g   solved by a learned/Neumann fixed point.
The known double-layer kernel is bounded with a curvature diagonal limit, so it is fed as an exact
feature and the network only learns a SMOOTH correction -> accuracy + well-conditioning, not from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

INV_2PI = 1.0 / (2.0 * torch.pi)


def double_layer_kernel(Xt, Xs, Ns, curv_s=None, self_kernel=False, eps=1e-9):
    """Known double-layer kernel k(x_i,y_j) = -1/(2pi) (y-x).n_y / |y-x|^2.

    Xt (B,T,2) targets, Xs (B,S,2)/Ns (B,S,2) sources+normals. If self_kernel, set the diagonal to the
    curvature limit -curv/(4pi). Returns (B,T,S)."""
    diff = Xs[:, None, :, :] - Xt[:, :, None, :]            # (B,T,S,2) = y - x
    r2 = (diff ** 2).sum(-1)                                # (B,T,S)
    num = (diff * Ns[:, None, :, :]).sum(-1)                # (y-x).n_y
    k = (-INV_2PI) * num / r2.clamp_min(eps)
    if self_kernel:
        B, S = Xs.shape[0], Xs.shape[1]
        diag = (-curv_s / (4 * torch.pi)).to(k.dtype)       # (B,S)
        idx = torch.arange(S, device=k.device)
        k[:, idx, idx] = diag
    return k


class KernelHypernet(nn.Module):
    """Smooth correction kernel g_theta(features(x_i,y_j)) -> scalar, with a decaying envelope so the
    correction stays local/bounded (and the known singular part dominates near-diagonal)."""

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0, init_scale=0.05, kappa_cond=False):
        super().__init__()
        self.feat_scale = feat_scale
        self.kappa_cond = kappa_cond
        din = 10 if kappa_cond else 9    # +1 for log(1+kappa) when conditioning on the PDE parameter
        layers = [nn.Linear(din, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            # init_scale=0 -> correction starts at exactly 0 (model == classical solver at init), which
            # matters when the assembled system is moderately conditioned (transmission first-kind block).
            self.net[-1].weight.mul_(init_scale); self.net[-1].bias.zero_()

    def _features(self, Xt, Xs, Nt, Ns, kappa=None):
        diff = Xs[:, None, :, :] - Xt[:, :, None, :]        # (B,T,S,2)
        r = diff.norm(dim=-1, keepdim=True).clamp_min(1e-6) # (B,T,S,1)
        dh = diff / r
        nt = Nt[:, :, None, :].expand_as(diff)
        ns = Ns[:, None, :, :].expand_as(diff)
        feats = [dh, r * self.feat_scale, torch.exp(-r * self.feat_scale),
                 (dh * nt).sum(-1, keepdim=True), (dh * ns).sum(-1, keepdim=True),
                 (nt * ns).sum(-1, keepdim=True), nt]
        if self.kappa_cond:
            if kappa is None:
                raise ValueError("kappa_cond=True requires a kappa input")
            k = torch.as_tensor(kappa, dtype=diff.dtype, device=diff.device).reshape(-1)
            lk = torch.log1p(k).reshape(-1, 1, 1, 1).expand(diff.shape[0], diff.shape[1], diff.shape[2], 1)
            feats.append(lk)                                # (B,T,S,1) = log(1+kappa)
        return torch.cat(feats, dim=-1)

    def forward(self, Xt, Xs, Nt, Ns, kappa=None):
        f = self._features(Xt, Xs, Nt, Ns, kappa)
        return self.net(f).squeeze(-1)                      # (B,T,S)


class BoundaryIntegralOperator(nn.Module):
    """K = (1+s)*K_known + K_smooth, with a learned scalar gate s on the known part. Applies arc weights.

    `assemble` builds the boundary system matrix A=(jump I + W*K0) for the 2nd-kind solve; `evaluate`
    builds the off-surface potential matrix (targets in domain) to map density -> u at query points."""

    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0, learn_known_gate=True, kappa_cond=False):
        super().__init__()
        self.smooth = KernelHypernet(hidden, n_layers, feat_scale, kappa_cond=kappa_cond)
        self.log_gate = nn.Parameter(torch.zeros(())) if learn_known_gate else None

    def _gate(self):
        return (1.0 + torch.tanh(self.log_gate)) if self.log_gate is not None else 1.0

    def assemble(self, Xb, Nb, wb, curv, jump, kappa=None):
        k_known = double_layer_kernel(Xb, Xb, Nb, curv_s=curv, self_kernel=True)
        k = self._gate() * k_known + self.smooth(Xb, Xb, Nb, Nb, kappa)
        D0 = k * wb[:, None, :]                              # (B,n_b,n_b) arc-weighted
        B, n = Xb.shape[0], Xb.shape[1]
        A = jump * torch.eye(n, device=Xb.device).unsqueeze(0) + D0
        return A

    def potential(self, Xq, Xb, Nb, wb, kappa=None):
        """(B,Q,n_b) off-surface evaluation matrix: u(x_q) = sum_j P[q,j] mu_j."""
        # query points have no normal; use zero normal in the smooth features (off-surface, regular)
        Nq = torch.zeros(Xq.shape[0], Xq.shape[1], 2, device=Xq.device)
        k_known = double_layer_kernel(Xq, Xb, Nb)
        k = self._gate() * k_known + self.smooth(Xq, Xb, Nq, Nb, kappa)
        return k * wb[:, None, :]


class FixedPointSolve(nn.Module):
    """Differentiable solve of A mu = g, A = jump*I + D0.  mode='direct' (torch.linalg.solve, exact) or
    'neumann' (the CORRECT 2nd-kind fixed point  mu <- (g - D0 mu)/jump, which converges since rho(D0)<1/2
    for the smooth double layer -- a contraction where it is mathematically correct, vs the chaotic-rollout
    arc where contraction was wrong)."""

    def __init__(self, mode="direct", n_iter=20, omega=0.5):
        super().__init__()
        self.mode = mode; self.n_iter = n_iter; self.omega = omega

    def forward(self, A, g, jump):
        if self.mode == "direct":
            return torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)
        # damped 2nd-kind fixed point: mu <- (1-w) mu + w (g - D0 mu)/jump.  Damping (w<1) tames the
        # constant (-1/2) eigenmode of the interior double layer that makes the undamped iteration oscillate.
        n = A.shape[-1]
        D0 = A - jump * torch.eye(n, device=A.device).unsqueeze(0)
        w = self.omega
        mu = g / jump
        for _ in range(self.n_iter):
            mu = (1 - w) * mu + w * (g - torch.bmm(D0, mu.unsqueeze(-1)).squeeze(-1)) / jump
        return mu
