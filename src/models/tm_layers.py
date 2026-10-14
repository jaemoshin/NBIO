"""Batched, differentiable torch versions of the transmission BEM kernels (mirror the validated numpy in
src/data/transmission.py): single layer (Kress self / trapezoidal cross) and adjoint double layer K*."""
from __future__ import annotations

import math

import torch

INV2PI = 1.0 / (2.0 * math.pi)


def kress_weights(n: int, device) -> torch.Tensor:
    """R_d, d=0..n-1 (n even): weights for int ln(4 sin^2((t-tau)/2)) g dtau ~ sum_j R_{(i-j)%n} g_j."""
    M = n // 2
    d = torch.arange(n, device=device, dtype=torch.float64)
    td = 2 * math.pi * d / n
    m = torch.arange(1, M, device=device, dtype=torch.float64)
    R = -(2 * math.pi / M) * (torch.cos(td[:, None] * m[None, :]) / m).sum(1) - (math.pi / M ** 2) * torch.cos(M * td)
    return R.to(torch.float32)


def known_single_layer(Xt, Xs, speed_s, selfblock: bool):
    """(B,T,S) known single-layer matrix  -1/2pi ln|x-y| * w_y.  selfblock -> Kress, else trapezoidal."""
    ns = Xs.shape[1]
    if not selfblock:
        d = torch.cdist(Xt, Xs).clamp_min(1e-12)
        return (-INV2PI) * torch.log(d) * (2 * math.pi / ns) * speed_s[:, None, :]
    n = ns
    R = kress_weights(n, Xt.device)
    idx = (torch.arange(n, device=Xt.device)[:, None] - torch.arange(n, device=Xt.device)[None, :]) % n
    Rmat = R[idx]                                                   # (n,n)
    r = torch.cdist(Xt, Xs).clamp_min(1e-12)                        # (B,n,n)
    t = 2 * math.pi * torch.arange(n, device=Xt.device) / n
    sin_half = torch.abs(torch.sin((t[:, None] - t[None, :]) / 2)).clamp_min(1e-12)
    M2 = torch.log(r / (2 * sin_half[None]))
    diag = torch.log(speed_s).diag_embed()                         # (B,n,n) diagonal ln|x'(t)|
    M2 = M2 - torch.diagonal(M2, dim1=-2, dim2=-1).diag_embed() + diag
    return (-INV2PI) * (0.5 * Rmat[None] + (2 * math.pi / n) * M2) * speed_s[:, None, :]


def known_adjoint_dlp(Xt, Nt, curv_t, Xs, speed_s, selfblock: bool):
    """(B,T,S) known adjoint double layer  K*[i,j]=(1/2pi)(y-x).n_x/|y-x|^2 * w_y. self diag -kappa/4pi."""
    ns = Xs.shape[1]
    diff = Xs[:, None, :, :] - Xt[:, :, None, :]                    # y - x  (B,T,S,2)
    r2 = (diff ** 2).sum(-1).clamp_min(1e-12)
    num = (diff * Nt[:, :, None, :]).sum(-1)
    k = INV2PI * num / r2
    if selfblock:   # replace diagonal with the curvature limit -kappa/4pi
        cur_diag = torch.diagonal(k, dim1=-2, dim2=-1).diag_embed()
        k = k - cur_diag + (-curv_t / (4 * math.pi)).diag_embed()
    w = (2 * math.pi / ns) * speed_s
    return k * w[:, None, :]
