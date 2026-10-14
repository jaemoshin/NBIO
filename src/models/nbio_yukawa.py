"""Yukawa (modified-Helmholtz) NBIO -- a known-kernel NBIO whose analytic split uses the YUKAWA
double-layer Green's kernel at a FIXED screening parameter kappa, plus the learned smooth correction.

This is the natural companion to the Laplace NBIO (src/models/nbio.py): for the screened-Poisson /
modified-Helmholtz problem  (-Delta + kappa^2) u = 0  the closed-form double-layer kernel is

    G(x,y)        = (1/2pi) K_0(kappa r),                       r = |x-y|
    dG/dn_y       = -(kappa/2pi) K_1(kappa r) (y-x).n_y / r,    (the double layer)

with the SAME leading singularity as Laplace (K_1(z) ~ 1/z), hence the SAME curvature diagonal limit
-curv/(4pi).  Plugging this known kernel into the M1 split means the model INITIALIZES at the classical
Yukawa Nystrom solver (gate=1, correction=0) and only learns a small smooth correction -> solver grade,
exactly as the Laplace NBIO does for Laplace.  Everything else (M2 differentiable 2nd-kind solve, M3
coordinate-free off-surface potential, zero-shot geometry/resolution) is inherited unchanged from NBIO.

We torch-port scipy.special.kn(0/1, .) via the standard rational/asymptotic approximations (Abramowitz &
Stegun 9.8.x) so the whole forward pass stays differentiable and runs on GPU.  Accuracy of the approximation
is ~1e-7 relative over the kappa*r range we hit (kappa=4, domain radius ~1.5), well below the operator floor.

Created as a NEW file (no edits to existing modules / the model registry); build it directly via
`build_yukawa_nbio(...)` rather than the global factory.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import KernelHypernet, FixedPointSolve
from .nbio import NBIO


# --------------------------- torch modified Bessel K_0, K_1 ---------------------------
# Polynomial approximations (Abramowitz & Stegun 9.8.5-9.8.8); double precision coefficients.
def _bessel_k0(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-12)
    small = x <= 2.0
    # small-argument branch: K0(x) = -ln(x/2) I0(x) + poly((x/2)^2).
    # NOTE A&S use two different scalings: I0 (9.8.1) in s=(x/3.75)^2, the K0 series (9.8.5) in t=(x/2)^2.
    s = (x / 3.75) ** 2
    i0 = 1.0 + s * (3.5156229 + s * (3.0899424 + s * (1.2067492
         + s * (0.2659732 + s * (0.0360768 + s * 0.0045813)))))
    t = (x / 2.0) ** 2
    ks = (-torch.log(x / 2.0) * i0
          - 0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590
          + t * (0.00262698 + t * (0.00010750 + t * 0.00000740))))))
    # large-argument branch: K0(x) = exp(-x)/sqrt(x) * poly(2/x)
    u = 2.0 / x
    kl = (torch.exp(-x) / torch.sqrt(x)) * (1.25331414 + u * (-0.07832358
         + u * (0.02189568 + u * (-0.01062446 + u * (0.00587872
         + u * (-0.00251540 + u * 0.00053208))))))
    return torch.where(small, ks, kl)


def _bessel_k1(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-12)
    small = x <= 2.0
    # I1 (9.8.3) in s=(x/3.75)^2; the K1 series (9.8.7) -> x*K1(x) is a poly in t=(x/2)^2.
    s = (x / 3.75) ** 2
    i1 = x * (0.5 + s * (0.87890594 + s * (0.51498869 + s * (0.15084934
         + s * (0.02658733 + s * (0.00301532 + s * 0.00032411))))))
    t = (x / 2.0) ** 2
    xk1 = (x * torch.log(x / 2.0) * i1
           + 1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897
           + t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))))
    ks = xk1 / x
    u = 2.0 / x
    kl = (torch.exp(-x) / torch.sqrt(x)) * (1.25331414 + u * (0.23498619
         + u * (-0.03655620 + u * (0.01504268 + u * (-0.00780353
         + u * (0.00325614 + u * (-0.00068245)))))))
    return torch.where(small, ks, kl)


def yukawa_double_layer_kernel(Xt, Xs, Ns, kappa, curv_s=None, self_kernel=False, eps=1e-9):
    """Known Yukawa double-layer  k(x,y) = -(kappa/2pi) K_1(kappa r) (y-x).n_y / r.  (B,T,2),(B,S,2),(B,S,2).

    Diagonal (self_kernel) is the curvature limit -curv/(4pi) -- identical to Laplace because the Yukawa
    double layer has the same -1/(2pi) * (y-x).n_y/r^2 leading behaviour (K_1(z) ~ 1/z)."""
    diff = Xs[:, None, :, :] - Xt[:, :, None, :]            # (B,T,S,2) = y - x
    r = diff.norm(dim=-1).clamp_min(eps)                    # (B,T,S)
    num = (diff * Ns[:, None, :, :]).sum(-1)                # (y-x).n_y
    k = -(kappa / (2.0 * torch.pi)) * _bessel_k1(kappa * r) * num / r
    if self_kernel:
        S = Xs.shape[1]
        diag = (-curv_s / (4.0 * torch.pi)).to(k.dtype)     # (B,S)
        idx = torch.arange(S, device=k.device)
        k[:, idx, idx] = diag
    return k


class YukawaBIO(nn.Module):
    """K = (1+gate)*K_known_yukawa + K_smooth (hypernet on relative geometry); arc-weighted; 2nd-kind A."""

    def __init__(self, kappa, hidden=64, n_layers=3, feat_scale=1.0, learn_known_gate=True):
        super().__init__()
        self.kappa = float(kappa)
        self.smooth = KernelHypernet(hidden, n_layers, feat_scale)
        self.log_gate = nn.Parameter(torch.zeros(())) if learn_known_gate else None

    def _gate(self):
        return (1.0 + torch.tanh(self.log_gate)) if self.log_gate is not None else 1.0

    def assemble(self, Xb, Nb, wb, curv, jump):
        k_known = yukawa_double_layer_kernel(Xb, Xb, Nb, self.kappa, curv_s=curv, self_kernel=True)
        k = self._gate() * k_known + self.smooth(Xb, Xb, Nb, Nb)
        D0 = k * wb[:, None, :]
        n = Xb.shape[1]
        return jump * torch.eye(n, device=Xb.device).unsqueeze(0) + D0

    def potential(self, Xq, Xb, Nb, wb):
        Nq = torch.zeros(Xq.shape[0], Xq.shape[1], 2, device=Xq.device)
        k_known = yukawa_double_layer_kernel(Xq, Xb, Nb, self.kappa)
        k = self._gate() * k_known + self.smooth(Xq, Xb, Nq, Nb)
        return k * wb[:, None, :]


class NBIOYukawa(NBIO):
    """NBIO with the KNOWN Yukawa double-layer kernel at a fixed kappa. Same forward contract as NBIO
    (batch dict {Xb,Nb,wb,curv,g,Xq} -> u (B,Q)); split=True uses the analytic Yukawa kernel + correction."""

    def __init__(self, kappa, hidden=64, n_layers=3, feat_scale=1.0, region="interior",
                 split=True, solve="direct", n_iter=20, omega=1.0):
        # NBIO.__init__ builds a Laplace BIO + solver; we then swap in the Yukawa BIO.
        super().__init__(hidden=hidden, n_layers=n_layers, feat_scale=feat_scale, region=region,
                         split=split, solve=solve, n_iter=n_iter, omega=omega, qbx=False)
        self.kappa = float(kappa)
        self.bio = YukawaBIO(kappa, hidden, n_layers, feat_scale, learn_known_gate=split)
        self.solver = FixedPointSolve(solve, n_iter, omega)


def build_yukawa_nbio(kappa, **kwargs):
    return NBIOYukawa(kappa=kappa, **kwargs)
