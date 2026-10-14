"""Differentiable, batched QBX (torch) for the known double-layer potential -- so close evaluation is part
of the NBIO forward pass, not a numpy post-process. The learned SMOOTH kernel correction has no singularity
and is evaluated directly; QBX is applied ONLY to the known singular double layer. This keeps all neural-
operator properties (learned correction M1, differentiable solve M2, coordinate-free zero-shot M3) intact.

Complex (Cauchy) form: D[mu](z) = Re F(z),  F(z) = -1/2pi sum_j mu_j n_j w_j / (zeta_j - z),  analytic off
Gamma. Local expansion about a center z0 pushed off Gamma by r_c ~ r_fac*h:
  c_n = -1/2pi sum_j mu_j n_j w_j / (zeta_j - z0)^(n+1),   D[mu](z) = Re sum_{n<=p} c_n (z - z0)^n,
valid for |z-z0| < dist(z0, Gamma). All ops are torch-differentiable wrt mu (and geometry); the nearest-node
argmin is a (sub)gradient-safe selection, exactly like max-pooling.
"""
from __future__ import annotations

import math

import torch


def arc_and_normal(Xb):
    """Xb (B,S,2) -> arc weights w (B,S) and outward unit normal (B,S,2), via periodic spectral derivative."""
    B, S, _ = Xb.shape
    k = torch.fft.fftfreq(S, d=1.0 / S, device=Xb.device)
    dX = torch.fft.ifft(1j * k[None, :, None] * torch.fft.fft(Xb, dim=1), dim=1).real
    speed = torch.linalg.norm(dX, dim=2).clamp_min(1e-12)
    normal = torch.stack([dX[..., 1], -dX[..., 0]], dim=2) / speed[..., None]
    w = (2 * math.pi / S) * speed
    return w, normal


def _upsample(arr, factor):
    """Band-limited periodic upsample along dim=1 (S -> S*factor), differentiable. arr (B,S) or (B,S,C)."""
    if factor == 1:
        return arr
    S = arr.shape[1]; m = S * factor
    A = torch.fft.rfft(arr, dim=1)
    shape = list(A.shape); shape[1] = m // 2 + 1
    out = torch.zeros(shape, dtype=A.dtype, device=arr.device)
    keep = min(A.shape[1], m // 2 + 1)
    out[:, :keep] = A[:, :keep]
    return torch.fft.irfft(out, n=m, dim=1) * (m / S)


def single_layer_qbx_multi(Xq, Xs_list, phi_list, Ns_list, order=12, r_fac=4.0, near_fac=2.0, oversample=4):
    """Differentiable torch QBX for the TOTAL single-layer potential of several curves (transmission):
    u(x) = sum_c -1/2pi int ln|x-zeta| phi_c ds. Center off nearest curve; coeffs integrate over ALL curves
    (Fourier-oversampled); guard vs all curves. Xq (B,Q,2); Xs_list[c],Ns_list[c] (B,Sc,2); phi_list[c] (B,Sc)."""
    Xb_all = torch.cat(Xs_list, dim=1); N_all = torch.cat(Ns_list, dim=1)
    wbb = torch.cat([arc_and_normal(X)[0] for X in Xs_list], dim=1)
    zf_l, wf_l, pf_l = [], [], []
    for X, p in zip(Xs_list, phi_list):
        Xf = _upsample(X, oversample); pf = _upsample(p, oversample); wf, _ = arc_and_normal(Xf)
        zf_l.append(torch.complex(Xf[..., 0], Xf[..., 1])); wf_l.append(wf); pf_l.append(pf)
    zf = torch.cat(zf_l, 1); wpf = torch.cat(wf_l, 1) * torch.cat(pf_l, 1)
    z = torch.complex(Xq[..., 0], Xq[..., 1])
    dvec = z[:, :, None] - zf[:, None, :]
    u = (-1.0 / (2 * math.pi)) * (torch.log(dvec.abs().clamp_min(1e-14)) * wpf[:, None, :]).sum(-1)
    zb = torch.complex(Xb_all[..., 0], Xb_all[..., 1])
    dmin, jmin = (z[:, :, None] - zb[:, None, :]).abs().min(-1)
    h = torch.gather(wbb, 1, jmin); rc = r_fac * h; near = dmin < near_fac * rc
    Nx = torch.gather(N_all[..., 0], 1, jmin); Ny = torch.gather(N_all[..., 1], 1, jmin)
    zn = torch.gather(Xb_all[..., 0], 1, jmin) + 1j * torch.gather(Xb_all[..., 1], 1, jmin)
    diff = z - zn
    side = torch.sign(diff.real * Nx + diff.imag * Ny); side = torch.where(side == 0, torch.ones_like(side), side)
    z0 = zn + side * rc * (Nx + 1j * Ny)
    diffc = z0[:, :, None] - zf[:, None, :]
    val = (-1.0 / (2 * math.pi)) * (torch.log(diffc) * wpf[:, None, :]).sum(-1)
    dz = z - z0; invd = 1.0 / diffc; powinv = invd.clone()
    for n in range(1, order + 1):
        cn = (-1.0 / (2 * math.pi)) * (((-1.0) ** (n - 1)) / n) * (powinv * wpf[:, None, :]).sum(-1)
        val = val + cn * dz ** n; powinv = powinv * invd
    distz0 = diffc.abs().min(-1).values
    use = near & (dz.abs() < distz0)
    return torch.where(use, val.real, u)


def double_layer_qbx(Xq, Xb, mu, Nb=None, wb=None, order=10, r_fac=4.0, near_fac=2.0, oversample=4):
    """Evaluate D[mu](x) at Xq (B,Q,2) from boundary Xb (B,S,2), density mu (B,S). QBX for near targets,
    direct for far; Fourier oversampling makes coefficient integrals accurate at the needed radius."""
    # center placement uses base-resolution geometry; coefficient/far integrals use the oversampled curve
    if wb is None or Nb is None:
        wb, Nb = arc_and_normal(Xb)
    Xf = _upsample(Xb, oversample)
    muf = _upsample(mu, oversample)
    wf, Nf = arc_and_normal(Xf)
    z = torch.complex(Xq[..., 0], Xq[..., 1])                       # (B,Q)
    zf = torch.complex(Xf[..., 0], Xf[..., 1])                      # (B,Sf)
    nf = torch.complex(Nf[..., 0], Nf[..., 1])
    af = (-1.0 / (2 * math.pi)) * muf * nf * wf                     # (B,Sf)
    denom = zf[:, None, :] - z[:, :, None]                          # (B,Q,Sf)
    u_direct = (af[:, None, :] / torch.where(denom.abs() < 1e-12, denom + 1e-12, denom)).sum(-1).real

    zb = torch.complex(Xb[..., 0], Xb[..., 1])
    distb = (zb[:, None, :] - z[:, :, None]).abs()
    dmin, jmin = distb.min(-1)                                      # nearest BASE node
    h = torch.gather(wb, 1, jmin); rc = r_fac * h
    near = dmin < near_fac * rc
    zn = torch.gather(Xb[..., 0], 1, jmin) + 1j * torch.gather(Xb[..., 1], 1, jmin)
    Nx = torch.gather(Nb[..., 0], 1, jmin); Ny = torch.gather(Nb[..., 1], 1, jmin)
    nn = Nx + 1j * Ny
    diff = z - zn
    side = torch.sign(diff.real * Nx + diff.imag * Ny)
    side = torch.where(side == 0, torch.ones_like(side), side)
    z0 = zn + side * rc * nn                                        # (B,Q) center
    invd = 1.0 / (zf[:, None, :] - z0[:, :, None])                  # (B,Q,Sf)
    dz = z - z0
    Fz = torch.zeros_like(z)
    powinv = invd.clone(); powdz = torch.ones_like(z)
    for _ in range(order + 1):
        cn = (af[:, None, :] * powinv).sum(-1)
        Fz = Fz + cn * powdz
        powinv = powinv * invd; powdz = powdz * dz
    distz0 = (zf[:, None, :] - z0[:, :, None]).abs().min(-1).values
    use = near & (dz.abs() < distz0)
    return torch.where(use, Fz.real, u_direct)
