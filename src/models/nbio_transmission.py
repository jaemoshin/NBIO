"""NBIO-Transmission: multi-region Neural Boundary Integral Operator for piecewise-constant conductivity.

Mirrors the validated indirect single-layer BEM, but every kernel block = KNOWN operator (so the model
INITIALIZES at the classical multi-region solver) + a learned SMOOTH correction (M1). The interface
coefficient lam_j = -1/2 (a0+a_j)/(a0-a_j) is the KNOWN physics (contrast enters the assembly exactly).
Block system assembled + solved differentiably (M2); solution = single-layer potential at query points (M3),
continuous in geometry / #inclusions / boundary resolution.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import KernelHypernet
from .tm_layers import known_single_layer, known_adjoint_dlp
from ..numerics.qbx_torch import _upsample, arc_and_normal, single_layer_qbx_multi


def _speed(wb, n):
    return wb * n / (2 * math.pi)


class NBIOTransmission(nn.Module):
    def __init__(self, hidden=64, n_layers=3, feat_scale=1.0, split=True, tikhonov=0.0, init_scale=0.0,
                 qbx=False, qbx_order=12, qbx_r_fac=4.0, qbx_near_fac=2.0, assembly_oversample=1):
        super().__init__()
        self.split = split
        self.tikhonov = tikhonov
        # zero-init corrections -> NBIO-tm starts exactly at the classical multi-region solver (cond~600)
        self.smooth_S = KernelHypernet(hidden, n_layers, feat_scale, init_scale=init_scale)
        self.smooth_K = KernelHypernet(hidden, n_layers, feat_scale, init_scale=init_scale)
        # close-eval (near-touching): QBX evaluation of the known single layer + oversampled near-singular
        # CROSS-block assembly (the learned smooth correction is unchanged -> training is preserved).
        self.qbx = qbx; self.qbx_kw = dict(order=qbx_order, r_fac=qbx_r_fac, near_fac=qbx_near_fac)
        self.assembly_oversample = assembly_oversample   # int, or "adaptive" (gap-aware)
        self.adaptive_k = 6.0; self.adaptive_max = 24
        self._interp_cache = {}

    def _interp(self, S, ov, device):
        key = (S, ov, str(device))
        if key not in self._interp_cache:
            self._interp_cache[key] = _upsample(torch.eye(S, device=device), ov).T  # (S_fine, S)
        return self._interp_cache[key]

    def _adaptive_ov(self, ct, cs):
        """Gap-aware refinement: choose oversample so the fine source spacing < gap/k for the (ct,cs) pair.
        ov ~ k * mean_spacing / min_distance, clipped. Localizes refinement to near-touching pairs."""
        dmin = torch.cdist(ct["X"], cs["X"]).amin(dim=(1, 2)).clamp_min(1e-9).min()
        h = cs["w"].mean()
        ov = int(torch.ceil(self.adaptive_k * h / dmin).item())
        return max(4, min(ov, self.adaptive_max))

    def _known_cross_oversampled(self, kind, ct, cs, ov):
        """Oversample the SOURCE curve quadrature for a near-singular cross block; unknowns stay coarse via
        a Fourier coarse->fine interpolation P. self-blocks (Kress/curvature) are already accurate."""
        if ov == "adaptive":
            ov = self._adaptive_ov(ct, cs)
        Xf = _upsample(cs["X"], ov); wf, Nf = arc_and_normal(Xf); Sf = Xf.shape[1]; speed_f = _speed(wf, Sf)
        if kind == "S":
            Kf = known_single_layer(ct["X"], Xf, speed_f, selfblock=False)
        else:
            Kf = known_adjoint_dlp(ct["X"], ct["N"], ct["curv"], Xf, speed_f, selfblock=False)
        P = self._interp(cs["X"].shape[1], ov, ct["X"].device)
        return torch.einsum("bts,sj->btj", Kf, P)

    def _curves(self, b):
        """List of dicts per curve: [outer, incl_0, ...]. Recovers speed from arc weights."""
        nbo = b["Xb_out"].shape[1]
        curves = [{"X": b["Xb_out"], "N": b["Nb_out"], "w": b["wb_out"], "curv": b["curv_out"],
                   "speed": _speed(b["wb_out"], nbo)}]
        Ninc = b["a_in"].shape[1]; nbi = b["Xb_in"].shape[2]
        for k in range(Ninc):
            curves.append({"X": b["Xb_in"][:, k], "N": b["Nb_in"][:, k], "w": b["wb_in"][:, k],
                           "curv": b["curv_in"][:, k], "speed": _speed(b["wb_in"][:, k], nbi)})
        return curves

    def _S_block(self, ct, cs, selfblock):
        ov = self.assembly_oversample
        if selfblock or ov == 1:
            K = known_single_layer(ct["X"], cs["X"], cs["speed"], selfblock)
        else:
            K = self._known_cross_oversampled("S", ct, cs, ov)
        if self.split:
            Ns = cs["N"]; Nt = ct.get("N", torch.zeros_like(ct["X"]))
            K = K + self.smooth_S(ct["X"], cs["X"], Nt, Ns) * cs["w"][:, None, :]
        return K

    def _K_block(self, ct, cs, selfblock):
        ov = self.assembly_oversample
        if selfblock or ov == 1:
            K = known_adjoint_dlp(ct["X"], ct["N"], ct["curv"], cs["X"], cs["speed"], selfblock)
        else:
            K = self._known_cross_oversampled("K", ct, cs, ov)
        if self.split:
            K = K + self.smooth_K(ct["X"], cs["X"], ct["N"], cs["N"]) * cs["w"][:, None, :]
        return K

    def forward(self, b):
        curves = self._curves(b)
        Bsz = b["g"].shape[0]; Ncur = len(curves)
        sizes = [c["X"].shape[1] for c in curves]
        off = [0]
        for s in sizes:
            off.append(off[-1] + s)
        M = off[-1]
        A = torch.zeros(Bsz, M, M, device=b["g"].device)
        rhs = torch.zeros(Bsz, M, device=b["g"].device)
        # row 0: outer Dirichlet  sum_c S_{0<-c} phi_c = g
        for c in range(Ncur):
            A[:, off[0]:off[1], off[c]:off[c + 1]] = self._S_block(curves[0], curves[c], c == 0)
        rhs[:, off[0]:off[1]] = b["g"]
        # rows j>=1: flux NP  [lam_j I + K*_jj] phi_j + sum_{c!=j} Kn_{j<-c} phi_c = 0
        a0 = 1.0
        for j in range(1, Ncur):
            aj = b["a_in"][:, j - 1]                                  # (B,)
            lam = -0.5 * (a0 + aj) / (a0 - aj)                       # (B,)
            for c in range(Ncur):
                blk = self._K_block(curves[j], curves[c], c == j)
                if c == j:
                    eye = torch.eye(sizes[j], device=b["g"].device)
                    blk = blk + lam[:, None, None] * eye[None]
                A[:, off[j]:off[j + 1], off[c]:off[c + 1]] = blk
        if self.tikhonov > 0:
            A = A + self.tikhonov * torch.eye(M, device=A.device)[None]
        phi = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)   # (B,M)
        dens = [phi[:, off[c]:off[c + 1]] for c in range(Ncur)]
        # evaluate single-layer potential at queries
        Xq = b["Xq"]; Nq = torch.zeros(Bsz, Xq.shape[1], 2, device=Xq.device)
        if self.qbx:
            # known single layer via differentiable multi-curve QBX (close-eval) + learned correction direct
            u = single_layer_qbx_multi(Xq, [c["X"] for c in curves], dens, [c["N"] for c in curves], **self.qbx_kw)
            if self.split:
                for c in range(Ncur):
                    cs = curves[c]
                    Psm = self.smooth_S(Xq, cs["X"], Nq, cs["N"]) * cs["w"][:, None, :]
                    u = u + torch.bmm(Psm, dens[c].unsqueeze(-1)).squeeze(-1)
            return u
        u = torch.zeros(Bsz, Xq.shape[1], device=Xq.device)
        for c in range(Ncur):
            cs = curves[c]
            P = known_single_layer(Xq, cs["X"], cs["speed"], selfblock=False)
            if self.split:
                P = P + self.smooth_S(Xq, cs["X"], Nq, cs["N"]) * cs["w"][:, None, :]
            u = u + torch.bmm(P, dens[c].unsqueeze(-1)).squeeze(-1)
        return u

    @torch.no_grad()
    def cond_number(self, b):
        curves = self._curves(b); sizes = [c["X"].shape[1] for c in curves]
        off = [0]
        for s in sizes:
            off.append(off[-1] + s)
        M = off[-1]; A = torch.zeros(b["g"].shape[0], M, M, device=b["g"].device)
        for c in range(len(curves)):
            A[:, off[0]:off[1], off[c]:off[c + 1]] = self._S_block(curves[0], curves[c], c == 0)
        for j in range(1, len(curves)):
            aj = b["a_in"][:, j - 1]; lam = -0.5 * (1.0 + aj) / (1.0 - aj)
            for c in range(len(curves)):
                blk = self._K_block(curves[j], curves[c], c == j)
                if c == j:
                    blk = blk + lam[:, None, None] * torch.eye(sizes[j], device=A.device)[None]
                A[:, off[j]:off[j + 1], off[c]:off[c + 1]] = blk
        return torch.linalg.cond(A).mean().item()
