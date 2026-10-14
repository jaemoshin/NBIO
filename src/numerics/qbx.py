"""Quadrature-By-Expansion (QBX) for close/near-singular evaluation of 2D Laplace single-layer potentials.

The near-touching failure is a CLOSE-EVALUATION problem: u(x)=-1/2pi int ln|x-zeta| phi ds is near-singular
when x is within ~a panel of the boundary, so the smooth (trapezoidal) quadrature loses all accuracy. QBX
forms a local expansion of the (harmonic) potential about a center z0 pushed off the surface by r_c (where
the smooth quadrature IS accurate), then evaluates the expansion at x. Complex formulation: u = Re F,
F(z) = -1/2pi int ln(z-zeta) phi ds is analytic off Gamma, with Taylor coefficients
  c_0 = -1/2pi sum_j w_j phi_j ln(z0 - zeta_j),
  c_n = -1/2pi ((-1)^(n-1)/n) sum_j w_j phi_j / (z0 - zeta_j)^n   (n>=1),
and u(x) = Re[ sum_{n=0}^p c_n (x - z0)^n ],  valid for |x-z0| < dist(z0, Gamma). Re is branch-independent.

Optional Fourier OVERSAMPLING of (geometry, density) to a finer boundary makes the coefficient integrals
accurate at smaller r_c -> needed for thin gaps. Targets far from Gamma fall back to direct quadrature.
"""
from __future__ import annotations

import numpy as np


def _fourier_upsample(arr, factor):
    """Periodic band-limited upsample along axis 0 (n -> n*factor) via rfft zero-pad."""
    n = arr.shape[0]; m = n * factor
    A = np.fft.rfft(arr, axis=0)
    out = np.zeros((m // 2 + 1,) + arr.shape[1:], dtype=complex)
    keep = min(A.shape[0], m // 2 + 1)
    out[:keep] = A[:keep]
    return np.fft.irfft(out, n=m, axis=0) * (m / n)


def oversample_curve(Xb, phi, factor):
    """Upsample boundary points + density; recompute arc weights from the upsampled geometry."""
    if factor == 1:
        return Xb, phi
    Xf = _fourier_upsample(Xb, factor)
    pf = _fourier_upsample(phi[:, None], factor)[:, 0]
    return Xf, pf


def _arc_weights(Xb):
    """Trapezoidal arc weights w_j = (2pi/n)*|x'(theta_j)| from periodic spectral derivative."""
    n = Xb.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    dX = np.fft.ifft(1j * k[:, None] * np.fft.fft(Xb, axis=0), axis=0).real
    speed = np.hypot(dX[:, 0], dX[:, 1])
    return (2 * np.pi / n) * speed


def _infer_normal(Xb):
    n = Xb.shape[0]; k = np.fft.fftfreq(n, d=1.0 / n)
    dX = np.fft.ifft(1j * k[:, None] * np.fft.fft(Xb, axis=0), axis=0).real
    sp = np.hypot(dX[:, 0], dX[:, 1])
    return np.stack([dX[:, 1], -dX[:, 0]], 1) / sp[:, None]


def multi_single_layer_qbx(Xq, curves_X, curves_phi, curves_N=None, order=12, r_fac=4.0,
                           near_fac=2.0, oversample=4):
    """QBX for the TOTAL single-layer potential of several curves, u(x)=sum_c SL_c[phi_c](x). Center is
    placed off the nearest curve; the local-expansion coefficients integrate over ALL curves; the
    convergence guard uses distance to ALL curves. (Needed for transmission: multi-region single-layer.)"""
    z = Xq[:, 0] + 1j * Xq[:, 1]
    zf, wf, pf = [], [], []
    for Xc, pc in zip(curves_X, curves_phi):
        Xfc, pfc = oversample_curve(Xc, pc, oversample)
        zf.append(Xfc[:, 0] + 1j * Xfc[:, 1]); wf.append(_arc_weights(Xfc)); pf.append(pfc)
    zf = np.concatenate(zf); wf = np.concatenate(wf); pf = np.concatenate(pf)
    zb, nbc, Nre, wbb = [], [], [], []
    for ci, Xc in enumerate(curves_X):
        Nc = curves_N[ci] if curves_N is not None else _infer_normal(Xc)
        zb.append(Xc[:, 0] + 1j * Xc[:, 1]); nbc.append(Nc[:, 0] + 1j * Nc[:, 1])
        Nre.append(Nc); wbb.append(_arc_weights(Xc))
    zb = np.concatenate(zb); nbc = np.concatenate(nbc); Nre = np.concatenate(Nre); wbb = np.concatenate(wbb)

    d = np.abs(z[:, None] - zf[None, :]).clip(1e-14)
    u = (-1.0 / (2 * np.pi)) * (np.log(d) * wf[None, :]) @ pf
    D = np.abs(z[:, None] - zb[None, :]); jmin = D.argmin(1); dmin = D[np.arange(len(z)), jmin]
    rc = r_fac * wbb[jmin]; near = dmin < near_fac * rc
    if not near.any():
        return u
    zt = z[near]; jm = jmin[near]
    diff = zt - zb[jm]
    side = np.sign(diff.real * Nre[jm, 0] + diff.imag * Nre[jm, 1]); side[side == 0] = 1.0
    z0 = zb[jm] + side * rc[near] * nbc[jm]
    inv = 1.0 / (z0[:, None] - zf[None, :])
    val = ((-1.0 / (2 * np.pi)) * (np.log(z0[:, None] - zf[None, :]) * wf[None, :]) @ pf).astype(complex)
    dz = zt - z0; powinv = inv.copy()
    for nn in range(1, order + 1):
        cn = (-1.0 / (2 * np.pi)) * (((-1.0) ** (nn - 1)) / nn) * (powinv * wf[None, :]) @ pf
        val = val + cn * dz ** nn; powinv = powinv * inv
    distz0 = np.abs(z0[:, None] - zf[None, :]).min(1)
    ok = np.abs(dz) < distz0
    out = u[near].copy(); out[ok] = val.real[ok]; u[near] = out
    return u


def single_layer_qbx(Xq, Xb, phi, Nb=None, order=12, r_fac=4.0, near_fac=2.0, oversample=4):
    """Evaluate u(x)=-1/2pi int ln|x-zeta| phi ds at targets Xq (Q,2). QBX for near targets, direct else.

    Xb (S,2) boundary nodes (one closed curve), phi (S,) density, Nb (S,2) outward normals (optional;
    inferred if None). Returns u (Q,)."""
    z = Xq[:, 0] + 1j * Xq[:, 1]
    # fine quadrature for coefficient integrals
    Xf, pf = oversample_curve(Xb, phi, oversample)
    wf = _arc_weights(Xf)
    zf = Xf[:, 0] + 1j * Xf[:, 1]
    # base (model-resolution) geometry for center placement + arc spacing
    zb = Xb[:, 0] + 1j * Xb[:, 1]
    wb = _arc_weights(Xb)
    if Nb is None:
        n = Xb.shape[0]; k = np.fft.fftfreq(n, d=1.0 / n)
        dX = np.fft.ifft(1j * k[:, None] * np.fft.fft(Xb, axis=0), axis=0).real
        sp = np.hypot(dX[:, 0], dX[:, 1])
        Nb = np.stack([dX[:, 1], -dX[:, 0]], 1) / sp[:, None]      # outward for CCW
    nb_c = Nb[:, 0] + 1j * Nb[:, 1]

    def direct(zt, zsrc, w, p):
        d = np.abs(zt[:, None] - zsrc[None, :]).clip(1e-14)
        return (-1.0 / (2 * np.pi)) * (np.log(d) * w[None, :]) @ p
    u = direct(z, zf, wf, pf)                                     # accurate for far targets

    D = np.abs(z[:, None] - zb[None, :])
    jmin = D.argmin(1); dmin = D[np.arange(len(z)), jmin]
    hloc = wb[jmin]
    rc = r_fac * hloc
    near = dmin < near_fac * rc
    if not near.any():
        return u
    zt = z[near]; jm = jmin[near]
    diff = zt - zb[jm]
    side = np.sign(diff.real * Nb[jm, 0] + diff.imag * Nb[jm, 1]); side[side == 0] = 1.0
    z0 = zb[jm] + side * rc[near] * nb_c[jm]                      # center off-surface, on the target's side
    inv = 1.0 / (z0[:, None] - zf[None, :])                       # (Qn, Sfine), smooth (z0 off Gamma)
    c0 = (-1.0 / (2 * np.pi)) * (np.log(z0[:, None] - zf[None, :]) * wf[None, :]) @ pf
    val = c0.astype(complex)
    dz = zt - z0
    powinv = inv.copy()                                          # inv^n
    for nn in range(1, order + 1):
        cn = (-1.0 / (2 * np.pi)) * (((-1.0) ** (nn - 1)) / nn) * (powinv * wf[None, :]) @ pf
        val = val + cn * dz ** nn
        powinv = powinv * inv
    u_near = val.real
    # only trust QBX where the expansion converges: |x-z0| < dist(z0, Gamma)
    distz0 = np.abs(z0[:, None] - zf[None, :]).min(1)
    ok = np.abs(dz) < distz0
    out_near = u[near].copy()
    out_near[ok] = u_near[ok]
    u[near] = out_near
    return u
