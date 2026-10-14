"""Stage B: inhomogeneous elliptic interface problem with a VOLUME SOURCE.

  -div(a grad u) = f  in each region,  outer Dirichlet u=g,  interface jumps [u]=alpha, [a u_n]=beta.

Split u = u_p + u_h:
  u_p = N[f]/a_k in region k, where N solves -Delta N = f (Newtonian potential, here a non-singular FFT
        periodic Poisson solve on a box containing Omega). This gives [u_p]=N(1/a+ -1/a-), [a u_p,n]=0.
  u_h = homogeneous piecewise-harmonic interface problem (Stage A) with data
        g_h = g - N|Gamma0 / a+,  [u_h] = alpha - N|Gamma1 (1/a+ - 1/a-),  [a u_h,n] = beta.
Then u = N/a_k + u_h.  The harmonic ambiguity in N (periodic vs free-space) cancels because the SAME N is
used for both the data modification and the reconstruction. No singular volume quadrature is needed.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .interface import interface_solve
from .transmission import build_curves


def gaussians(rng, n=5, box=0.85):
    """Gaussian sources kept tight inside a large FFT box so f is periodic-compatible (mean~0).
    Returns (c, A, s) — same tuple signature as callers expect."""
    c = rng.uniform(-box, box, (n, 2))
    A = rng.normal(size=n)
    s = rng.uniform(0.18, 0.32, n)
    return c, A, s


def phi_and_f(X, c, A, s):
    """phi = sum A exp(-|x-c|^2/2s^2);  f = -Laplace(phi) = -sum A exp(..)(r^2-2s^2)/s^4."""
    diff = X[:, None, :] - c[None, :, :]; r2 = (diff ** 2).sum(-1)
    g = A[None, :] * np.exp(-r2 / (2 * s[None, :] ** 2))
    return g.sum(1), -(g * ((r2 - 2 * s[None, :] ** 2) / s[None, :] ** 4)).sum(1)


def volume_potential_fft(f_func, L=2.8, n=1024):
    """Periodic FFT Poisson solve  -Delta N = f - mean(f)  on [-L,L]^2;
    returns a BICUBIC interpolator N(x) giving O(h^4) interpolation error.
    n=1024 (default) resolves the C-inf bump Laplacian to ~1e-6 std accuracy."""
    xs = np.linspace(-L, L, n, endpoint=False); h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    F = f_func(np.stack([X.ravel(), Y.ravel()], 1)).reshape(n, n)
    F = F - F.mean()
    k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    KX, KY = np.meshgrid(k, k, indexing="ij"); k2 = KX ** 2 + KY ** 2; k2[0, 0] = 1.0
    Nhat = np.fft.fft2(F) / k2; Nhat[0, 0] = 0.0
    N = np.fft.ifft2(Nhat).real

    def interp(pts):
        # Convert physical coords to fractional grid indices; map_coordinates handles wrapping.
        gx = (pts[:, 0] + L) / h   # fractional index along axis-0
        gy = (pts[:, 1] + L) / h   # fractional index along axis-1
        coords = np.array([gx, gy])
        return map_coordinates(N, coords, order=3, mode='wrap')

    return interp


def inhomogeneous_interface_solve(c_out, incl, a_plus, a_minus, f_func, g, alpha, beta,
                                  nbo, nbi, L=2.8, ngrid=1024):
    """Solve the inhomogeneous interface problem; returns evaluate(Xq, region) in {'plus','minus'}."""
    N = volume_potential_fft(f_func, L, ngrid)
    curves = build_curves(c_out, [incl], nbo, nbi)
    G0, G1 = curves[0], curves[1]
    g_h = g - N(G0["X"]) / a_plus
    alpha_h = alpha - N(G1["X"]) * (1.0 / a_plus - 1.0 / a_minus)
    beta_h = beta
    uh_eval, _ = interface_solve(c_out, incl, a_plus, a_minus, g_h, alpha_h, beta_h, nbo, nbi)

    def evaluate(Xq, region):
        ak = a_minus if region == "minus" else a_plus
        return N(Xq) / ak + uh_eval(Xq, region)
    return evaluate
