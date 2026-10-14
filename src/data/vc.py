"""Stage 2: variable PRINCIPAL-coefficient elliptic PDE on the torus,  div(a(x) grad u) = f,  a>0 smooth.

This is the regime where NO closed-form Green's function exists (a multiplies the top derivative), so a
constant-coefficient (FFT-Laplacian) prior is only APPROXIMATE and the learned correction is NECESSARY --
the genuine test the boundary-integral Stages 0/1 (known kernel) and the Yukawa lower-order term could not give.

GT: variable-coefficient finite-difference (harmonic-mean face conductivities) sparse solve, mean-zero.
SELF-VALIDATED by the method of manufactured solutions (recover a known u* to O(h^2)).
Constant-coefficient prior u_lap = (-a_bar Lap)^-1 f via FFT (the 'wrong fixed kernel' baseline / prior).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def smooth_field(rng, n, n_modes=4, decay=1.0):
    """Random smooth periodic field on an n x n torus grid (low-mode Fourier, real)."""
    fh = np.zeros((n, n), dtype=complex)
    for kx in range(-n_modes, n_modes + 1):
        for ky in range(-n_modes, n_modes + 1):
            if kx == 0 and ky == 0:
                continue
            amp = rng.normal() + 1j * rng.normal()
            fh[kx % n, ky % n] += amp / (1 + (kx ** 2 + ky ** 2) ** (0.5 * (1 + decay)))
    f = np.fft.ifft2(fh).real
    return (f - f.mean()) / (f.std() + 1e-9)


def sample_a(rng, n, contrast=3.0, n_modes=4):
    """Positive coefficient a = exp(s * field), contrast = max/min ~ exp(2s*||.||). Returns (n,n)."""
    g = smooth_field(rng, n, n_modes)
    s = 0.5 * np.log(contrast)
    a = np.exp(s * g / (np.abs(g).max() + 1e-9))
    return a


def _faces(a):
    """Harmonic-mean conductivities on the +x and +y faces (periodic)."""
    ax = 2 * a * np.roll(a, -1, axis=0) / (a + np.roll(a, -1, axis=0))   # face between i and i+1 (x)
    ay = 2 * a * np.roll(a, -1, axis=1) / (a + np.roll(a, -1, axis=1))
    return ax, ay


def build_operator(a, h):
    """Sparse div(a grad .) on periodic n x n grid (negative-definite); returns -div(a grad) (SPD-ish)."""
    n = a.shape[0]
    ax, ay = _faces(a)                       # ax[i,j]=cond on face (i->i+1); ay similarly in y
    axm = np.roll(ax, 1, axis=0)             # face (i-1 -> i)
    aym = np.roll(ay, 1, axis=1)
    idx = lambda i, j: ((i % n) * n + (j % n))
    I = np.arange(n)[:, None] * n + np.arange(n)[None, :]
    diag = (ax + axm + ay + aym).ravel() / h ** 2
    rows = [I.ravel(), I.ravel(), I.ravel(), I.ravel(), I.ravel()]
    cols = [I.ravel(),
            (np.roll(I, -1, 0)).ravel(), (np.roll(I, 1, 0)).ravel(),
            (np.roll(I, -1, 1)).ravel(), (np.roll(I, 1, 1)).ravel()]
    vals = [diag,
            -ax.ravel() / h ** 2, -axm.ravel() / h ** 2,
            -ay.ravel() / h ** 2, -aym.ravel() / h ** 2]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return A    # approximates -div(a grad)


def solve_vc(a, f, h=None, eps=1e-8):
    """Solve -div(a grad u) = f on the torus, mean-zero. (a,f) are (n,n); returns u (n,n)."""
    n = a.shape[0]; h = h or (2 * np.pi / n)
    A = build_operator(a, h) + eps * sp.identity(n * n)   # eps removes the constant null space
    u = spla.spsolve(A.tocsc(), f.ravel())
    u = u.reshape(n, n); return u - u.mean()


def const_prior(a, f, h=None):
    """u_lap = (-a_bar Laplacian)^-1 f via FFT (constant-coefficient 'wrong fixed kernel' prior)."""
    n = a.shape[0]; h = h or (2 * np.pi / n)
    abar = float(a.mean())
    k = np.fft.fftfreq(n, d=h / (2 * np.pi))            # integer wavenumbers
    KX, KY = np.meshgrid(k, k, indexing="ij")
    lap = (KX ** 2 + KY ** 2); lap[0, 0] = 1.0
    uh = np.fft.fft2(f) / (abar * lap); uh[0, 0] = 0.0
    return np.fft.ifft2(uh).real


def validate_fd(seed=0, n=128, contrast=3.0):
    """Manufactured: pick u*, a; set f=-div(a grad u*) spectrally; solve; compare. Returns rel-err (O(h^2))."""
    rng = np.random.default_rng(seed); h = 2 * np.pi / n
    a = sample_a(rng, n, contrast); us = smooth_field(rng, n, 3); us = us - us.mean()
    k = np.fft.fftfreq(n, d=h / (2 * np.pi)); KX, KY = np.meshgrid(k, k, indexing="ij")
    ush = np.fft.fft2(us)
    ux = np.fft.ifft2(1j * KX * ush).real; uy = np.fft.ifft2(1j * KY * ush).real
    fx = a * ux; fy = a * uy
    div = np.fft.ifft2(1j * KX * np.fft.fft2(fx) + 1j * KY * np.fft.fft2(fy)).real
    f = -div                                            # f = -div(a grad u*)
    u = solve_vc(a, f, h)
    return float(np.linalg.norm(u - us) / (np.linalg.norm(us) + 1e-12))


def make_sample(rng, n, contrast):
    a = sample_a(rng, n, contrast)
    f = smooth_field(rng, n, 4)                         # mean-zero source
    u = solve_vc(a, f)
    ulap = const_prior(a, f)
    return {"a": a.astype(np.float32), "f": f.astype(np.float32),
            "u": u.astype(np.float32), "ulap": ulap.astype(np.float32)}
