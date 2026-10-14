"""2D Laplace boundary-value-problem data on a family of smooth star-shaped domains.

Ground truth is a MANUFACTURED harmonic solution (sum of point-charge potentials whose singularities
are outside the solution region) -> interior/exterior values are EXACT (no discretization error in GT).
A spectrally-accurate Nystrom BEM (double-layer, 2nd-kind) is provided as the classical gold-standard
baseline and is SELF-VALIDATED against the exact solution in `validate_bem`.

Geometry family (star-shaped, smooth):  r(theta) = r0 * (1 + sum_k a_k cos k.theta + b_k sin k.theta),
coefficients decaying in k so curves stay smooth & non-self-intersecting.

Task (operator learning across geometries):  (boundary geometry + Dirichlet data g) -> u at query points.
  interior problem: charges OUTSIDE Omega, query points INSIDE.
  exterior problem: charges INSIDE  Omega, query points OUTSIDE (unbounded domain; volumetric grids cannot).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import kn   # modified Bessel K_0, K_1 (Yukawa / modified-Helmholtz)


# ----------------------------- geometry -----------------------------
def sample_coeffs(rng: np.random.Generator, n_modes=5, decay=1.6, amp=0.32, r0=1.0):
    """Random star-shaped boundary r(theta)=r0*(1+sum a_k cos+b_k sin). Rescaled to stay star-shaped."""
    k = np.arange(1, n_modes + 1)
    scale = amp / k ** decay
    a = rng.normal(0, 1, n_modes) * scale
    b = rng.normal(0, 1, n_modes) * scale
    # keep 1+sum(...) bounded in (0.35, 1.65): rescale if the worst-case deviation is too large
    dev = np.sum(np.abs(a) + np.abs(b))
    if dev > 0.55:
        a *= 0.55 / dev; b *= 0.55 / dev
    return {"r0": float(r0), "a": a, "b": b}


def boundary(coeffs: Dict, n_b: int):
    """Return boundary samples (uniform in theta) with EXACT geometric quantities from the Fourier series.

    -> dict: theta, X (n_b,2), normal (n_b,2 outward), speed |x'(theta)| (n_b,), curvature (n_b,),
             w (n_b,) trapezoidal arc weights = (2pi/n_b)*speed.
    """
    r0 = coeffs["r0"]; a = coeffs["a"]; b = coeffs["b"]; K = len(a)
    th = np.linspace(0, 2 * np.pi, n_b, endpoint=False)
    k = np.arange(1, K + 1)[:, None]                      # (K,1)
    C = np.cos(k * th[None, :]); S = np.sin(k * th[None, :])
    r = r0 * (1 + a @ C + b @ S)                          # (n_b,)
    rp = r0 * ((-a * k[:, 0]) @ S + (b * k[:, 0]) @ C)    # r'(theta)
    rpp = r0 * ((-a * k[:, 0] ** 2) @ C + (-b * k[:, 0] ** 2) @ S)  # r''(theta)
    ct, st = np.cos(th), np.sin(th)
    X = np.stack([r * ct, r * st], axis=1)                # (n_b,2)
    xp = rp * ct - r * st; yp = rp * st + r * ct          # x'(theta), y'(theta)
    speed = np.hypot(xp, yp)
    normal = np.stack([yp, -xp], axis=1) / speed[:, None]  # outward for CCW
    curv = (r ** 2 + 2 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5   # signed curvature
    w = (2 * np.pi / n_b) * speed
    return {"theta": th, "X": X, "normal": normal, "speed": speed, "curv": curv, "w": w, "r": r}


def radius_at(coeffs: Dict, theta: np.ndarray):
    r0 = coeffs["r0"]; a = coeffs["a"]; b = coeffs["b"]
    k = np.arange(1, len(a) + 1)[:, None]
    return r0 * (1 + a @ np.cos(k * theta[None, :]) + b @ np.sin(k * theta[None, :]))


# ----------------------------- manufactured solution -----------------------------
def sample_charges(rng, region: str, coeffs: Dict, n_charges=6):
    """Point charges whose log-potentials are harmonic in the SOLUTION region.
    interior problem -> charges OUTSIDE Omega; exterior problem -> charges INSIDE Omega."""
    rmax = float(radius_at(coeffs, np.linspace(0, 2 * np.pi, 256)).max())
    rmin = float(radius_at(coeffs, np.linspace(0, 2 * np.pi, 256)).min())
    ang = rng.uniform(0, 2 * np.pi, n_charges)
    if region == "interior":            # charges outside -> harmonic inside
        rad = rng.uniform(rmax * 1.25, rmax * 2.2, n_charges)
    else:                               # exterior: charges inside -> harmonic outside
        rad = rng.uniform(rmin * 0.15, rmin * 0.7, n_charges)
    c = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    q = rng.normal(0, 1, n_charges)
    q -= q.mean()                       # zero net charge (well-posed, decaying-ish exterior)
    return {"c": c, "q": q}


def eval_u(points: np.ndarray, charges: Dict, kappa: float = 0.0) -> np.ndarray:
    """Manufactured solution from point charges. kappa=0: Laplace, u=sum q_m (-1/2pi) log|x-c_m|.
    kappa>0: modified Helmholtz (Yukawa) Delta u - kappa^2 u = 0, u=sum q_m (1/2pi) K_0(kappa|x-c_m|)."""
    c = charges["c"]; q = charges["q"]
    d = np.linalg.norm(points[:, None, :] - c[None, :, :], axis=2)  # (P, M)
    if kappa > 0:
        return (1.0 / (2 * np.pi)) * (kn(0, kappa * d) @ q)
    return (-1.0 / (2 * np.pi)) * (np.log(d) @ q)


# ----------------------------- query-point sampling -----------------------------
def sample_queries(rng, coeffs: Dict, region: str, n_q: int, margin=0.12, r_out=2.4):
    """Strictly-interior (r<(1-margin) r(theta)) or exterior (between (1+margin) Gamma and r_out) points."""
    pts = []
    rmax = float(radius_at(coeffs, np.linspace(0, 2 * np.pi, 256)).max())
    while len(pts) < n_q:
        if region == "interior":
            ang = rng.uniform(0, 2 * np.pi, n_q)
            rb = radius_at(coeffs, ang).ravel()
            rad = rng.uniform(0, 1, n_q) ** 0.5 * (1 - margin) * rb
            cand = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        else:
            cand = rng.uniform(-r_out, r_out, (n_q * 3, 2))
            ang = np.arctan2(cand[:, 1], cand[:, 0])
            rb = radius_at(coeffs, ang).ravel()
            rr = np.linalg.norm(cand, axis=1)
            cand = cand[(rr > (1 + margin) * rb) & (rr < r_out)]
        pts.extend(cand.tolist())
    return np.array(pts[:n_q])


# ----------------------------- Nystrom BEM (classical baseline + self-check) -----------------------------
def _double_layer_kernel(Xt: np.ndarray, Xs: np.ndarray, Ns: np.ndarray):
    """k(x_i, y_j) = dG/dn_y = -1/(2pi) (y-x).n_y / |y-x|^2.  Xt targets (T,2), Xs/Ns sources (S,2)."""
    diff = Xs[None, :, :] - Xt[:, None, :]               # (T,S,2) = y - x
    r2 = np.sum(diff ** 2, axis=2)                       # (T,S)
    num = np.sum(diff * Ns[None, :, :], axis=2)          # (y-x).n_y
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (-1.0 / (2 * np.pi)) * num / r2
    return k, r2


def _yukawa_double_layer_kernel(Xt, Xs, Ns, kappa, eps=1e-12):
    """Yukawa double-layer  dG/dn_y = -(kappa/2pi) K_1(kappa r) (y-x).n_y / r,  G=(1/2pi)K_0(kappa r).
    Same leading singularity as Laplace (K_1(z)~1/z) -> same curvature diagonal limit; differs smoothly."""
    diff = Xs[None, :, :] - Xt[:, None, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    num = np.sum(diff * Ns[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = -(kappa / (2 * np.pi)) * kn(1, kappa * np.maximum(r, eps)) * num / np.maximum(r, eps)
    return k


def bem_solve(bd: Dict, g: np.ndarray, region: str = "interior", kappa: float = 0.0):
    """Solve 2nd-kind double-layer BIE (-/+ 1/2 I + D0) mu = g; return density mu and an evaluator.
    kappa=0 Laplace, kappa>0 Yukawa (modified Helmholtz)."""
    X, N, w, curv = bd["X"], bd["normal"], bd["w"], bd["curv"]
    n_b = X.shape[0]
    if kappa > 0:
        K0 = _yukawa_double_layer_kernel(X, X, N, kappa)
    else:
        K0, _ = _double_layer_kernel(X, X, N)            # (n_b,n_b)
    diag = -curv / (4 * np.pi)                           # double-layer diagonal limit (same for both)
    np.fill_diagonal(K0, diag)
    D0 = K0 * w[None, :]                                  # apply trapezoidal arc weights to source index
    jump = -0.5 if region == "interior" else 0.5         # interior limit jump for outward normal
    A = jump * np.eye(n_b) + D0
    if region == "exterior":
        # pure double layer is rank-deficient for 2D exterior Dirichlet (constant nullspace);
        # deflate with a rank-one term using the arc weights so the system is invertible (Wendland).
        A = A + np.outer(w, w) / (w.sum() + 1e-12)
    mu, *_ = np.linalg.lstsq(A, g, rcond=None)            # least-squares: robust to residual deficiency

    def evaluate(Xq: np.ndarray) -> np.ndarray:
        k = (_yukawa_double_layer_kernel(Xq, X, N, kappa) if kappa > 0
             else _double_layer_kernel(Xq, X, N)[0])
        return (k * w[None, :]) @ mu
    return mu, evaluate


def validate_bem(seed=0, n_b=256, region="interior", n_test=400, kappa=0.0) -> float:
    """Max rel error of Nystrom-BEM vs the EXACT manufactured solution (should be ~1e-6 for n_b=256)."""
    rng = np.random.default_rng(seed)
    coeffs = sample_coeffs(rng)
    bd = boundary(coeffs, n_b)
    ch = sample_charges(rng, region, coeffs)
    g = eval_u(bd["X"], ch, kappa)
    _, evaluate = bem_solve(bd, g, region, kappa)
    Xq = sample_queries(rng, coeffs, region, n_test)
    u_exact = eval_u(Xq, ch, kappa)
    u_bem = evaluate(Xq)
    return float(np.linalg.norm(u_bem - u_exact) / (np.linalg.norm(u_exact) + 1e-12))


# ----------------------------- dataset assembly -----------------------------
@dataclass
class BIEConfig:
    n_modes: int = 5
    region: str = "interior"
    n_b: int = 128          # boundary discretization
    n_q: int = 256          # interior/exterior query points per sample
    n_charges: int = 6
    kappa: float = 0.0      # 0 = Laplace; >0 = Yukawa / modified Helmholtz
    with_bem: bool = True   # store classical-BEM prediction (baseline) per sample


def make_sample(rng, cfg: BIEConfig) -> Dict[str, np.ndarray]:
    coeffs = sample_coeffs(rng, cfg.n_modes)
    bd = boundary(coeffs, cfg.n_b)
    ch = sample_charges(rng, cfg.region, coeffs, cfg.n_charges)
    g = eval_u(bd["X"], ch, cfg.kappa)
    Xq = sample_queries(rng, coeffs, cfg.region, cfg.n_q)
    uq = eval_u(Xq, ch, cfg.kappa)
    out = {"Xb": bd["X"], "Nb": bd["normal"], "wb": bd["w"], "curv": bd["curv"], "speed": bd["speed"],
           "g": g, "Xq": Xq, "uq": uq, "a": coeffs["a"], "b": coeffs["b"]}
    if cfg.with_bem:
        try:
            mu, evaluate = bem_solve(bd, g, cfg.region, cfg.kappa)
            out["bem_uq"] = evaluate(Xq); out["mu"] = mu
        except np.linalg.LinAlgError:
            out["bem_uq"] = np.full_like(uq, np.nan); out["mu"] = np.full_like(g, np.nan)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
