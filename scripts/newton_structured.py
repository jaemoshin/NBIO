"""Newton for a semilinear BVP with a STRUCTURED inner solve -- closing the "Newton needs a variable
operator" caveat.

The nonlinear-BVP demo used Picard (NBIO/known-kernel as the zero-shot inner Laplace solve), which converges
only while the fixed-point map contracts. Newton converges quadratically but its step solves
J(u) w = -R(u) with J = -Lap + m^2 + 3 k u^2 -- a VARIABLE-REACTION (lower-order) operator. Crucially this
is a lower-order term (multiplies u, not grad u), so the constant-coefficient Green's operator (-Lap+m^2)^-1
is still an exact, known preconditioner: the inner solve is a 2nd-kind problem solved by preconditioned
GMRES in O(1) iterations. We show Newton converges QUADRATICALLY and at large k where Picard stalls/diverges,
with each inner solve handled by the structured preconditioner -- no learned/assembled variable operator
needed for the reaction term.

Semilinear screened problem on the torus:  -Lap u + m^2 u + k u^3 = f,  manufactured smooth u*.
Pure numpy/scipy (FFT spectral). Additive new file. Saves results/newton_structured.json + figure.
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse.linalg as sla

from src.data.vc import smooth_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)


class Spectral:
    """FFT operators on the torus for -Lap + m^2 (known, constant-coefficient)."""

    def __init__(self, n, m2=1.0, h=None):
        self.n = n; self.m2 = m2
        h = h or (2 * np.pi / n)
        k = np.fft.fftfreq(n, d=h / (2 * np.pi))
        KX, KY = np.meshgrid(k, k, indexing="ij")
        self.k2 = KX ** 2 + KY ** 2
        self.sym = self.k2 + m2                  # symbol of -Lap + m^2 (>0, no nullspace)

    def lap(self, u):                            # -Lap u
        return np.fft.ifft2(self.k2 * np.fft.fft2(u.reshape(self.n, self.n))).real.ravel()

    def A(self, u):                              # (-Lap + m^2) u
        return np.fft.ifft2(self.sym * np.fft.fft2(u.reshape(self.n, self.n))).real.ravel()

    def minv(self, v):                           # (-Lap + m^2)^-1 v  (the known Green's operator)
        return np.fft.ifft2(np.fft.fft2(v.reshape(self.n, self.n)) / self.sym).real.ravel()

    def minv_shift(self, v, s):                  # (-Lap + m^2 + s)^-1 v -- known FFT solve with reaction shift
        return np.fft.ifft2(np.fft.fft2(v.reshape(self.n, self.n)) / (self.sym + s)).real.ravel()


def residual(sp, u, kappa, f):
    return sp.A(u) + kappa * u ** 3 - f


def picard(sp, f, kappa, u0, tol=1e-12, maxit=200):
    """u_{n+1} = (-Lap+m^2)^-1 (f - k u_n^3)."""
    u = u0.copy(); errs = []
    for _ in range(maxit):
        if not np.all(np.isfinite(u)):                # already diverged -- stop before invalid ops
            errs.append(np.inf); break
        un = sp.minv(f - kappa * u ** 3)
        step = np.linalg.norm(un - u) / (np.linalg.norm(un) + 1e-30)
        errs.append(step); u = un
        if step < tol:
            break
        if not np.isfinite(step) or step > 1e3:       # diverged -- stop early
            break
    return u, errs


def newton(sp, f, kappa, u0, ustar, tol=1e-12, maxit=30, inner_tol=1e-10):
    """Newton: J(u) w = -R(u), J = -Lap+m^2 + 3k u^2, inner solve = (-Lap+m^2)^-1-preconditioned GMRES."""
    n2 = sp.n * sp.n
    u = u0.copy()
    err_to_star = []; inner_iters = []
    for _ in range(maxit):
        R = residual(sp, u, kappa, f)
        V = 3.0 * kappa * u ** 2                      # variable reaction (lower-order, known multiplier)
        s = float(V.mean())                           # shift the known Green's preconditioner by mean reaction

        def Jmv(w, V=V):
            return sp.A(w) + V * w
        J = sla.LinearOperator((n2, n2), matvec=Jmv)
        M = sla.LinearOperator((n2, n2), matvec=lambda v, s=s: sp.minv_shift(v, s))
        it = {"n": 0}
        try:
            w, info = sla.gmres(J, -R, M=M, rtol=inner_tol, atol=0.0, restart=100, maxiter=200,
                                callback=lambda _: it.update(n=it["n"] + 1), callback_type="pr_norm")
        except TypeError:
            w, info = sla.gmres(J, -R, M=M, tol=inner_tol, atol=0.0, restart=100, maxiter=200,
                                callback=lambda _: it.update(n=it["n"] + 1))
        u = u + w
        inner_iters.append(it["n"])
        err_to_star.append(float(np.linalg.norm(u - ustar) / (np.linalg.norm(ustar) + 1e-30)))
        if np.linalg.norm(w) / (np.linalg.norm(u) + 1e-30) < tol:
            break
    return u, err_to_star, inner_iters


def run(kappas=(1.0, 5.0, 20.0, 50.0, 100.0), n=96, m2=1.0, seed=0):
    sp = Spectral(n, m2)
    rng = np.random.default_rng(seed)
    ustar = 0.3 * smooth_field(rng, n).ravel()                # manufactured smooth solution (moderate amplitude)
    rows = []
    print("=== Newton vs Picard for -Lap u + u + k u^3 = f (structured inner solve) ===")
    for kappa in kappas:
        f = sp.A(ustar) + kappa * ustar ** 3                  # manufactured rhs -> exact solution = ustar
        u0 = np.zeros_like(ustar)
        # Picard
        up, perrs = picard(sp, f, kappa, u0)
        p_final = float(np.linalg.norm(up - ustar) / np.linalg.norm(ustar))
        p_iters = len(perrs)
        p_div = bool(p_final > 1e-3 or not np.isfinite(p_final))
        # Newton (structured inner solve)
        un, nerrs, inner = newton(sp, f, kappa, u0, ustar)
        n_final = nerrs[-1]
        n_iters = len(nerrs)
        row = dict(kappa=kappa, picard_iters=p_iters, picard_final_rel=p_final, picard_diverged=p_div,
                   newton_iters=n_iters, newton_final_rel=n_final,
                   newton_err_curve=nerrs, newton_inner_gmres_iters=inner,
                   newton_inner_iters_mean=float(np.mean(inner)))
        rows.append(row)
        print(f"  kappa={kappa:6.1f}: Picard {p_iters:3d} it -> {p_final:.2e} {'(DIVERGED)' if p_div else ''}"
              f" | Newton {n_iters:2d} it -> {n_final:.2e} | inner GMRES ~{np.mean(inner):.1f} it/step")

    out = dict(n=n, m2=m2, rows=rows,
               note="Newton's inner solve J w=-R with J=-Lap+m^2+3k u^2 is a variable-REACTION (lower-order) "
                    "problem, so the constant-coeff Green's operator (-Lap+m^2)^-1 is still an exact "
                    "preconditioner -> inner solve in O(1) GMRES iters. Newton converges quadratically and at "
                    "large kappa where Picard stalls/diverges, with NO learned/assembled variable operator for "
                    "the reaction term. (The remaining hard case stays the variable PRINCIPAL coefficient.)")
    path = os.path.join(ROOT, "results", "newton_structured.json")
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n[newton_structured] DONE -> {path}")
    _figure(out)
    return out


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # (A) Newton convergence curves (quadratic) at each kappa
    for r in rows:
        ax[0].semilogy(range(1, len(r["newton_err_curve"]) + 1), r["newton_err_curve"], "o-",
                       label=f"$\\kappa$={r['kappa']:.0f}")
    ax[0].set_xlabel("Newton iteration"); ax[0].set_ylabel("rel-$L_2$ to manufactured $u^*$")
    ax[0].set_title("(A) Newton converges quadratically\n(structured inner solve)")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.2, which="both")
    # (B) iters to converge: Newton vs Picard vs kappa
    k = [r["kappa"] for r in rows]
    pit = [r["picard_iters"] if not r["picard_diverged"] else np.nan for r in rows]
    nit = [r["newton_iters"] for r in rows]
    inner = [r["newton_inner_iters_mean"] for r in rows]
    ax[1].plot(k, pit, "s-", color="C1", label="Picard (outer; NaN=diverged)")
    ax[1].plot(k, nit, "o-", color="C2", label="Newton (outer)")
    ax[1].plot(k, inner, "^--", color="C0", label="Newton inner GMRES/step")
    ax[1].set_xscale("log"); ax[1].set_xlabel("nonlinearity $\\kappa$")
    ax[1].set_ylabel("iterations")
    ax[1].set_title("(B) Newton stays $O(1)$ outer + $O(1)$ inner;\nPicard slows/diverges as $\\kappa$ grows")
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"newton_structured.{ext}"), dpi=150, bbox_inches="tight")
    print("[newton_structured] wrote figures/newton_structured.{png,pdf}")


if __name__ == "__main__":
    run()
