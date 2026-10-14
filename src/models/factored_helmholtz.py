"""Real-valued, full-learned (NO-SPLIT) factored boundary operator for the HELMHOLTZ rank stress test.

The Laplace Method-4 model (src/models/scalable_factored.FactoredNBIO) learns a SEPARABLE/low-rank SMOOTH
*correction* on top of the analytically known Laplace double-layer kernel:

    K[i,j] = gate * K_known_doublelayer[i,j]  +  phi(node_i) . psi(node_j)        (split=True)

For oscillatory Helmholtz (Delta u + k^2 u = 0) there is NO real Laplace-style known kernel to lean on
(the true Green's function H_0^(1)(k r) is complex and high-rank in the far field), so we ask the FACTORED
kernel to learn the FULL operator from scratch -- i.e. split=False, no known double-layer term:

    K[i,j] = phi(node_i) . psi(node_j)            (sum over r latent channels, REAL throughout)

This file is a THIN wrapper around the existing FactoredNBIO. FactoredNBIO already supports the no-split
path (split=False -> self.log_gate is None, _assemble/_potential drop the known kernel), so we reuse it
verbatim and only fix the construction to the no-split Helmholtz configuration and expose `rank`. Nothing
in FactoredNBIO is modified.

The boundary system assembled is the second-kind form

    A = jump * I + (K_factored * w),     A mu = g       (real torch.linalg.solve / FixedPointSolve)

and the interior field is evaluated with the SAME factored kernel, feeding zero normal/curvature on the
query (off-surface) side -- exactly as FactoredNBIO._potential does. Everything is real-valued: the
manufactured plane-wave targets are real exact Helmholtz solutions, so a real learned operator is the
honest object to fit, and its required rank vs k is the quantity of interest.
"""
from __future__ import annotations

from .scalable_factored import FactoredNBIO


class FactoredHelmholtzNBIO(FactoredNBIO):
    """FactoredNBIO pinned to the NO-SPLIT / full-learned-kernel (Helmholtz) mode.

    K[i,j] = phi(node_i) . psi(node_j) with phi, psi small per-node MLPs over [x, y, nx, ny, curv] -> R^r.
    No known kernel, no gate. Real-valued. `rank` is the separable rank r being stress-tested vs wavenumber.
    """

    def __init__(self, rank=8, hidden=64, n_layers=3, feat_scale=1.0, region="interior",
                 solve="direct", n_iter=20, omega=1.0, init_scale=0.05):
        # split=False forces the pure factored (full-operator) path: no known double-layer term, no gate.
        super().__init__(rank=rank, hidden=hidden, n_layers=n_layers, feat_scale=feat_scale,
                         region=region, split=False, solve=solve, n_iter=n_iter, omega=omega,
                         init_scale=init_scale)
        assert self.log_gate is None and self.split is False, "Helmholtz model must be no-split (full learned kernel)"
