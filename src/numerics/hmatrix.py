"""Recursive H-matrix (hierarchical low-rank) for the NBIO second-kind boundary operator.

Closes the scalability caveat end-to-end: builds A = (-1/2 I + W K0) for the interior double-layer
WITHOUT ever forming the dense matrix -- admissible (well-separated) blocks are compressed directly from
the kernel by partially-pivoted Adaptive Cross Approximation (ACA), near/diagonal blocks are kept dense at
the leaves. Matvec is recursive; the second-kind system is then solved by GMRES in O(1) iterations.

Geometry note: the boundary is a closed curve, so a simple bisection of the arc ordering yields two arcs
that touch at BOTH junctions (near-field) -- a pure HODLR is therefore wrong here. We use a standard
H-matrix with the geometric admissibility  dist(c_s,c_t) > eta*(rad_s+rad_t): opposite arcs compress,
near-junction arcs recurse to small dense leaves. This gives O(n log n) storage/matvec with no dense
assembly.

Pure numpy. Standalone (depends only on src.data.bie for geometry/kernel); additive new file.
"""
from __future__ import annotations

import numpy as np

from ..data.bie import boundary, sample_coeffs, _double_layer_kernel


# ----------------------------- geometry / kernel access -----------------------------
class Operator:
    """Lazily provides entries of A = -1/2 I + D0, D0[i,j] = K0(x_i; x_j, n_j) * w_j, diag = -curv/(4pi)."""

    def __init__(self, bd):
        self.X = bd["X"]; self.N = bd["normal"]; self.w = bd["w"]; self.curv = bd["curv"]
        self.n = self.X.shape[0]
        self.jump = -0.5

    def block(self, I, J):
        """Dense sub-block A[I, J] (handles self/diagonal entries correctly)."""
        Xt = self.X[I]; Xs = self.X[J]; Ns = self.N[J]
        K, r2 = _double_layer_kernel(Xt, Xs, Ns)             # (|I|,|J|)
        # fix any coincident points (self term -> curvature limit)
        eqmask = r2 < 1e-30
        if eqmask.any():
            # map back which (a,b) are the same global node
            ia, jb = np.where(eqmask)
            same = I[ia] == J[jb]
            K[ia[same], jb[same]] = (-self.curv[I[ia[same]]] / (4 * np.pi))
        D0 = K * self.w[None, J]
        A = D0
        # add jump*I where global indices coincide
        eqI = (I[:, None] == J[None, :])
        if eqI.any():
            a, b = np.where(eqI)
            A[a, b] += self.jump
        return A

    def kernel_block(self, I, J):
        """Smooth far-field block (no self/jump) for ACA on admissible pairs: D0[I,J] = K0*w."""
        Xt = self.X[I]; Xs = self.X[J]; Ns = self.N[J]
        K, _ = _double_layer_kernel(Xt, Xs, Ns)
        return K * self.w[None, J]

    def kernel_rows(self, I, J):
        return self.kernel_block(I, J)


# ----------------------------- ACA (partial pivoting) -----------------------------
def aca(get_rows, get_cols, m, n, tol=1e-8, max_rank=64):
    """Partially-pivoted ACA of an implicit m x n block.
    get_rows(i) -> row i (length n); get_cols(j) -> col j (length m). Returns U(m,k), V(k,n)."""
    U = np.zeros((m, 0)); V = np.zeros((0, n))
    Rrows, Rcols = [], []
    approx_fro2 = 0.0
    i = 0
    used_rows = set(); used_cols = set()
    for _ in range(min(max_rank, m, n)):
        used_rows.add(i)
        row = get_rows(i).astype(np.float64).copy()
        if U.shape[1] > 0:
            row -= U[i, :] @ V
        j = int(np.argmax(np.abs(_mask(row, used_cols))))
        delta = row[j]
        if abs(delta) < 1e-300:
            # pick an unused row
            nxt = _first_unused(m, used_rows)
            if nxt is None:
                break
            i = nxt
            continue
        col = get_cols(j).astype(np.float64).copy()
        if U.shape[1] > 0:
            col -= U @ V[:, j]
        row = row / delta
        used_cols.add(j)
        U = np.column_stack([U, col]); V = np.vstack([V, row])
        # update Frobenius estimate (Bebendorf): ||u||^2||v||^2 + cross terms
        un2 = col @ col; vn2 = row @ row
        cross = 0.0
        for t in range(U.shape[1] - 1):
            cross += (U[:, t] @ col) * (V[t, :] @ row)
        approx_fro2 += un2 * vn2 + 2 * cross
        if un2 * vn2 <= (tol ** 2) * max(approx_fro2, 1e-300):
            break
        # next pivot row = largest residual entry in last col
        cand = _mask(col, used_rows)
        i = int(np.argmax(np.abs(cand)))
    return U, V


def _mask(v, used):
    out = v.copy()
    if used:
        idx = np.fromiter(used, dtype=int)
        idx = idx[idx < len(out)]
        out[idx] = 0.0
    return out


def _first_unused(m, used):
    for i in range(m):
        if i not in used:
            return i
    return None


# ----------------------------- cluster tree + block tree -----------------------------
class Cluster:
    __slots__ = ("a", "b", "c", "rad", "left", "right")

    def __init__(self, a, b, X):
        self.a = a; self.b = b
        pts = X[a:b]
        self.c = pts.mean(0)
        self.rad = float(np.sqrt(((pts - self.c) ** 2).sum(1)).max()) if b > a else 0.0
        self.left = self.right = None

    @property
    def idx(self):
        return np.arange(self.a, self.b)


def build_cluster(a, b, X, leaf):
    cl = Cluster(a, b, X)
    if b - a > leaf:
        m = (a + b) // 2
        cl.left = build_cluster(a, m, X, leaf)
        cl.right = build_cluster(m, b, X, leaf)
    return cl


def admissible(s, t, eta):
    d = np.sqrt(((s.c - t.c) ** 2).sum())
    return d > eta * (s.rad + t.rad)


class HMatrix:
    def __init__(self, op, leaf=64, tol=1e-8, max_rank=64, eta=1.5):
        self.op = op; self.leaf = leaf; self.tol = tol; self.max_rank = max_rank; self.eta = eta
        self.root = build_cluster(0, op.n, op.X, leaf)
        self.dense = []      # (Iarange a..b, Jarange a..b, M)
        self.low = []        # (sa,sb, ta,tb, U, V)  block s(rows=targets t? ) -> we store rows=t, cols=s
        self.storage = 0
        self.dense_entries = 0
        self._build(self.root, self.root)

    def _build(self, t, s):
        # block rows = targets t, cols = sources s
        if admissible(s, t, self.eta) and (t.b - t.a) > 0 and (s.b - s.a) > 0:
            I = t.idx; J = s.idx
            m, n = len(I), len(J)

            def get_rows(i, I=I, J=J):
                return self.op.kernel_block(I[i:i + 1], J)[0]

            def get_cols(j, I=I, J=J):
                return self.op.kernel_block(I, J[j:j + 1])[:, 0]

            U, V = aca(get_rows, get_cols, m, n, self.tol, self.max_rank)
            if U.shape[1] == 0 or U.shape[1] >= min(m, n) // 2:
                # not worth compressing -> dense
                M = self.op.block(I, J)
                self.dense.append((t.a, t.b, s.a, s.b, M)); self.storage += M.size; self.dense_entries += M.size
            else:
                self.low.append((t.a, t.b, s.a, s.b, U, V)); self.storage += U.size + V.size
            return
        # inadmissible
        t_leaf = t.left is None; s_leaf = s.left is None
        if t_leaf and s_leaf:
            I = t.idx; J = s.idx
            M = self.op.block(I, J)
            self.dense.append((t.a, t.b, s.a, s.b, M)); self.storage += M.size; self.dense_entries += M.size
            return
        # recurse on the larger side(s)
        ts = [t.left, t.right] if not t_leaf else [t]
        ss = [s.left, s.right] if not s_leaf else [s]
        for tt in ts:
            for ssub in ss:
                self._build(tt, ssub)

    def matvec(self, x):
        y = np.zeros(self.op.n, dtype=np.float64)
        for (ta, tb, sa, sb, M) in self.dense:
            y[ta:tb] += M @ x[sa:sb]
        for (ta, tb, sa, sb, U, V) in self.low:
            y[ta:tb] += U @ (V @ x[sa:sb])
        return y

    @property
    def storage_fraction(self):
        return self.storage / float(self.op.n ** 2)

    @property
    def n_blocks(self):
        return len(self.dense), len(self.low)
