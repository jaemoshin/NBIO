#!/usr/bin/env python
"""P0/P1: panel-based QBX single-layer on TRIANGULATED surfaces (numpy, float64).

P0 = triangulate the SAME analytic surfaces we solve spectrally (star-sphere via subdivided
icosahedron projection; torus via periodic quad grid) so manufactured harmonic GT gives exact
validation. P1 = panel pipeline: piecewise-linear density (hat functions), 3-pt Gauss per triangle
for the far field, LOCAL QBX for near/self: target x_i gets center c_i = x_i + beta*n_i; panels
within R_near are subdivided L levels and integrated against the Legendre expansion about c_i;
far panels are evaluated directly at x_i (smooth there).  Expect O(h^p_panel) convergence
(panel-limited), NOT spectral — measured honestly vs refinement.
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np

PI = math.pi
GAUSS3 = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])   # degree-2, w=A/3


# ------------------------------------------------ meshes of analytic surfaces
def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],
                  [10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],
                  [2,4,11],[6,2,10],[8,6,7],[9,8,1]])
    return v, f


def subdivide(v, f):
    v = list(map(tuple, v)); idx = {vv: i for i, vv in enumerate(v)}
    def mid(a, b):
        m = tuple((np.array(v[a]) + np.array(v[b])) / 2)
        if m not in idx:
            idx[m] = len(v); v.append(m)
        return idx[m]
    nf = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    V = np.array(v); V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, np.array(nf)


def star_r(dirs, pa=0.10, pb=0.06, pc=0.04, amp=0.3):
    x, y, z = dirs.T
    st2 = x * x + y * y
    c2 = (x * x - y * y) / np.maximum(st2, 1e-30); s2 = 2 * x * y / np.maximum(st2, 1e-30)
    return 1 + amp * (pa * z + pb * st2 * c2 + pc * st2 * s2)


def mesh_star_sphere(k=3, amp=0.3):
    v, f = icosahedron()
    for _ in range(k):
        v, f = subdivide(v, f)
    return v * star_r(v, amp=amp)[:, None], f


def mesh_torus(nt=40, nph=16, R=1.0, r0=0.35):
    th = np.arange(nt) * 2 * PI / nt; ph = np.arange(nph) * 2 * PI / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    ring = R + r0 * np.cos(PH)
    V = np.stack([ring * np.cos(TH), ring * np.sin(TH), r0 * np.sin(PH)], -1).reshape(-1, 3)
    idx = lambda i, j: (i % nt) * nph + (j % nph)
    F = []
    for i in range(nt):
        for j in range(nph):
            F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                  [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return V, np.array(F)


def mesh_geom(V, F):
    e1 = V[F[:, 1]] - V[F[:, 0]]; e2 = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2); fa = 0.5 * np.linalg.norm(fn, axis=1)
    fn = fn / (2 * fa[:, None])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn * fa[:, None])
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    # orient outward (works for our star-shaped/toroidal meshes centered near origin)
    ctr = V.mean(0)
    sgn = np.sign(((V - ctr) * vn).sum(1)).mean()
    if sgn < 0:
        vn = -vn
    return fa, fn, vn


def tri_quad(V, F, fa, levels=0):
    """Quadrature points/weights/barycentric-owner columns for (optionally subdivided) triangles."""
    P = V[F]                                  # (M,3,3)
    for _ in range(levels):
        A, B, C = P[:, 0], P[:, 1], P[:, 2]
        AB, BC, CA = (A + B) / 2, (B + C) / 2, (C + A) / 2
        P = np.concatenate([np.stack([A, AB, CA], 1), np.stack([B, BC, AB], 1),
                            np.stack([C, CA, BC], 1), np.stack([AB, BC, CA], 1)])
    M = P.shape[0]
    e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]
    a = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    pts = (GAUSS3[None, :, :, None] * P[:, None, :, :]).sum(2).reshape(-1, 3)     # (M*3,3)
    w = np.repeat(a / 3, 3)
    return pts, w, M


def local_thickness(V, vn, h):
    """Per-vertex distance to the nearest OPPOSITE-FACING vertex (thin-part detector)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    tau = np.full(len(V), np.inf)
    pairs = tree.query_ball_point(V, r=8 * h)
    for i, nb in enumerate(pairs):
        nb = np.array([j for j in nb if j != i])
        if len(nb) == 0:
            continue
        opp = nb[(vn[nb] * vn[i]).sum(1) < -0.3]
        if len(opp):
            tau[i] = np.linalg.norm(V[opp] - V[i], axis=1).min()
    return tau


def assemble_panel_qbx(V, F, beta_mult=2.0, rnear_mult=3.0, sub_levels=2, p=10, adaptive=True):
    N = V.shape[0]
    fa, fn, vn = mesh_geom(V, F)
    h = math.sqrt(fa.mean())
    beta = beta_mult * h; R_near = rnear_mult * beta
    beta_i = np.full(N, beta)
    if adaptive:  # thin parts: keep the center inside (never past ~30% of local thickness)
        tau = local_thickness(V, vn, h)
        beta_i = np.minimum(beta, 0.3 * tau)
        beta_i = np.maximum(beta_i, 0.5 * h)   # floor: never collapse below h/2
    # base quad (no subdivision) with barycentric column mapping
    P0 = V[F]
    e1 = P0[:, 1] - P0[:, 0]; e2 = P0[:, 2] - P0[:, 0]
    aF = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    qp = (GAUSS3[None, :, :, None] * P0[:, None, :, :]).sum(2)                    # (M,3,3)
    S = np.zeros((N, N))
    fc = P0.mean(1)                                                               # face centers
    for i in range(N):
        beta_t = beta_i[i]
        x = V[i]; c = x + beta_t * vn[i]
        dfc = np.linalg.norm(fc - x, axis=1)
        near = dfc < R_near
        # FAR: direct at x, 3-pt Gauss, hat weights = barycentric of the Gauss pts
        for g in range(3):
            y = qp[~near, g]; wq = aF[~near] / 3
            G = 1.0 / (4 * PI * np.linalg.norm(y - x, axis=1))
            for k in range(3):
                np.add.at(S[i], F[~near, k], G * wq * GAUSS3[g, k])
        # NEAR: subdivided panels integrated against the Legendre expansion about c
        Fn_ = F[near]
        if len(Fn_):
            Pn = V[Fn_]
            colw = np.repeat(GAUSS3[None], Pn.shape[0], 0)                        # bary owner wts
            cols = np.repeat(Fn_[:, None, :], 3, 1)                               # (m,3g,3v)
            Pc, colsc = Pn, cols
            # smaller beta needs finer near-quadrature: sub-tri diameter ~< beta/2
            lv = sub_levels + max(0, int(math.ceil(math.log2(max(1.0, beta / max(beta_t, 1e-12))))))
            for _ in range(min(lv, sub_levels + 3)):
                A, B, C = Pc[:, 0], Pc[:, 1], Pc[:, 2]
                AB, BC, CA = (A + B) / 2, (B + C) / 2, (C + A) / 2
                Pc = np.concatenate([np.stack([A, AB, CA], 1), np.stack([B, BC, AB], 1),
                                     np.stack([C, CA, BC], 1), np.stack([AB, BC, CA], 1)])
                colsc = np.concatenate([colsc] * 4)
                # barycentric owner weights for children w.r.t. PARENT vertices
                W = colw
                W1 = (np.stack([W[:, 0], (W[:, 0]+W[:, 1])/2, (W[:, 0]+W[:, 2])/2], 1))
                W2 = (np.stack([W[:, 1], (W[:, 1]+W[:, 2])/2, (W[:, 0]+W[:, 1])/2], 1))
                W3 = (np.stack([W[:, 2], (W[:, 0]+W[:, 2])/2, (W[:, 1]+W[:, 2])/2], 1))
                W4 = (np.stack([(W[:, 0]+W[:, 1])/2, (W[:, 1]+W[:, 2])/2, (W[:, 0]+W[:, 2])/2], 1))
                colw = np.concatenate([W1, W2, W3, W4])
            e1s = Pc[:, 1] - Pc[:, 0]; e2s = Pc[:, 2] - Pc[:, 0]
            a_s = 0.5 * np.linalg.norm(np.cross(e1s, e2s), axis=1)
            for g in range(3):
                y = (GAUSS3[g][None, :, None] * Pc).sum(1)
                d = y - c; r_y = np.linalg.norm(d, axis=1).clip(1e-30)
                cosg = -(vn[i][None, :] * d).sum(1) / r_y
                Sker = 1.0 / r_y
                Pm1 = np.ones_like(cosg); Pl = cosg.copy()
                Sker = Sker + beta_t / r_y ** 2 * Pl
                for l in range(1, p):
                    Pn_ = ((2 * l + 1) * cosg * Pl - l * Pm1) / (l + 1)
                    Pm1, Pl = Pl, Pn_
                    Sker = Sker + beta_t ** (l + 1) / r_y ** (l + 2) * Pl
                Gq = Sker / (4 * PI) * (a_s / 3)
                # hat value of PARENT vertex k at this Gauss point = sum_childvtx GAUSS3[g,cv]*colw[m,cv,k]
                hatk = np.einsum("v,mvk->mk", GAUSS3[g], colw)                    # (m,3)
                for k in range(3):
                    np.add.at(S[i], colsc[:, 0, k], Gq * hatk[:, k])
    return S, vn, fa


def u_harm(X, c, q):
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2).clip(1e-12)
    return (1.0 / (4 * PI)) * ((1.0 / d) @ q)


def _fib(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n); th = PI * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def run(kind, k, seed=0, **kw):
    rng = np.random.default_rng(seed)
    if kind == "star":
        V, F = mesh_star_sphere(k=k)
        Xq = _fib(200) * rng.uniform(0.05, 0.4, (200, 1))
    else:
        V, F = mesh_torus(nt=10 * 2 ** k, nph=4 * 2 ** k)
        aa = np.linspace(0, 2 * PI, 200, endpoint=False)
        off = rng.uniform(0, 0.12, 200); be = rng.uniform(0, 2 * PI, 200)
        ring = 1.0 + off * np.cos(be)
        Xq = np.stack([ring * np.cos(aa), ring * np.sin(aa), off * np.sin(be)], 1)
    S, vn, fa = assemble_panel_qbx(V, F, **kw)
    c = _fib(8) * rng.uniform(2.0, 2.8, (8, 1)); q = rng.normal(size=8); q -= q.mean()
    g = u_harm(V, c, q)
    sig = np.linalg.solve(S + 1e-12 * np.eye(len(V)), g)
    # interior eval: 3-pt Gauss direct with linear density
    P0 = V[F]; aF = fa
    uq = u_harm(Xq, c, q)
    up = np.zeros(len(Xq))
    for gq in range(3):
        y = (GAUSS3[gq][None, :, None] * P0).sum(1)
        sig_y = (GAUSS3[gq][None, :] * sig[F]).sum(1)
        D = np.linalg.norm(Xq[:, None, :] - y[None, :, :], axis=2).clip(1e-15)
        up += (1.0 / (4 * PI * D) * (sig_y * aF / 3)[None, :]).sum(1)
    return float(np.linalg.norm(up - uq) / np.linalg.norm(uq)), len(V)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/panel_qbx.json"); a = ap.parse_args()
    res = {}
    for kind in ("star", "torus"):
        for k in (2, 3, 4):
            e, N = run(kind, k, sub_levels=3, p=10)
            res[f"{kind}_k{k}"] = {"N": N, "rel_l2": e}
            print(f"[{kind}] k={k} N={N}  rel-L2 {e:.3e}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
