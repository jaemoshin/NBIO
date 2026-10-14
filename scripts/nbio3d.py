#!/usr/bin/env python
"""3D NBIO: extend the boundary-integral operator to surfaces. Known 3D Laplace single-layer
S(x,y)=1/(4pi|x-y|) (self-patch diagonal) + learned smooth correction (M1) + differentiable solve (M2) +
off-surface evaluation (M3). Manufactured harmonic GT (exterior point charges -> exact interior values).
Trained on a family of deformed spheres; zero-shot eval on unseen deformation, resolution, and exterior.
Baselines: bare structure (no correction = classical single-layer solve) and a DeepSet-3D black box."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.utils.common import set_seed, get_device, configure_torch, save_json

PI = math.pi


# ----------------------------- geometry + manufactured data -----------------------------
def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def deformed_sphere(n, rng, amp=1.0):
    X = fib_sphere(n); x, y, z = X.T
    r = 1.0 + amp * (0.10 * z + 0.06 * (x * x - y * y) + 0.05 * rng.normal() * x * y + 0.04 * rng.normal() * y * z)
    Xs = X * r[:, None]
    w = (4 * PI / n) * r ** 2                      # area weights (radial-graph approx)
    nrm = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
    return Xs.astype(np.float32), w.astype(np.float32), nrm.astype(np.float32)


def u3(X, c, q):
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2).clip(1e-9)
    return (1.0 / (4 * PI)) * ((1.0 / d) @ q)


def sample3d(rng, n, amp=1.0, exterior=False, n_q=360):
    Xs, w, nrm = deformed_sphere(n, rng, amp)
    ang = fib_sphere(8)
    if exterior:
        c = ang * rng.uniform(0.1, 0.55, (8, 1)); qpts = fib_sphere(n_q) * rng.uniform(1.3, 1.9, (n_q, 1))
    else:
        c = ang * rng.uniform(1.7, 2.6, (8, 1)); qpts = fib_sphere(n_q) * rng.uniform(0.0, 0.7, (n_q, 1))
    q = rng.normal(size=8); q -= q.mean()
    g = u3(Xs, c, q); uq = u3(qpts, c, q)
    return {"X": Xs, "w": w, "nrm": nrm, "g": g.astype(np.float32),
            "Xq": qpts.astype(np.float32), "uq": uq.astype(np.float32)}


def batch(rng, B, n, amp=1.0, exterior=False, device="cpu"):
    s = [sample3d(rng, n, amp, exterior) for _ in range(B)]
    t = lambda k: torch.tensor(np.stack([x[k] for x in s]), device=device)
    return {k: t(k) for k in s[0]}


# ----------------------------- models -----------------------------
def _mlp(din, dh, dout, n, init0=False):
    L = [nn.Linear(din, dh), nn.GELU()]
    for _ in range(n - 1):
        L += [nn.Linear(dh, dh), nn.GELU()]
    L += [nn.Linear(dh, dout)]
    m = nn.Sequential(*L)
    if init0:
        nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


def feats3d(Xt, Xs, Nt, Ns):
    diff = Xs[:, None, :, :] - Xt[:, :, None, :]            # (B,T,S,3)
    r = diff.norm(dim=-1, keepdim=True).clamp_min(1e-9); dh = diff / r
    nt = Nt[:, :, None, :]; ns = Ns[:, None, :, :]
    return torch.cat([dh, r, torch.exp(-r), (dh * nt).sum(-1, True),
                      (dh * ns).sum(-1, True), (nt * ns).sum(-1, True)], -1)   # 8


class NBIO3D(nn.Module):
    """known 3D single layer + learned smooth correction; differentiable single-layer Dirichlet solve."""
    def __init__(self, hidden=32, n_layers=2, split=True, eps=1e-8):
        super().__init__()
        self.split = split; self.eps = eps
        self.corr = _mlp(8, hidden, 1, n_layers, init0=True) if split else None

    def _S(self, Xt, Xs, w, Nt, Ns, selfblock):
        D = torch.cdist(Xt, Xs)
        S = (1.0 / (4 * PI * D.clamp_min(1e-12))) * w[:, None, :]
        if selfblock:
            pr = torch.sqrt(w / PI)                                  # self-patch single-layer value
            S = S - torch.diagonal(S, dim1=-2, dim2=-1).diag_embed() + (pr / 2).diag_embed()
        if self.split:
            S = S + self.corr(feats3d(Xt, Xs, Nt, Ns)).squeeze(-1) * w[:, None, :]
        return S

    def forward(self, b):
        X, w, nrm, g, Xq = b["X"], b["w"], b["nrm"], b["g"], b["Xq"]
        A = self._S(X, X, w, nrm, nrm, selfblock=True)
        A = A + self.eps * torch.eye(X.shape[1], device=X.device)[None]
        sigma = torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)
        Nq = torch.zeros(Xq.shape[0], Xq.shape[1], 3, device=Xq.device)
        P = self._S(Xq, X, w, Nq, nrm, selfblock=False)
        return torch.bmm(P, sigma.unsqueeze(-1)).squeeze(-1)


class DeepSet3D(nn.Module):
    def __init__(self, hidden=128, p=128):
        super().__init__()
        self.enc = _mlp(7, hidden, p, 3); self.dec = _mlp(3 + p, hidden, 1, 4)

    def forward(self, b):
        nodes = torch.cat([b["X"], b["nrm"], b["g"].unsqueeze(-1)], -1)
        code = self.enc(nodes).mean(1)
        Q = b["Xq"].shape[1]
        return self.dec(torch.cat([b["Xq"], code[:, None, :].expand(-1, Q, -1)], -1)).squeeze(-1)


def rel(u, uq):
    return (torch.linalg.vector_norm(u - uq, dim=1) / torch.linalg.vector_norm(uq, dim=1).clamp_min(1e-12)).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="nbio3d"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=360); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/nbio3d"); a = ap.parse_args()
    set_seed(a.seed); configure_torch(); dev = get_device(True)
    if a.model == "nbio3d": model = NBIO3D(split=True)
    elif a.model == "nbio3d_structure": model = NBIO3D(split=False)
    else: model = DeepSet3D()
    model = model.to(dev)
    np_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[3d] model={a.model} trainable={np_train}")
    rng = np.random.default_rng(a.seed)
    if np_train > 0:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(a.epochs):
            model.train(); b = batch(rng, 8, a.n, amp=1.0, device=dev)
            loss = rel(model(b), b["uq"]); opt.zero_grad(); loss.backward(); opt.step()
            if ep % 20 == 0: print(f"  ep{ep} train rel {loss.item():.3e}")
    model.eval()
    res = {}
    with torch.no_grad():
        for tag, kw in [("indist", dict(amp=1.0)), ("zs_strong_deform", dict(amp=1.8)),
                        ("zs_res720", dict(amp=1.0)), ("zs_exterior", dict(amp=1.0, exterior=True))]:
            n_eval = 720 if tag == "zs_res720" else a.n
            rr = np.mean([rel(model(batch(np.random.default_rng(1000 + i), 4, n_eval, device=dev, **kw)),
                              batch(np.random.default_rng(1000 + i), 4, n_eval, device=dev, **kw)["uq"]).item()
                          for i in range(3)])
            res[tag] = float(rr); print(f"    {tag:18s}: rel-L2 {rr:.3e}")
    os.makedirs(a.out, exist_ok=True)
    save_json({"model": a.model, "n_params": np_train, "eval": res}, os.path.join(a.out, "eval.json"))


if __name__ == "__main__":
    main()
