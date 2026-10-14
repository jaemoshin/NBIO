"""Continuous-time parabolic boundary-integral NBIO prototypes.

These modules test the time-dependent extension of NBIO in the setting where the heat equation is represented
by a causal boundary layer potential.  The initial prototype used the heat single-layer formula

    u(x,t) = int_0^t int_Gamma H(x-y,t-s) mu(y,s) ds_y ds,

with the 2D heat kernel H(z,tau) = (4 pi tau)^-1 exp(-|z|^2 / (4 tau)).
The stronger time-dynamic variant below uses the causal heat double layer instead.  Its boundary trace has a
local jump term, giving the second-kind Volterra solve that mirrors elliptic NBIO more closely.

`ParabolicBIENBIO` is the structure-only version: it solves the discrete causal Volterra boundary system for
the density and evaluates the potential at arbitrary query times.  `ContinuousTimeHeatLayer` is a learned
density approximation that keeps the analytic heat potential but replaces the causal solve by an MLP.
`ContinuousTimeDeepSet` is a black-box set baseline for the same `(x,t)` query contract.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mlp(din: int, hidden: int, dout: int, layers: int) -> nn.Sequential:
    body = [nn.Linear(din, hidden), nn.GELU()]
    for _ in range(layers - 1):
        body += [nn.Linear(hidden, hidden), nn.GELU()]
    body += [nn.Linear(hidden, dout)]
    return nn.Sequential(*body)


def heat_kernel_torch(Xt, tq, Xs, ts, eps: float = 1e-8):
    """Return causal 2D heat kernel values.

    Args:
        Xt: target coordinates, `(B,T,2)`.
        tq: target times, `(B,T)`.
        Xs: source coordinates, `(B,S,2)`.
        ts: source times, `(B,S)`.

    Returns:
        `(B,T,S)` matrix with zero entries for non-causal interactions.
    """
    tau = tq[:, :, None] - ts[:, None, :]
    causal = tau > eps
    tau_safe = tau.clamp_min(eps)
    r2 = ((Xt[:, :, None, :] - Xs[:, None, :, :]) ** 2).sum(-1)
    h = torch.exp(-r2 / (4.0 * tau_safe)) / (4.0 * math.pi * tau_safe)
    return torch.where(causal, h, torch.zeros_like(h))


def heat_double_layer_kernel_torch(Xt, tq, Xs, ts, Ns, eps: float = 1e-8):
    """Causal 2D heat double-layer kernel dH/dn_y.

    `dH/dn_y = ((x-y).n_y)/(2 tau) H(x-y,tau)`.  The kernel is off-surface regular; on the boundary the
    jump term is handled explicitly by `ParabolicDoubleLayerNBIO`.
    """
    tau = tq[:, :, None] - ts[:, None, :]
    causal = tau > eps
    tau_safe = tau.clamp_min(eps)
    diff = Xt[:, :, None, :] - Xs[:, None, :, :]
    r2 = (diff ** 2).sum(-1)
    h = torch.exp(-r2 / (4.0 * tau_safe)) / (4.0 * math.pi * tau_safe)
    num = (diff * Ns[:, None, :, :]).sum(-1)
    k = num * h / (2.0 * tau_safe)
    return torch.where(causal, k, torch.zeros_like(k))


def flatten_boundary_time(Xb, values, times):
    """Time-major flattening helper for `(B,Nt,Nb)` values."""
    B, Nt, Nb = values.shape
    Xs = Xb[:, None, :, :].expand(B, Nt, Nb, 2).reshape(B, Nt * Nb, 2)
    ts = times[None, :, None].expand(B, Nt, Nb).reshape(B, Nt * Nb).to(Xb.device)
    vals = values.reshape(B, Nt * Nb)
    return Xs, ts, vals


class ParabolicBIENBIO(nn.Module):
    """Zero-parameter causal heat boundary-integral solve.

    Input batch fields:
      Xb `(B,Nb,2)`, wb `(B,Nb)`, g `(B,Nt,Nb)`, t_obs `(Nt,)`, t_src `(Nt,)`,
      Xq `(B,Q,2)`, tq `(B,Q)`, dt scalar.

    The solve is block lower triangular in time.  Each source time is the midpoint of an interval and each
    boundary observation time is the right endpoint, so the same-time diagonal block has tau=dt/2 rather than
    the singular tau=0 endpoint.
    """

    def __init__(self, ridge: float = 0.0):
        super().__init__()
        self.ridge = ridge

    def solve_density(self, batch):
        Xb, wb, g = batch["Xb"], batch["wb"], batch["g"]
        t_obs, t_src = batch["t_obs"].to(Xb.device), batch["t_src"].to(Xb.device)
        dt = float(batch["dt"])
        B, Nt, Nb = g.shape
        mu_steps = []
        eye = torch.eye(Nb, device=Xb.device, dtype=Xb.dtype).expand(B, Nb, Nb)
        for n in range(Nt):
            rhs = g[:, n, :].clone()
            Xt = Xb
            tq = torch.full((B, Nb), float(t_obs[n]), device=Xb.device, dtype=Xb.dtype)
            for m, mu_m in enumerate(mu_steps):
                ts = torch.full((B, Nb), float(t_src[m]), device=Xb.device, dtype=Xb.dtype)
                K = heat_kernel_torch(Xt, tq, Xb, ts) * (wb[:, None, :] * dt)
                rhs = rhs - torch.bmm(K, mu_m.unsqueeze(-1)).squeeze(-1)
            ts = torch.full((B, Nb), float(t_src[n]), device=Xb.device, dtype=Xb.dtype)
            Kdiag = heat_kernel_torch(Xt, tq, Xb, ts) * (wb[:, None, :] * dt)
            if self.ridge:
                Kdiag = Kdiag + self.ridge * eye
            mu_n = torch.linalg.solve(Kdiag, rhs.unsqueeze(-1)).squeeze(-1)
            mu_steps.append(mu_n)
        return torch.stack(mu_steps, dim=1)

    def evaluate_density(self, batch, mu):
        Xb, wb, Xq, tq = batch["Xb"], batch["wb"], batch["Xq"], batch["tq"]
        t_src = batch["t_src"].to(Xb.device)
        dt = float(batch["dt"])
        Xs, ts, muf = flatten_boundary_time(Xb, mu, t_src)
        weights = wb[:, None, :].expand_as(mu).reshape(mu.shape[0], -1) * dt
        Kq = heat_kernel_torch(Xq, tq, Xs, ts)
        return torch.bmm(Kq, (muf * weights).unsqueeze(-1)).squeeze(-1)

    def forward(self, batch, return_density: bool = False):
        mu = self.solve_density(batch)
        u = self.evaluate_density(batch, mu)
        return (u, mu) if return_density else u


class ParabolicDoubleLayerNBIO(nn.Module):
    """Zero-parameter second-kind causal heat double-layer solve.

    The interior Dirichlet trace is discretized as

        g_n = jump * mu_n + sum_{m<=n} dt * D(t_n,t_{m+1/2}) mu_m,

    with `jump=-1/2` for the interior limit.  This is a block-lower-triangular Volterra system whose current
    block is second-kind, in contrast to the first-kind single-layer prototype.
    """

    def __init__(self, jump: float = -0.5, ridge: float = 0.0):
        super().__init__()
        self.jump = float(jump)
        self.ridge = float(ridge)

    def _block(self, Xb, Nb, wb, tq_scalar, ts_scalar, dt):
        B, Nbnd, _ = Xb.shape
        tq = torch.full((B, Nbnd), float(tq_scalar), device=Xb.device, dtype=Xb.dtype)
        ts = torch.full((B, Nbnd), float(ts_scalar), device=Xb.device, dtype=Xb.dtype)
        return heat_double_layer_kernel_torch(Xb, tq, Xb, ts, Nb) * (wb[:, None, :] * dt)

    def solve_density(self, batch):
        Xb, Nb, wb, g = batch["Xb"], batch["Nb"], batch["wb"], batch["g"]
        t_obs, t_src = batch["t_obs"].to(Xb.device), batch["t_src"].to(Xb.device)
        dt = float(batch["dt"])
        B, Nt, Nbnd = g.shape
        eye = torch.eye(Nbnd, device=Xb.device, dtype=Xb.dtype).expand(B, Nbnd, Nbnd)
        mu_steps = []
        conds = []
        for n in range(Nt):
            rhs = g[:, n, :].clone()
            for m, mu_m in enumerate(mu_steps):
                K = self._block(Xb, Nb, wb, t_obs[n], t_src[m], dt)
                rhs = rhs - torch.bmm(K, mu_m.unsqueeze(-1)).squeeze(-1)
            A = self.jump * eye + self._block(Xb, Nb, wb, t_obs[n], t_src[n], dt)
            if self.ridge:
                A = A + self.ridge * eye
            conds.append(torch.linalg.cond(A).detach())
            mu_n = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
            mu_steps.append(mu_n)
        mu = torch.stack(mu_steps, dim=1)
        self._last_cond = torch.stack(conds, dim=1).mean().detach()
        return mu

    def evaluate_density(self, batch, mu):
        Xb, Nb, wb, Xq, tq = batch["Xb"], batch["Nb"], batch["wb"], batch["Xq"], batch["tq"]
        t_src = batch["t_src"].to(Xb.device)
        dt = float(batch["dt"])
        Xs, ts, muf = flatten_boundary_time(Xb, mu, t_src)
        Ns = Nb[:, None, :, :].expand(mu.shape[0], mu.shape[1], mu.shape[2], 2).reshape(mu.shape[0], -1, 2)
        weights = wb[:, None, :].expand_as(mu).reshape(mu.shape[0], -1) * dt
        Kq = heat_double_layer_kernel_torch(Xq, tq, Xs, ts, Ns)
        return torch.bmm(Kq, (muf * weights).unsqueeze(-1)).squeeze(-1)

    def forward(self, batch, return_density: bool = False):
        mu = self.solve_density(batch)
        u = self.evaluate_density(batch, mu)
        return (u, mu) if return_density else u


class ContinuousTimeHeatLayer(nn.Module):
    """Learned continuous-time heat-layer model.

    This keeps the analytic heat potential and learns a boundary-time density from geometry and observed
    boundary values.  It can be queried at arbitrary `(x,t)`, but unlike `ParabolicBIENBIO` it does not
    explicitly invert the causal boundary system.
    """

    def __init__(self, hidden: int = 128, layers: int = 4):
        super().__init__()
        # token features: Xb(2), normal(2), time, sin/cos time, g, cumulative g, global mean/std of g
        self.density = _mlp(11, hidden, 1, layers)

    def forward(self, batch, return_density: bool = False):
        Xb, Nb, wb, g, Xq, tq = batch["Xb"], batch["Nb"], batch["wb"], batch["g"], batch["Xq"], batch["tq"]
        t_src = batch["t_src"].to(Xb.device)
        dt = float(batch["dt"])
        B, Nt, N = g.shape
        T = float(batch["T"])
        t = (t_src / max(T, 1e-8))[None, :, None].expand(B, Nt, N)
        sin_t = torch.sin(2 * math.pi * t)
        cos_t = torch.cos(2 * math.pi * t)
        X = Xb[:, None, :, :].expand(B, Nt, N, 2)
        Nrm = Nb[:, None, :, :].expand(B, Nt, N, 2)
        cum_g = torch.cumsum(g, dim=1) / torch.arange(1, Nt + 1, device=g.device, dtype=g.dtype)[None, :, None]
        mean_g = g.mean(dim=(1, 2), keepdim=True).expand(B, Nt, N)
        std_g = g.std(dim=(1, 2), keepdim=True).clamp_min(1e-6).expand(B, Nt, N)
        feats = torch.cat([
            X, Nrm, t[..., None], sin_t[..., None], cos_t[..., None],
            g[..., None], cum_g[..., None], mean_g[..., None], std_g[..., None],
        ], dim=-1)
        mu = self.density(feats).squeeze(-1)
        Xs, ts, muf = flatten_boundary_time(Xb, mu, t_src)
        weights = wb[:, None, :].expand_as(mu).reshape(B, -1) * dt
        Kq = heat_kernel_torch(Xq, tq, Xs, ts)
        u = torch.bmm(Kq, (muf * weights).unsqueeze(-1)).squeeze(-1)
        return (u, mu) if return_density else u


class ContinuousTimeDoubleLayer(nn.Module):
    """Learned density approximation with an analytic causal heat double-layer potential."""

    def __init__(self, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.density = _mlp(11, hidden, 1, layers)

    def forward(self, batch, return_density: bool = False):
        Xb = batch["Xb"]
        Nb = batch["Nb"]
        wb = batch["wb"]
        g = batch["g"]
        Xq = batch["Xq"]
        tq = batch["tq"]
        t_src = batch["t_src"].to(Xb.device)
        dt = float(batch["dt"])
        B, Nt, N = g.shape
        T = float(batch["T"])
        t = (t_src / max(T, 1e-8))[None, :, None].expand(B, Nt, N)
        sin_t = torch.sin(2 * math.pi * t)
        cos_t = torch.cos(2 * math.pi * t)
        X = Xb[:, None, :, :].expand(B, Nt, N, 2)
        Nrm = Nb[:, None, :, :].expand(B, Nt, N, 2)
        cum_g = torch.cumsum(g, dim=1) / torch.arange(1, Nt + 1, device=g.device, dtype=g.dtype)[None, :, None]
        mean_g = g.mean(dim=(1, 2), keepdim=True).expand(B, Nt, N)
        std_g = g.std(dim=(1, 2), keepdim=True).clamp_min(1e-6).expand(B, Nt, N)
        feats = torch.cat([
            X, Nrm, t[..., None], sin_t[..., None], cos_t[..., None],
            g[..., None], cum_g[..., None], mean_g[..., None], std_g[..., None],
        ], dim=-1)
        mu = self.density(feats).squeeze(-1)
        Xs, ts, muf = flatten_boundary_time(Xb, mu, t_src)
        Ns = Nb[:, None, :, :].expand(B, Nt, N, 2).reshape(B, Nt * N, 2)
        weights = wb[:, None, :].expand_as(mu).reshape(B, -1) * dt
        Kq = heat_double_layer_kernel_torch(Xq, tq, Xs, ts, Ns)
        u = torch.bmm(Kq, (muf * weights).unsqueeze(-1)).squeeze(-1)
        return (u, mu) if return_density else u


class ContinuousTimeDeepSet(nn.Module):
    """Black-box continuous-time boundary-set baseline.

    Boundary-time tokens are encoded into a global code; `(x,t)` query coordinates are decoded from that
    code.  It has the same arbitrary-time output contract but no causal heat-kernel structure.
    """

    def __init__(self, hidden: int = 192, code: int = 128, enc_layers: int = 3, dec_layers: int = 4):
        super().__init__()
        self.enc = _mlp(8, hidden, code, enc_layers)
        self.dec = _mlp(code + 3, hidden, 1, dec_layers)

    def forward(self, batch):
        Xb, Nb, g, Xq, tq = batch["Xb"], batch["Nb"], batch["g"], batch["Xq"], batch["tq"]
        t_obs = batch["t_obs"].to(Xb.device)
        B, Nt, N = g.shape
        T = float(batch["T"])
        t = (t_obs / max(T, 1e-8))[None, :, None].expand(B, Nt, N)
        X = Xb[:, None, :, :].expand(B, Nt, N, 2)
        Nrm = Nb[:, None, :, :].expand(B, Nt, N, 2)
        feats = torch.cat([
            X, Nrm, t[..., None], torch.sin(2 * math.pi * t)[..., None],
            torch.cos(2 * math.pi * t)[..., None], g[..., None],
        ], dim=-1).reshape(B, Nt * N, 8)
        code = self.enc(feats).mean(dim=1)
        tq_norm = (tq / max(T, 1e-8)).unsqueeze(-1)
        q = torch.cat([Xq, tq_norm, code[:, None, :].expand(-1, Xq.shape[1], -1)], dim=-1)
        return self.dec(q).squeeze(-1)
