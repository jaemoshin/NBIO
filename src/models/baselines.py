"""Learned-operator baselines, same (boundary data + query coords) interface as NBIO.

DeepONetBoundary : branch on fixed-count boundary sensors [g, Xb, Nb]; trunk on query coords. The canonical
                   operator-learning baseline. Fixed sensor count -> cannot change boundary resolution.
DeepSetBoundary  : permutation-invariant per-node encoder over boundary nodes -> global code; decode at
                   query coords. Geometry-aware, handles ANY boundary resolution (set-based), but has NO
                   boundary-integral / Green's structure -> the head-to-head test of whether M1+M2 matter.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(din, dh, dout, n):
    layers = [nn.Linear(din, dh), nn.GELU()]
    for _ in range(n - 1):
        layers += [nn.Linear(dh, dh), nn.GELU()]
    layers += [nn.Linear(dh, dout)]
    return nn.Sequential(*layers)


class _CrossAttentionBlock(nn.Module):
    """Query tokens attend to boundary/interface tokens; no Green/BIE structure."""

    def __init__(self, width=128, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.qnorm = nn.LayerNorm(width)
        self.knorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, q, kv):
        h, _ = self.attn(self.qnorm(q), self.knorm(kv), self.knorm(kv), need_weights=False)
        q = q + h
        return q + self.ff(q)


class BoundaryTransformer(nn.Module):
    """Geometry-aware black-box baseline: boundary tokens -> query cross-attention -> u(x_q).

    This is a stronger comparator than DeepSet/DeepONet because each query can attend to all boundary
    points at the current resolution. It is still a black-box neural operator: no analytic kernel,
    no second-kind solve, no quadrature/potential structure.
    """

    def __init__(self, width=128, n_layers=4, heads=4, token_layers=2, region="interior"):
        super().__init__()
        self.btok = _mlp(5, width, width, token_layers)       # [Xb(2), Nb(2), g]
        self.qtok = _mlp(2, width, width, token_layers)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(width, heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, batch):
        b = self.btok(torch.cat([batch["Xb"], batch["Nb"], batch["g"].unsqueeze(-1)], dim=-1))
        q = self.qtok(batch["Xq"])
        for blk in self.blocks:
            q = blk(q, b)
        return self.head(q).squeeze(-1)


class KappaBoundaryTransformer(nn.Module):
    """Amortized boundary transformer conditioned on the screening parameter."""

    def __init__(self, width=128, n_layers=4, heads=4, token_layers=2,
                 kappa_scale=4.0, normalize_load=False,
                 include_weights=False, region="interior"):
        super().__init__()
        self.kappa_scale = float(kappa_scale)
        self.normalize_load = bool(normalize_load)
        self.include_weights = bool(include_weights)
        self.btok = _mlp(
            6 + int(self.include_weights), width, width, token_layers,
        )
        self.qtok = _mlp(3, width, width, token_layers)
        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(width, heads) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, batch):
        if "kappa" not in batch:
            raise KeyError("KappaBoundaryTransformer requires batch['kappa']")
        if self.include_weights and "wb" not in batch:
            raise KeyError("include_weights=True requires batch['wb']")
        if self.normalize_load:
            load_scale = batch["g"].square().mean(
                dim=-1, keepdim=True,
            ).sqrt().clamp_min(1e-8)
            boundary_values = batch["g"] / load_scale
        else:
            load_scale = None
            boundary_values = batch["g"]
        parameter = batch["kappa"].reshape(-1, 1, 1) / self.kappa_scale
        boundary_parameter = parameter.expand(-1, batch["Xb"].shape[1], -1)
        query_parameter = parameter.expand(-1, batch["Xq"].shape[1], -1)
        boundary_features = [
            batch["Xb"],
            batch["Nb"],
            boundary_values.unsqueeze(-1),
            boundary_parameter,
        ]
        if self.include_weights:
            normalized_weights = (
                batch["wb"] / batch["wb"].mean(dim=-1, keepdim=True)
            )
            boundary_features.append(normalized_weights.unsqueeze(-1))
        boundary = self.btok(torch.cat(boundary_features, dim=-1))
        query = self.qtok(torch.cat((batch["Xq"], query_parameter), dim=-1))
        for block in self.blocks:
            query = block(query, boundary)
        output = self.head(query).squeeze(-1)
        return output if load_scale is None else output * load_scale


class TransmissionTransformer(nn.Module):
    """Cross-attention baseline for transmission/interface problems.

    Tokens include outer Dirichlet values, interface geometry, curve identity, and log contrast. The model
    can handle variable inclusion counts and boundary resolution because it attends over the token set.
    """

    def __init__(self, width=128, n_layers=4, heads=4, token_layers=2):
        super().__init__()
        self.btok = _mlp(7, width, width, token_layers)
        self.qtok = _mlp(2, width, width, token_layers)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(width, heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def _nodes(self, b):
        Xo, No, g = b["Xb_out"], b["Nb_out"], b["g"]
        B, nbo = g.shape
        outer = torch.cat([Xo, No, g.unsqueeze(-1),
                           torch.ones(B, nbo, 1, device=g.device), torch.zeros(B, nbo, 1, device=g.device)], -1)
        Xi, Ni, a = b["Xb_in"], b["Nb_in"], b["a_in"]
        Bn, N, nbi, _ = Xi.shape
        logc = torch.log(a).clamp(-4, 4)[:, :, None, None].expand(Bn, N, nbi, 1)
        incl = torch.cat([Xi, Ni, torch.zeros(Bn, N, nbi, 1, device=g.device),
                          torch.zeros(Bn, N, nbi, 1, device=g.device), logc], -1).reshape(Bn, N * nbi, 7)
        return torch.cat([outer, incl], dim=1)

    def forward(self, b):
        kv = self.btok(self._nodes(b))
        q = self.qtok(b["Xq"])
        for blk in self.blocks:
            q = blk(q, kv)
        return self.head(q).squeeze(-1)


class DeepONetBoundary(nn.Module):
    def __init__(self, n_sensors=128, p=128, hidden=256, n_layers=4, region="interior"):
        super().__init__()
        self.n_sensors = n_sensors
        self.branch = _mlp(5 * n_sensors, hidden, p, n_layers)   # [g,Xb(2),Nb(2)] per sensor
        self.trunk = _mlp(2, hidden, p, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def _sensor_vec(self, batch):
        Xb, Nb, g = batch["Xb"], batch["Nb"], batch["g"]
        nb = Xb.shape[1]
        if nb != self.n_sensors:    # resample boundary signals to the fixed sensor count (linear in index)
            idx = torch.linspace(0, nb - 1, self.n_sensors, device=Xb.device).round().long()
            Xb, Nb, g = Xb[:, idx], Nb[:, idx], g[:, idx]
        return torch.cat([g.unsqueeze(-1), Xb, Nb], dim=-1).flatten(1)   # (B, 5*n_sensors)

    def forward(self, batch):
        b = self.branch(self._sensor_vec(batch))            # (B,p)
        t = self.trunk(batch["Xq"])                         # (B,Q,p)
        return torch.einsum("bp,bqp->bq", b, t) + self.bias


def _iface_tokens(b):
    """Per-curve tokens for the interface benchmark: [x,y,is_outer,g,nx,ny,alpha,beta,logc] (9 dims)."""
    Xo, g = b["Xb_out"], b["g"]; B, nbo = g.shape
    Xi, Ni, al, be = b["Xb_in"], b["Nb_in"], b["alpha"], b["beta"]; nbi = al.shape[1]
    lc = b["logc"]; z = lambda *s: torch.zeros(*s, device=g.device)
    outer = torch.cat([Xo, torch.ones(B, nbo, 1, device=g.device), g[..., None], z(B, nbo, 2),
                       z(B, nbo, 2), lc[:, None, :].expand(B, nbo, 1)], -1)
    inner = torch.cat([Xi, z(B, nbi, 1), z(B, nbi, 1), Ni, al[..., None], be[..., None],
                       lc[:, None, :].expand(B, nbi, 1)], -1)
    return torch.cat([outer, inner], 1)                       # (B, nbo+nbi, 9)


class InterfaceDeepSet(nn.Module):
    def __init__(self, hidden=256, p=128, enc_layers=3, dec_layers=4):
        super().__init__()
        self.enc = _mlp(9, hidden, p, enc_layers); self.dec = _mlp(2 + p, hidden, 1, dec_layers)

    def forward(self, b):
        code = self.enc(_iface_tokens(b)).mean(1)
        Q = b["Xq"].shape[1]
        return self.dec(torch.cat([b["Xq"], code[:, None, :].expand(-1, Q, -1)], -1)).squeeze(-1)


class InterfaceTransformer(nn.Module):
    """Strong black-box: query points cross-attend to all boundary/interface tokens."""
    def __init__(self, d=128, heads=4, layers=2):
        super().__init__()
        self.tok = _mlp(9, d, d, 2); self.q = _mlp(2, d, d, 2)
        self.blocks = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = _mlp(d, d, 1, 2)

    def forward(self, b):
        ctx = self.tok(_iface_tokens(b)); h = self.q(b["Xq"])
        for att, ln in zip(self.blocks, self.ln):
            a, _ = att(h, ctx, ctx); h = ln(h + a)
        return self.head(h).squeeze(-1)


def _src_tokens(b):
    """Interface tokens + the volume source f sampled at each node: 10 dims."""
    base = _iface_tokens(b)
    fcat = torch.cat([b["f_out"], b["f_in"]], dim=1)[..., None]
    return torch.cat([base, fcat], -1)


class InterfaceSourceDeepSet(nn.Module):
    def __init__(self, hidden=256, p=128, enc_layers=3, dec_layers=4):
        super().__init__()
        self.enc = _mlp(10, hidden, p, enc_layers); self.dec = _mlp(3 + p, hidden, 1, dec_layers)

    def forward(self, b):
        code = self.enc(_src_tokens(b)).mean(1); Q = b["Xq"].shape[1]
        return self.dec(torch.cat([b["Xq"], b["fq"][..., None], code[:, None, :].expand(-1, Q, -1)], -1)).squeeze(-1)


class InterfaceSourceTransformer(nn.Module):
    def __init__(self, d=128, heads=4, layers=3):
        super().__init__()
        self.tok = _mlp(10, d, d, 2); self.q = _mlp(3, d, d, 2)
        self.blocks = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = _mlp(d, d, 1, 2)

    def forward(self, b):
        ctx = self.tok(_src_tokens(b)); h = self.q(torch.cat([b["Xq"], b["fq"][..., None]], -1))
        for att, ln in zip(self.blocks, self.ln):
            a, _ = att(h, ctx, ctx); h = ln(h + a)
        return self.head(h).squeeze(-1)


class DeepSetTransmission(nn.Module):
    """Black-box baseline for transmission: permutation-invariant encoder over ALL boundary/interface nodes
    (with curve-type + log-contrast features) -> global code -> decode at query coords. Handles any
    #inclusions, but has NO boundary-integral / Green's structure."""

    def __init__(self, hidden=128, p=128, enc_layers=3, dec_layers=4):
        super().__init__()
        self.enc = _mlp(7, hidden, p, enc_layers)       # [x,y,nx,ny,val,is_outer,log_contrast]
        self.dec = _mlp(2 + p, hidden, 1, dec_layers)

    def _nodes(self, b):
        Xo, No, g = b["Xb_out"], b["Nb_out"], b["g"]
        B, nbo = g.shape
        outer = torch.cat([Xo, No, g.unsqueeze(-1),
                           torch.ones(B, nbo, 1, device=g.device), torch.zeros(B, nbo, 1, device=g.device)], -1)
        Xi, Ni, a = b["Xb_in"], b["Nb_in"], b["a_in"]   # (B,N,nbi,2),(B,N,nbi,2),(B,N)
        Bn, N, nbi, _ = Xi.shape
        logc = torch.log(a).clamp(-4, 4)[:, :, None, None].expand(Bn, N, nbi, 1)
        incl = torch.cat([Xi, Ni, torch.zeros(Bn, N, nbi, 1, device=g.device),
                          torch.zeros(Bn, N, nbi, 1, device=g.device), logc], -1).reshape(Bn, N * nbi, 7)
        return torch.cat([outer, incl], dim=1)          # (B, nbo+N*nbi, 7)

    def forward(self, b):
        code = self.enc(self._nodes(b)).mean(dim=1)     # (B,p)
        Xq = b["Xq"]; Q = Xq.shape[1]
        return self.dec(torch.cat([Xq, code[:, None, :].expand(-1, Q, -1)], -1)).squeeze(-1)


class DeepSetBoundary(nn.Module):
    def __init__(self, hidden=128, p=128, enc_layers=3, dec_layers=4, region="interior"):
        super().__init__()
        self.enc = _mlp(5, hidden, p, enc_layers)           # per node [Xb(2),Nb(2),g]
        self.dec = _mlp(2 + p, hidden, 1, dec_layers)       # [Xq, global code]

    def forward(self, batch):
        Xb, Nb, g, Xq = batch["Xb"], batch["Nb"], batch["g"], batch["Xq"]
        nodes = torch.cat([Xb, Nb, g.unsqueeze(-1)], dim=-1)    # (B,n_b,5)
        code = self.enc(nodes).mean(dim=1)                      # (B,p) permutation-invariant
        Q = Xq.shape[1]
        dec_in = torch.cat([Xq, code[:, None, :].expand(-1, Q, -1)], dim=-1)
        return self.dec(dec_in).squeeze(-1)                    # (B,Q)
