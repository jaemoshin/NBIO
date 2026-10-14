"""Compact periodic 2D FNO + Green-preconditioned FNO for Stage 2 (variable-coefficient elliptic).

fno         : black-box  [a,f,x,y] -> u.
green_fno   : u = u_lap + correction([a,f,u_lap]) where u_lap is the KNOWN constant-coefficient (FFT) solve
              -- the structure-informed prior; the FNO learns the variable-coefficient correction the prior
              cannot capture. (Stage-2 'learned kernel necessary' test; u_lap alone = wrong fixed kernel.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, 2))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, 2))

    def _mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(w))

    def forward(self, x):
        B, C, H, W = x.shape
        m = min(self.modes, H // 2, W // 2 + 1)
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = self._mul(xf[:, :, :m, :m], self.w1[:, :, :m, :m])
        out[:, :, -m:, :m] = self._mul(xf[:, :, -m:, :m], self.w2[:, :, :m, :m])
        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, in_ch=4, width=32, modes=16, n_layers=4, use_prior=False, init_corr=1.0):
        super().__init__()
        self.use_prior = use_prior
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.specs = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
        self.act = nn.GELU()
        self.corr_scale = init_corr
        if use_prior:
            with torch.no_grad():
                self.proj[-1].weight.mul_(0.1); self.proj[-1].bias.zero_()

    def _grid(self, B, H, W, device):
        ys = torch.linspace(0, 1, H, device=device); xs = torch.linspace(0, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy])[None].expand(B, -1, -1, -1)

    def forward(self, batch):
        a, f = batch["a"], batch["f"]                       # (B,H,W)
        B, H, W = a.shape
        grid = self._grid(B, H, W, a.device)
        if self.use_prior:
            ulap = batch["ulap"]
            inp = torch.stack([a, f, ulap], dim=1)
        else:
            inp = torch.cat([torch.stack([a, f], dim=1), grid], dim=1)
        h = self.lift(inp)
        for s, w in zip(self.specs, self.ws):
            h = h + self.act(s(h) + w(h))
        corr = self.proj(h).squeeze(1)
        return batch["ulap"] + self.corr_scale * corr if self.use_prior else corr
