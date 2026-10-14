"""Strong geometry-aware neural-operator baselines on the SAME interior-Laplace boundary->interior task
NBIO solves, to answer the reviewer question "are the baselines too simple?".

These are SOTA operators *designed for irregular geometry / point clouds*, in contrast to the existing
black-box baselines (DeepONet / DeepSet / BoundaryTransformer / FNO):

  GNO   (Graph Neural Operator, Li et al. 2020): a kernel-integral message-passing layer over a
        neighborhood graph.  u(x) = (1/|A(x)|) sum_{y in A(x)} k_theta(x, y, f(y)) f(y), with the kernel
        k_theta an MLP on relative coordinates + node features. Source nodes = boundary points with
        features [x,y,nx,ny,g]; output nodes = interior query points. Multi-layer GNO with a query-coord
        lift and a pointwise projection head.

  GINO  (Geometry-Informed Neural Operator, Li et al. 2023): GNO-encode the boundary point cloud onto a
        regular latent grid -> a 2D FNO on the latent grid -> GNO-decode to the interior query points.
        This is the canonical recipe for mapping an irregular geometry through an FFT-based spectral
        operator.

  GeoFNO (Geo-FNO, Li et al. 2022, OPTIONAL third): a learned diffeomorphic deformation maps the point
        cloud to a uniform [0,1]^2 latent grid, a standard 2D FNO acts there, and the field is sampled
        back to the (deformed) query points by bilinear interpolation. Implemented as a FAITHFUL pure-torch
        version (the neuraloperator 2.0.0 release does not ship a standalone Geo-FNO model).

Provenance:
  * GNO and GINO are built on the canonical `neuraloperator` library reference layers/models
    (neuralop.layers.gno_block.GNOBlock, neuralop.models.GINO). The exact installed version is recorded at
    runtime by the training script (results/strong_baselines.json -> "neuralop_version").
  * GeoFNO is a FAITHFUL pure-torch reimplementation (deformation MLP + reference SpectralConv2d).

Interface: every model is `forward(batch) -> (B, n_q)` with the same batch dict the rest of the repo uses
(Xb,Nb,wb,curv,g,Xq,uq). GNO/GINO loop over the batch dimension because each sample has a DIFFERENT
boundary geometry (the library's GINO batches only when geometry is shared across the batch); GeoFNO is
fully vectorized across the batch. Coordinates are affinely normalized into [0,1]^2 (the domain bbox is
~[-1.5,1.5]^2) so the latent grid and the graph radius are well posed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------
# Coordinate normalization: domain bbox ~ [-1.5, 1.5]^2  ->  [0,1]^2 (a fixed, data-derived affine map).
_COORD_LO = -1.6
_COORD_HI = 1.6


def _to_unit(x):
    return (x - _COORD_LO) / (_COORD_HI - _COORD_LO)


# ====================================================================================================
# GNO -- Graph Neural Operator (library neuralop.layers.gno_block.GNOBlock)
# ====================================================================================================
class GNO(nn.Module):
    """Stacked Graph-Neural-Operator kernel-integral layers, boundary point cloud -> interior queries.

    width        : hidden channels carried through the GNO stack / lift / head.
    n_layers     : number of GNOBlock kernel-integral layers.
    radius       : neighborhood radius (in normalized [0,1]^2 units) defining the graph A(x).
    mlp_layers   : hidden layer sizes of the edge kernel MLP k_theta(x,y,f(y)).
    transform    : neuralop IntegralTransform type. 'nonlinear' = (d) int k(x,y,f(y)) f(y) dy -- the most
                   expressive canonical GNO; 'nonlinear_kernelonly' = (c) int k(x,y,f(y)) dy.
    """

    def __init__(self, width=64, n_layers=3, radius=0.08, mlp_layers=(128, 128),
                 transform="nonlinear", region="interior"):
        super().__init__()
        from neuralop.layers.gno_block import GNOBlock

        self.width = width
        self.radius = float(radius)
        # Lift boundary features [x,y,nx,ny,g] (5) and query coords (2) into `width` channels.
        self.b_lift = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.q_lift = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))

        mlp = list(mlp_layers)
        # First GNO layer: boundary nodes (y) -> query nodes (x). Subsequent layers refine on the query
        # graph (query -> query) so deeper kernels can act, like the multi-layer GNO of Li et al. 2020.
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GNOBlock(
                in_channels=width, out_channels=width, coord_dim=2, radius=self.radius,
                transform_type=transform, reduction="mean", pos_embedding_type="transformer",
                channel_mlp_layers=mlp, use_torch_scatter_reduce=False,
                use_open3d_neighbor_search=False))
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def _one(self, Xb, Nb, g, Xq):
        # Xb (nb,2) Nb (nb,2) g (nb,) Xq (nq,2), all already in [0,1]^2.
        f_b = self.b_lift(torch.cat([Xb, Nb, g.unsqueeze(-1)], dim=-1))      # (nb, width)
        h = self.q_lift(Xq)                                                  # (nq, width)
        # layer 0: integrate boundary -> query
        h = h + self.norms[0](self.layers[0](y=Xb, x=Xq, f_y=f_b))
        # deeper layers: query graph self-refinement
        for i in range(1, len(self.layers)):
            h = h + self.norms[i](self.layers[i](y=Xq, x=Xq, f_y=h))
        return self.head(h).squeeze(-1)                                      # (nq,)

    def forward(self, batch):
        Xb, Nb, g, Xq = batch["Xb"], batch["Nb"], batch["g"], batch["Xq"]
        Xb, Xq = _to_unit(Xb), _to_unit(Xq)
        outs = [self._one(Xb[b], Nb[b], g[b], Xq[b]) for b in range(Xb.shape[0])]
        return torch.stack(outs, dim=0)                                      # (B, nq)


# ====================================================================================================
# GINO -- Geometry-Informed Neural Operator (library neuralop.models.GINO)
# ====================================================================================================
class GINO(nn.Module):
    """GNO-encode boundary -> latent [0,1]^2 grid -> 2D FNO -> GNO-decode to interior queries.

    grid          : latent grid resolution (grid x grid).
    fno_modes     : Fourier modes kept per dim in the latent FNO.
    fno_width     : FNO hidden channels.
    fno_layers    : number of FNO blocks.
    in_radius     : encode-GNO radius (boundary -> latent grid), normalized units.
    out_radius    : decode-GNO radius (latent grid -> queries), normalized units.
    """

    def __init__(self, grid=32, fno_modes=16, fno_width=48, fno_layers=4,
                 in_radius=0.12, out_radius=0.12, gno_mlp=(96, 96),
                 in_transform="nonlinear_kernelonly", out_transform="linear", region="interior"):
        super().__init__()
        from neuralop.models import GINO as _GINO

        self.grid_n = int(grid)
        self.net = _GINO(
            in_channels=5, out_channels=1, gno_coord_dim=2,
            in_gno_radius=float(in_radius), out_gno_radius=float(out_radius),
            fno_n_modes=(int(fno_modes), int(fno_modes)), fno_hidden_channels=int(fno_width),
            fno_n_layers=int(fno_layers),
            in_gno_transform_type=in_transform, out_gno_transform_type=out_transform,
            gno_use_open3d=False, gno_use_torch_scatter=False,
            in_gno_channel_mlp_hidden_layers=list(gno_mlp),
            out_gno_channel_mlp_hidden_layers=list(gno_mlp),
            in_gno_pos_embed_type="transformer", out_gno_pos_embed_type="transformer")

    def _latent_grid(self, device, dtype):
        g = torch.linspace(0.0, 1.0, self.grid_n, device=device, dtype=dtype)
        grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1)     # (H,H,2)
        return grid.unsqueeze(0)                                            # (1,H,H,2)

    def forward(self, batch):
        Xb, Nb, g, Xq = batch["Xb"], batch["Nb"], batch["g"], batch["Xq"]
        Xb, Xq = _to_unit(Xb), _to_unit(Xq)
        latent = self._latent_grid(Xb.device, Xb.dtype)
        feats = torch.cat([Xb, Nb, g.unsqueeze(-1)], dim=-1)                # (B,nb,5)
        outs = []
        for b in range(Xb.shape[0]):
            # GINO batches only when geometry is shared; here each sample has its own boundary -> b=1.
            o = self.net(input_geom=Xb[b:b + 1], latent_queries=latent,
                         output_queries=Xq[b:b + 1], x=feats[b:b + 1])      # (1, nq, 1)
            outs.append(o.squeeze(0).squeeze(-1))
        return torch.stack(outs, dim=0)                                     # (B, nq)


# ====================================================================================================
# GeoFNO -- FAITHFUL pure-torch reimplementation (Li et al. 2022, "Fourier Neural Operator with Learned
# Deformations for PDEs on General Geometries"). Deformation MLP -> uniform [0,1]^2 latent grid -> 2D FNO
# -> bilinear sample-back to the (deformed) query points.
# ====================================================================================================
class _SpectralConv2d(nn.Module):
    """Reference truncated-spectral 2D conv (same construction as src/models/fno2d.py)."""

    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
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


class GeoFNO(nn.Module):
    """Geo-FNO: learned deformation phi: physical [0,1]^2 -> uniform latent [0,1]^2.

    The boundary signal (Dirichlet data + normals) is scattered onto the uniform latent grid via a soft
    (Gaussian-kernel) deformation push-forward using phi(Xb); a standard 2D FNO acts on the latent grid;
    the latent field is then pulled back to the query points by bilinear interpolation at phi(Xq). The
    deformation is parametrized by an MLP and learned end-to-end (the defining feature of Geo-FNO).
    """

    def __init__(self, grid=40, modes=16, width=40, n_layers=4, deform_hidden=64,
                 scatter_sigma=0.04, region="interior"):
        super().__init__()
        self.grid_n = int(grid)
        self.scatter_sigma = float(scatter_sigma)
        # Learned deformation phi (a residual MLP on coords): physical -> latent coords in [0,1]^2.
        self.deform = nn.Sequential(
            nn.Linear(2, deform_hidden), nn.GELU(),
            nn.Linear(deform_hidden, deform_hidden), nn.GELU(),
            nn.Linear(deform_hidden, 2))
        # Lift the scattered boundary channels [g, nx, ny, density] (+2 latent-grid coords) -> width.
        self.lift = nn.Conv2d(4 + 2, width, 1)
        self.specs = nn.ModuleList([_SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.act = nn.GELU()
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def _phi(self, X):
        # residual deformation, then squash to [0,1]^2 so it lands on the latent grid.
        return torch.sigmoid(X + self.deform(X))

    def _scatter(self, phi_b, feat_b):
        # phi_b (B,nb,2) in [0,1]; feat_b (B,nb,C). Gaussian push-forward onto (H,H) latent grid.
        B, nb, C = feat_b.shape
        H = self.grid_n
        device = feat_b.device
        g = torch.linspace(0.0, 1.0, H, device=device)
        gy, gx = torch.meshgrid(g, g, indexing="ij")                       # (H,H)
        grid_pts = torch.stack([gx, gy], dim=-1).reshape(1, H * H, 2)      # (1,HH,2)
        d2 = (phi_b.unsqueeze(2) - grid_pts.unsqueeze(1)).pow(2).sum(-1)   # (B,nb,HH)
        w = torch.exp(-d2 / (2.0 * self.scatter_sigma ** 2))               # (B,nb,HH)
        wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-6)                  # (B,1,HH)
        # weighted feature field + a "density" channel (how much boundary mass landed on each cell)
        field = torch.einsum("bnh,bnc->bhc", w, feat_b) / wsum.transpose(1, 2)  # (B,HH,C)
        density = (wsum.transpose(1, 2) / nb)                              # (B,HH,1)
        field = torch.cat([field, density], dim=-1)                       # (B,HH,C+1)
        return field.transpose(1, 2).reshape(B, C + 1, H, H)              # (B,C+1,H,H)

    def _grid_chan(self, B, H, device):
        g = torch.linspace(0.0, 1.0, H, device=device)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([gx, gy])[None].expand(B, -1, -1, -1)          # (B,2,H,H)

    def forward(self, batch):
        Xb, Nb, g, Xq = batch["Xb"], batch["Nb"], batch["g"], batch["Xq"]
        Xb, Xq = _to_unit(Xb), _to_unit(Xq)
        B, H = Xb.shape[0], self.grid_n
        phi_b = self._phi(Xb)                                              # (B,nb,2)
        feat_b = torch.cat([g.unsqueeze(-1), Nb], dim=-1)                  # (B,nb,3): [g,nx,ny]
        field = self._scatter(phi_b, feat_b)                              # (B,4,H,H) (3 feats + density)
        inp = torch.cat([field, self._grid_chan(B, H, Xb.device)], dim=1)  # (B,6,H,H)
        h = self.lift(inp)
        for s, w in zip(self.specs, self.ws):
            h = h + self.act(s(h) + w(h))
        u_grid = self.proj(h)                                             # (B,1,H,H)
        # pull back to query points at phi(Xq) by bilinear interpolation; grid_sample wants [-1,1].
        phi_q = self._phi(Xq)                                             # (B,nq,2) in [0,1]
        samp = (phi_q * 2.0 - 1.0).unsqueeze(1)                           # (B,1,nq,2)
        u_q = F.grid_sample(u_grid, samp, mode="bilinear", align_corners=True)  # (B,1,1,nq)
        return u_q.squeeze(1).squeeze(1)                                  # (B,nq)
