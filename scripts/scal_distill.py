#!/usr/bin/env python
"""Method 6 -- neural compression of the inverse operator (distillation).

NBIO's per-sample cost is the O(n^2) assemble + O(n^3) dense solve of  A mu = g  for the boundary
density mu, followed by an O(Q*n) off-surface potential evaluation.  This script tests whether we can
AMORTIZE the solve: train a network to map the boundary Dirichlet data g (with boundary geometry as
features) DIRECTLY to the boundary density mu, then recover the interior solution with the KNOWN
double-layer potential (no learned correction, no linear solve at inference).

    g (+ Xb, Nb, curv)  --[net]-->  mu_hat        (amortized; no solve)
    u(x_q) = sum_j Dkernel(x_q, Xb_j, Nb_j) * wb_j * mu_hat_j     (known interior double layer)

This trades accuracy for inference speed.  We quantify both:
  * interior rel-L2 of the reconstructed u  vs the exact uq,
  * rel-L2 of the density itself (mu_hat vs the classical BEM mu),
  * whether density error or the potential evaluation dominates,
  * ms/sample of  distill (net + potential)  vs  full assembled-solve NBIO,  at n_b in {128,256,512}.

Self-contained.  Imports existing functions only; creates no shared state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.utils.common import set_seed, get_device, configure_torch, count_params
from src.data.loader import BIEDataset, to_device
from src.data.bie import (sample_coeffs, boundary, sample_charges, eval_u, sample_queries,
                          bem_solve, _double_layer_kernel)
from src.models import build_model

NBIO_REF_REL_L2 = 4.787004468198575e-07   # results/nbio_s0/eval.json -> eval.interior_test.rel_l2


# ----------------------------- distillation model -----------------------------
class DistillDensityNet(nn.Module):
    """Per-node DeepSet-style encoder over the n_b boundary nodes:  (g, Xb, Nb, curv) -> mu (n_b,).

    g -> mu is a smooth boundary-to-boundary map (mu = A^{-1} g for the 2nd-kind operator A).  Each node
    gets a local feature (Dirichlet datum + position + outward normal + curvature); a shared MLP encodes
    it; a global mean/max pool over the boundary supplies the nonlocal context that the inverse operator
    needs (A^{-1} couples all nodes); a shared decoder MLP then reads (local, context) -> scalar density.
    Permutation-equivariant over nodes and resolution-agnostic (works for any n_b)."""

    def __init__(self, hidden=128, enc_layers=3, dec_layers=3):
        super().__init__()
        din = 1 + 2 + 2 + 1   # g, Xb(2), Nb(2), curv
        enc = [nn.Linear(din, hidden), nn.GELU()]
        for _ in range(enc_layers - 1):
            enc += [nn.Linear(hidden, hidden), nn.GELU()]
        self.enc = nn.Sequential(*enc)
        # decoder reads node embedding concatenated with mean+max pooled context
        ddin = hidden * 3
        dec = [nn.Linear(ddin, hidden), nn.GELU()]
        for _ in range(dec_layers - 1):
            dec += [nn.Linear(hidden, hidden), nn.GELU()]
        dec += [nn.Linear(hidden, 1)]
        self.dec = nn.Sequential(*dec)

    def forward(self, batch):
        Xb, Nb, curv, g = batch["Xb"], batch["Nb"], batch["curv"], batch["g"]   # (B,n,2),(B,n,2),(B,n),(B,n)
        feat = torch.cat([g.unsqueeze(-1), Xb, Nb, curv.unsqueeze(-1)], dim=-1)  # (B,n,6)
        h = self.enc(feat)                                                       # (B,n,H)
        ctx = torch.cat([h.mean(dim=1, keepdim=True).expand_as(h),
                         h.max(dim=1, keepdim=True).values.expand_as(h)], dim=-1)
        mu = self.dec(torch.cat([h, ctx], dim=-1)).squeeze(-1)                   # (B,n)
        return mu


# ----------------------------- known double-layer potential (torch, interior) -----------------------------
def double_layer_eval_torch(Xq, Xb, Nb, wb, mu, eps=1e-9):
    """u(x_q) = sum_j k(x_q, y_j) wb_j mu_j,  k = -1/(2pi) (y-x).n_y / |y-x|^2  (known interior double layer).
    Matches src.data.bie._double_layer_kernel exactly; no learned correction, no solve.  (B,Q)."""
    diff = Xb[:, None, :, :] - Xq[:, :, None, :]          # (B,Q,n,2) = y - x
    r2 = (diff ** 2).sum(-1)                              # (B,Q,n)
    num = (diff * Nb[:, None, :, :]).sum(-1)              # (y-x).n_y
    k = (-1.0 / (2 * np.pi)) * num / r2.clamp_min(eps)    # (B,Q,n)
    return (k * wb[:, None, :] * mu[:, None, :]).sum(-1)  # (B,Q)


def rel_l2_rowwise(pred, target, dim=-1):
    num = torch.linalg.vector_norm(pred - target, dim=dim)
    den = torch.linalg.vector_norm(target, dim=dim).clamp_min(1e-12)
    return (num / den).mean()


# ----------------------------- training -----------------------------
def train_distill(model, train_ds, val_ds, device, epochs, lr, batch_size):
    from torch.utils.data import DataLoader
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 1e9; best_state = None
    for ep in range(epochs):
        model.train()
        for batch in tl:
            batch = to_device(batch, device)
            opt.zero_grad()
            mu_hat = model(batch)
            # rel-L2 on the density (scale-invariant; mu magnitude varies across samples)
            loss = rel_l2_rowwise(mu_hat, batch["mu"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval(); vnum = 0.0; vn = 0
        with torch.no_grad():
            for batch in vl:
                batch = to_device(batch, device)
                vnum += rel_l2_rowwise(model(batch), batch["mu"]).item() * batch["mu"].shape[0]
                vn += batch["mu"].shape[0]
        vr = vnum / max(vn, 1)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[distill] epoch {ep+1:3d}/{epochs}  val density rel-L2 = {vr:.4e}", flush=True)
        if vr < best:
            best = vr; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


# ----------------------------- evaluation -----------------------------
@torch.no_grad()
def eval_distill(model, test_ds, device, batch_size=32):
    from torch.utils.data import DataLoader
    tl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    u_num = u_n = 0.0, 0
    u_acc = 0.0; mu_acc = 0.0; n = 0
    # also: rel-L2 of u reconstructed from the EXACT classical mu (isolates the potential-eval error
    # floor -- how much accuracy the known double-layer evaluation itself loses, independent of the net)
    u_floor_acc = 0.0
    for batch in tl:
        batch = to_device(batch, device)
        b = batch["uq"].shape[0]
        mu_hat = model(batch)
        u_hat = double_layer_eval_torch(batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"], mu_hat)
        u_acc += rel_l2_rowwise(u_hat, batch["uq"]).item() * b
        mu_acc += rel_l2_rowwise(mu_hat, batch["mu"]).item() * b
        u_floor = double_layer_eval_torch(batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"], batch["mu"])
        u_floor_acc += rel_l2_rowwise(u_floor, batch["uq"]).item() * b
        n += b
    return {"distill_interior_rel_l2": u_acc / n,
            "density_rel_l2": mu_acc / n,
            "eval_floor_rel_l2_from_exact_mu": u_floor_acc / n}


@torch.no_grad()
def nbio_interior_rel_l2(nbio, test_ds, device, batch_size=32):
    from torch.utils.data import DataLoader
    tl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    nbio.eval(); acc = 0.0; n = 0
    for batch in tl:
        batch = to_device(batch, device)
        u = nbio(batch)
        acc += rel_l2_rowwise(u, batch["uq"]).item() * batch["uq"].shape[0]
        n += batch["uq"].shape[0]
    return acc / n


# ----------------------------- microbenchmark -----------------------------
def _synth_batch(n_b, B, device, n_q=256, seed=1234):
    """Build a B-sample batch at boundary resolution n_b from the SAME data-generating process (bie.py)."""
    rng = np.random.default_rng(seed)
    Xb, Nb, wb, curv, g, Xq = [], [], [], [], [], []
    for _ in range(B):
        coeffs = sample_coeffs(rng)
        bd = boundary(coeffs, n_b)
        ch = sample_charges(rng, "interior", coeffs)
        gi = eval_u(bd["X"], ch)
        Xq_i = sample_queries(rng, coeffs, "interior", n_q)
        Xb.append(bd["X"]); Nb.append(bd["normal"]); wb.append(bd["w"])
        curv.append(bd["curv"]); g.append(gi); Xq.append(Xq_i)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    return {"Xb": t(Xb), "Nb": t(Nb), "wb": t(wb), "curv": t(curv), "g": t(g), "Xq": t(Xq)}


@torch.no_grad()
def _time_ms_per_sample(fn, batch, device, warmup=5, iters=20):
    B = batch["g"].shape[0]
    for _ in range(warmup):
        fn(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / (iters * B)


def microbenchmark(distill, nbio, device, n_bs=(128, 256, 512), B=16):
    distill.eval(); nbio.eval()

    def distill_fwd(batch):
        mu_hat = distill(batch)
        return double_layer_eval_torch(batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"], mu_hat)

    out = {"distill": {}, "nbio": {}}
    for n_b in n_bs:
        batch = _synth_batch(n_b, B, device)
        out["distill"][str(n_b)] = _time_ms_per_sample(distill_fwd, batch, device)
        out["nbio"][str(n_b)] = _time_ms_per_sample(lambda b: nbio(b), batch, device)
        print(f"[bench] n_b={n_b:4d}  distill={out['distill'][str(n_b)]:.4f} ms  "
              f"nbio={out['nbio'][str(n_b)]:.4f} ms  speedup x{out['nbio'][str(n_b)]/out['distill'][str(n_b)]:.2f}",
              flush=True)
    return out


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/scal_distill.json")
    args = ap.parse_args()

    set_seed(args.seed); configure_torch()
    device = get_device(True)
    print(f"[distill] device={device}", flush=True)

    train_ds = BIEDataset(os.path.join(args.data_dir, "interior_train.pt"))
    val_ds = BIEDataset(os.path.join(args.data_dir, "interior_val.pt"))
    test_ds = BIEDataset(os.path.join(args.data_dir, "interior_test.pt"))
    assert "mu" in train_ds.keys, "interior_train.pt must contain the classical density 'mu'"
    print(f"[distill] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  mu target present", flush=True)

    model = DistillDensityNet(hidden=args.hidden).to(device)
    print(f"[distill] params={count_params(model)}", flush=True)
    best_val = train_distill(model, train_ds, val_ds, device, args.epochs, args.lr, args.batch_size)
    print(f"[distill] best val density rel-L2 = {best_val:.4e}", flush=True)

    ev = eval_distill(model, test_ds, device)
    print(f"[distill] interior u rel-L2     = {ev['distill_interior_rel_l2']:.4e}", flush=True)
    print(f"[distill] density mu rel-L2     = {ev['density_rel_l2']:.4e}", flush=True)
    print(f"[distill] eval floor (exact mu) = {ev['eval_floor_rel_l2_from_exact_mu']:.4e}", flush=True)

    # full assembled-solve NBIO (the thing we are amortizing away)
    nbio_ckpt = torch.load("results/nbio_s0/best.pt", map_location="cpu", weights_only=False)
    nbio = build_model(nbio_ckpt["config"])
    nbio.load_state_dict(nbio_ckpt["state_dict"])
    nbio = nbio.to(device).eval()
    nbio_rel_l2 = nbio_interior_rel_l2(nbio, test_ds, device)
    print(f"[distill] NBIO interior rel-L2  = {nbio_rel_l2:.4e}  (ref {NBIO_REF_REL_L2:.4e})", flush=True)

    bench = microbenchmark(model, nbio, device)

    result = {
        "distill_interior_rel_l2": ev["distill_interior_rel_l2"],
        "density_rel_l2": ev["density_rel_l2"],
        "eval_floor_rel_l2_from_exact_mu": ev["eval_floor_rel_l2_from_exact_mu"],
        "best_val_density_rel_l2": best_val,
        "nbio_interior_rel_l2": nbio_rel_l2,
        "nbio_interior_rel_l2_ref": NBIO_REF_REL_L2,
        "inference_ms": {"distill": bench["distill"], "nbio": bench["nbio"]},
        "speedup_nbio_over_distill": {nb: bench["nbio"][nb] / bench["distill"][nb]
                                      for nb in bench["nbio"]},
        "accuracy_loss_factor_distill_over_nbio": ev["distill_interior_rel_l2"] / nbio_rel_l2,
        "n_params_distill": count_params(model),
        "device": str(device),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("[distill] wrote " + args.out, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
