#!/usr/bin/env python
"""(B) Clean M1/M2/M3 ABLATION with multi-seed error bars.

Isolates the three NBIO mechanisms on the interior-Laplace test set (rel-L2, mean +/- std over seeds):

  variant            split  solve     isolates
  -----------------  -----  --------  ------------------------------------------------------------------
  nbio_full          True   direct    M1 (known kernel) + M2 (direct solve) + M3 + learned correction
  nbio_nosplit       False  direct    drop the KNOWN double-layer kernel -> learn the whole kernel from
                                      scratch (isolates M1: how much the analytic split buys).
  nbio_neumann       True   neumann   replace the direct solve with the damped 2nd-kind fixed point
                                      (isolates M2: direct vs iterative -- should be ~equal).
  nbio_classical     True   direct    "no learned correction": gate=1, smooth correction zeroed -> the
                                      pure classical Nystrom solver THROUGH THE SAME CODE PATH (upper
                                      bound / ceiling; needs NO training).

The learned-correction-NECESSITY claim is cross-referenced (not rerun) from the existing Yukawa result
(wrong/unknown kernel): there the correct-kernel split is unavailable, and the learned correction is what
closes the gap (see results/nbio_yuk_s0 vs results/deeponet_yuk_s0 in reports/ood_ablation.md).

Modes:
  --mode eval : load the trained multi-seed checkpoints + build the training-free classical variant,
                eval on interior_test.pt, write results/ablation.json.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import BIEDataset, to_device
from src.training.trainer import rel_l2
from src.utils.common import configure_torch, get_device, save_json
from src.models import build_model
from src.models.nbio import NBIO


@torch.no_grad()
def _shard_eval(model, shard_path, device, batch_size=32):
    ds = BIEDataset(shard_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    errs, bems, n = [], [], 0
    for batch in loader:
        batch = to_device(batch, device)
        bs = batch["uq"].shape[0]
        errs.append(rel_l2(model(batch), batch["uq"]).item() * bs); n += bs
        if "bem_uq" in batch:
            b = batch["bem_uq"]; finite = torch.isfinite(b).all(dim=1)
            if finite.any():
                bems.append(rel_l2(b[finite], batch["uq"][finite]).item() * int(finite.sum()))
    return (sum(errs) / max(n, 1), (sum(bems) / max(n, 1)) if bems else float("nan"))


def _load(run_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    m = build_model(ckpt["config"])
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval(), int(sum(p.numel() for p in m.parameters()))


def _classical_nbio(device):
    """Training-free pure classical Nystrom solver via the NBIO code path: gate=1, smooth correction = 0."""
    m = NBIO(hidden=64, n_layers=3, split=True, solve="direct", region="interior")
    with torch.no_grad():
        for p in m.bio.smooth.parameters():
            p.zero_()
        m.bio.log_gate.zero_()            # gate = 1 + tanh(0) = 1 -> exactly the known double layer
    return m.to(device).eval(), int(sum(p.numel() for p in m.parameters()))


# variant -> (run-dir prefix, expects-training, takeaway)
VARIANTS = {
    "nbio_full":      ("results/ablM_nbio_full_s",      True),
    "nbio_nosplit":   ("results/ablM_nbio_nosplit_s",   True),
    "nbio_neumann":   ("results/ablM_nbio_neumann_s",   True),
}


def evaluate(args):
    configure_torch()
    device = get_device(True)
    test = os.path.join(args.data_dir, "interior_test.pt")
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"test_shard": test, "seeds": seeds, "variants": {}}

    bem_ref = None
    for vname, (prefix, _trained) in VARIANTS.items():
        errs, npars = [], []
        for s in seeds:
            run = f"{prefix}{s}"
            if not os.path.exists(os.path.join(run, "best.pt")):
                print(f"  [skip] {run} (no best.pt)")
                continue
            m, npar = _load(run, device)
            e, bem = _shard_eval(m, test, device); errs.append(e); npars.append(npar)
            if bem_ref is None:
                bem_ref = bem
        if not errs:
            print(f"  [warn] no checkpoints for {vname}"); continue
        out["variants"][vname] = {"rel_l2_seeds": errs, "rel_l2_mean": float(np.mean(errs)),
                                  "rel_l2_std": float(np.std(errs)), "n_params": npars[0], "n_seeds": len(errs)}
        print(f"  {vname:14s}: {np.mean(errs):.3e} +/- {np.std(errs):.1e}  (n_params={npars[0]}, seeds={len(errs)})")

    # training-free classical upper bound (deterministic; one eval)
    mc, nparc = _classical_nbio(device)
    ec, bem = _shard_eval(mc, test, device)
    if bem_ref is None:
        bem_ref = bem
    out["variants"]["nbio_classical"] = {"rel_l2_seeds": [ec], "rel_l2_mean": float(ec),
                                         "rel_l2_std": 0.0, "n_params": nparc, "n_seeds": 1,
                                         "note": "training-free: gate=1, smooth correction zeroed"}
    print(f"  {'nbio_classical':14s}: {ec:.3e} (training-free, n_params={nparc})")
    out["bem_reference_rel_l2"] = bem_ref
    print(f"  {'BEM reference':14s}: {bem_ref:.3e}")

    save_json(out, os.path.join("results", "ablation.json"))
    print("[ablation] wrote results/ablation.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval"], default="eval")
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
