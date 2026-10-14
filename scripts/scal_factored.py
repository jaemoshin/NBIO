#!/usr/bin/env python
"""Method 4 driver: train/eval the low-rank (SEPARABLE) NBIO correction at several ranks, compare to the
full pairwise NBIO baseline, and microbenchmark the correction matvec cost (O(n.r) vs O(n^2)).

Outputs results/scal_factored.json:
  {
    "ranks": [...],
    "per_rank": {r: {"val_rel_l2":..., "test_rel_l2":..., "n_params":..., "cond":...}, ...},
    "baseline": {"test_rel_l2":..., "n_params":..., "ckpt":...},
    "matvec_ms": {"n_b":[...], "ranks":[...], "factored":{r:[ms per n_b]}, "dense":[ms per n_b]},
    "meta": {...}
  }
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.utils.common import set_seed, get_device, configure_torch, save_json, count_params
from src.data.loader import BIEDataset, to_device
from src.models.scalable_factored import FactoredNBIO
from src.models.layers import KernelHypernet
from src.models import build_model
from src.training.trainer import train, rel_l2

RANKS = [4, 8, 16, 32]
DATA = "data_cache"
OUT_JSON = "results/scal_factored.json"
BASELINE_CKPT = "results/nbio_s0/best.pt"


def _cfg(rank, epochs, batch_size):
    return {
        "seed": 0, "device": "cuda",
        "train": {"batch_size": batch_size, "epochs": epochs, "lr": 1e-3,
                  "weight_decay": 0.0, "grad_clip": 1.0},
        "model": {"name": "factored_nbio",
                  "kwargs": {"rank": rank, "hidden": 64, "n_layers": 3, "feat_scale": 1.0,
                             "region": "interior", "split": True, "solve": "direct"}},
    }


@torch.no_grad()
def eval_rel_l2(model, loader, device):
    model.eval()
    num = 0.0; n = 0
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch)
        num += rel_l2(pred, batch["uq"]).item() * batch["uq"].shape[0]
        n += batch["uq"].shape[0]
    return num / max(n, 1)


@torch.no_grad()
def eval_baseline(device, test_loader):
    """Load the full pairwise NBIO baseline and eval on the same interior_test set (reference ~4.8e-7)."""
    if not os.path.exists(BASELINE_CKPT):
        return {"test_rel_l2": None, "n_params": None, "ckpt": BASELINE_CKPT, "note": "checkpoint missing"}
    ckpt = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    rl2 = eval_rel_l2(model, test_loader, device)
    return {"test_rel_l2": rl2, "n_params": count_params(model), "ckpt": BASELINE_CKPT,
            "config": ckpt["config"]}


def microbench_matvec(device, n_bs=(128, 256, 512, 1024, 2048), ranks=RANKS,
                      batch=4, warmup=5, iters=30):
    """Microbenchmark the CORRECTION matvec only.

    factored: PHI @ (PSI^T @ v)  with PHI,PSI (B,n,r)        -> O(n.r)
    dense:    K @ v   with K = KernelHypernet(Xt,Xs,Nt,Ns)   -> O(n^2) (assemble + matvec)
    We time the dense path two ways: (a) matvec given a precomputed dense K (pure O(n^2) matvec),
    and (b) assemble-then-matvec (what NBIO actually pays per forward, O(n^2) hypernet evals)."""
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    dense_net = KernelHypernet(hidden=64, n_layers=3, feat_scale=1.0).to(device).eval()
    factored = {r: FactoredNBIO(rank=r, hidden=64, n_layers=3).to(device).eval() for r in ranks}

    res = {"n_b": list(n_bs), "ranks": list(ranks), "batch": batch, "iters": iters,
           "factored_ms": {r: [] for r in ranks},
           "dense_matvec_ms": [], "dense_assemble_matvec_ms": []}

    torch.manual_seed(0)
    for n in n_bs:
        X = torch.randn(batch, n, 2, device=device)
        N = torch.randn(batch, n, 2, device=device)
        N = N / N.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        curv = torch.randn(batch, n, device=device)
        v = torch.randn(batch, n, device=device)

        with torch.no_grad():
            # ---- factored: O(n.r) ----
            for r in ranks:
                m = factored[r]
                PHI = m.factors_phi(X, N, curv)
                PSI = m.factors_psi(X, N, curv)
                for _ in range(warmup):
                    m.correction_matvec(PHI, PSI, v)
                sync(); t0 = time.perf_counter()
                for _ in range(iters):
                    m.correction_matvec(PHI, PSI, v)
                sync()
                res["factored_ms"][r].append(1000.0 * (time.perf_counter() - t0) / iters)

            # ---- dense pure matvec given precomputed K: O(n^2) ----
            K = dense_net(X, X, N, N)                            # (B,n,n)
            for _ in range(warmup):
                torch.bmm(K, v.unsqueeze(-1))
            sync(); t0 = time.perf_counter()
            for _ in range(iters):
                torch.bmm(K, v.unsqueeze(-1))
            sync()
            res["dense_matvec_ms"].append(1000.0 * (time.perf_counter() - t0) / iters)

            # ---- dense assemble-then-matvec: O(n^2) hypernet evals (what NBIO pays) ----
            for _ in range(warmup):
                Kk = dense_net(X, X, N, N); torch.bmm(Kk, v.unsqueeze(-1))
            sync(); t0 = time.perf_counter()
            for _ in range(iters):
                Kk = dense_net(X, X, N, N); torch.bmm(Kk, v.unsqueeze(-1))
            sync()
            res["dense_assemble_matvec_ms"].append(1000.0 * (time.perf_counter() - t0) / iters)

        print(f"[matvec] n_b={n:5d}  factored={[round(res['factored_ms'][r][-1],4) for r in ranks]} "
              f"dense_matvec={res['dense_matvec_ms'][-1]:.4f} "
              f"dense_assemble={res['dense_assemble_matvec_ms'][-1]:.4f} ms")
    return res


def main():
    configure_torch()
    set_seed(0)
    device = get_device(True)
    print(f"[scal_factored] device={device}")

    epochs = int(os.environ.get("EPOCHS", "80"))
    batch_size = int(os.environ.get("BATCH", "32"))

    train_set = BIEDataset(os.path.join(DATA, "interior_train.pt"))
    val_set = BIEDataset(os.path.join(DATA, "interior_val.pt"))
    test_set = BIEDataset(os.path.join(DATA, "interior_test.pt"))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    per_rank = {}
    for r in RANKS:
        set_seed(0)
        cfg = _cfg(r, epochs, batch_size)
        model = FactoredNBIO(**cfg["model"]["kwargs"])
        out_dir = f"results/scal_factored_r{r}"
        npar = count_params(model)
        print(f"\n[scal_factored] === rank={r}  n_params={npar}  epochs={epochs} bs={batch_size} ===")
        hist = train(model, train_set, val_set, cfg, device, out_dir)
        # reload best by val
        best = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(best["state_dict"]); model = model.to(device)
        test_rl2 = eval_rel_l2(model, test_loader, device)
        try:
            sample = {k: v.unsqueeze(0) for k, v in test_set[0].items()}
            cond = model.condition_number(to_device(sample, device))
        except Exception:
            cond = None
        per_rank[r] = {"val_rel_l2": hist["best_val_rel_l2"], "test_rel_l2": test_rl2,
                       "n_params": npar, "cond": cond}
        print(f"[scal_factored] rank={r}  val={hist['best_val_rel_l2']:.4e}  test={test_rl2:.4e}  cond={cond}")

    print("\n[scal_factored] === baseline (full pairwise NBIO) ===")
    baseline = eval_baseline(device, test_loader)
    print(f"[scal_factored] baseline test_rel_l2={baseline.get('test_rel_l2')}")

    print("\n[scal_factored] === matvec microbenchmark ===")
    matvec = microbench_matvec(device)

    out = {
        "ranks": RANKS,
        "per_rank": {str(k): v for k, v in per_rank.items()},
        "baseline": baseline,
        "matvec_ms": {
            "n_b": matvec["n_b"], "ranks": matvec["ranks"], "batch": matvec["batch"],
            "iters": matvec["iters"],
            "factored": {str(k): v for k, v in matvec["factored_ms"].items()},
            "dense_matvec": matvec["dense_matvec_ms"],
            "dense_assemble_matvec": matvec["dense_assemble_matvec_ms"],
        },
        "meta": {"epochs": epochs, "batch_size": batch_size, "device": str(device),
                 "node_feat_dim": FactoredNBIO.NODE_FEAT_DIM,
                 "n_b_train": int(train_set.d["Xb"].shape[1])},
    }
    save_json(out, OUT_JSON)
    print(f"\n[scal_factored] wrote {OUT_JSON}")
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
