#!/usr/bin/env python
"""Method-4 (low-rank learned kernel) stress test on HELMHOLTZ.

The paper's scalability subsection claims the learned smooth-kernel correction is genuinely low-rank
(rank 4 matches full accuracy) -- but that was shown only for LAPLACE, whose smooth double-layer correction
is intrinsically low-rank. Oscillatory Helmholtz (Delta u + k^2 u = 0) becomes higher-rank as the wavenumber
k grows. This driver quantifies how the rank required by a separable/factored REAL learned operator grows
with k.

For each N (wavelengths across the domain diameter) in {1,2,4,8,16} and each separable rank r in {4,8,16,32}:
  train FactoredHelmholtzNBIO (no-split, full learned real kernel) on helm_N{N}_train (val on val),
  ~80 epochs, batch 32, Adam 1e-3, rel_l2 loss; eval test rel-L2.

Builds accuracy[N][r] = test rel-L2 and reports the rank needed to reach 1e-3 at each N. Saves
results/scal_helmholtz_factored.json and figures/helmholtz_factored.png (test rel-L2 vs rank, one curve per N).

ADDITIVE / NEW FILE ONLY. Reuses src.training.trainer.train/rel_l2, src.data.loader.BIEDataset,
src.models.factored_helmholtz.FactoredHelmholtzNBIO, src.utils.common -- all unchanged.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.utils.common import set_seed, get_device, configure_torch, save_json, count_params
from src.data.loader import BIEDataset, to_device
from src.models.factored_helmholtz import FactoredHelmholtzNBIO
from src.training.trainer import train, rel_l2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_cache")
OUT_JSON = os.path.join(ROOT, "results", "scal_helmholtz_factored.json")
FIG_PNG = os.path.join(ROOT, "figures", "helmholtz_factored.png")

N_WL = [1, 2, 4, 8, 16]
RANKS = [4, 8, 16, 32]
TARGET = 1e-3        # accuracy threshold for "rank needed"


def _cfg(epochs, batch_size):
    return {"seed": 0, "device": "cuda",
            "train": {"batch_size": batch_size, "epochs": epochs, "lr": 1e-3,
                      "weight_decay": 0.0, "grad_clip": 1.0}}


@torch.no_grad()
def eval_rel_l2(model, loader, device):
    model.eval(); num = 0.0; n = 0
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch)
        num += rel_l2(pred, batch["uq"]).item() * batch["uq"].shape[0]
        n += batch["uq"].shape[0]
    return num / max(n, 1)


def rank_needed(per_rank, target=TARGET):
    """Smallest rank r (in RANKS, ascending) reaching test rel-L2 <= target, else None (with best achieved)."""
    for r in RANKS:
        if per_rank[str(r)] is not None and per_rank[str(r)] <= target:
            return r
    return None


def figure(per_N, diam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    cmap = plt.get_cmap("viridis")
    for i, N in enumerate(N_WL):
        ys = [per_N[str(N)]["per_rank"][str(r)] for r in RANKS]
        k = per_N[str(N)]["k"]
        ax.plot(RANKS, ys, "o-", color=cmap(i / max(len(N_WL) - 1, 1)),
                label=f"N={N} wl (k={k:.1f})")
    ax.axhline(TARGET, ls="--", color="0.4", alpha=0.7, label=f"target {TARGET:g}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(RANKS); ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlabel("separable rank r")
    ax.set_ylabel("test rel-L2")
    ax.set_title("Helmholtz factored learned operator: accuracy vs rank\n"
                 "(curves shift up/right as wavenumber k grows -> higher rank needed)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PNG), exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150, bbox_inches="tight")
    print(f"[scal_helmholtz_factored] wrote {FIG_PNG}")


def main():
    configure_torch()
    device = get_device(True)
    epochs = int(os.environ.get("EPOCHS", "80"))
    batch_size = int(os.environ.get("BATCH", "32"))
    print(f"[scal_helmholtz_factored] device={device} epochs={epochs} batch={batch_size}")

    # diameter: recompute from boundary the same way gen does (mean 2*max|X|), and infer n_b from a shard
    from scripts.gen_helmholtz_factored import compute_diam
    diam, diam_pair_mean, diam_pair_max = compute_diam()
    print(f"[scal_helmholtz_factored] diam={diam:.4f} (pairwise mean {diam_pair_mean:.4f}, max {diam_pair_max:.4f})")

    per_N = {}
    n_b = None
    for N in N_WL:
        k = 2 * 3.141592653589793 * N / diam
        train_set = BIEDataset(os.path.join(DATA, f"helm_N{N}_train.pt"))
        val_set = BIEDataset(os.path.join(DATA, f"helm_N{N}_val.pt"))
        test_set = BIEDataset(os.path.join(DATA, f"helm_N{N}_test.pt"))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        n_b = int(train_set.d["Xb"].shape[1])

        per_rank = {}
        n_params = {}
        for r in RANKS:
            set_seed(0)
            cfg = _cfg(epochs, batch_size)
            model = FactoredHelmholtzNBIO(rank=r, hidden=64, n_layers=3, region="interior", solve="direct")
            npar = count_params(model)
            out_dir = os.path.join(ROOT, "results", f"scal_helm_N{N}_r{r}")
            print(f"\n[scal_helmholtz_factored] === N={N} (k={k:.3f})  rank={r}  n_params={npar} ===")
            hist = train(model, train_set, val_set, cfg, device, out_dir)
            best = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(best["state_dict"]); model = model.to(device)
            test_rl2 = eval_rel_l2(model, test_loader, device)
            per_rank[str(r)] = test_rl2
            n_params[str(r)] = npar
            print(f"[scal_helmholtz_factored] N={N} r={r}  val={hist['best_val_rel_l2']:.4e}  test={test_rl2:.4e}")

        rn = rank_needed(per_rank)
        per_N[str(N)] = {"k": k, "per_rank": per_rank, "n_params": n_params,
                         "rank_needed_1e-3": rn, "best_rel_l2": min(per_rank.values())}
        print(f"[scal_helmholtz_factored] N={N}: rank needed for {TARGET:g} = {rn}  "
              f"(best achieved {min(per_rank.values()):.2e})")

    out = {
        "diam": diam, "diam_pairwise_mean": diam_pair_mean, "diam_pairwise_max": diam_pair_max,
        "n_b": n_b, "ranks": RANKS, "target": TARGET,
        "per_N": per_N,
        "meta": {"epochs": epochs, "batch_size": batch_size, "device": str(device),
                 "lr": 1e-3, "loss": "rel_l2", "model": "FactoredHelmholtzNBIO(split=False, real)",
                 "N_wavelengths": N_WL},
    }
    save_json(out, OUT_JSON)
    print(f"\n[scal_helmholtz_factored] wrote {OUT_JSON}")
    figure(per_N, diam)
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
