#!/usr/bin/env python
"""Stage 2/3: kappa-CONDITIONED NBIO across the modified-Helmholtz family.

Adds log(1+kappa) to the correction network (kappa_cond=True) so the learned regular remainder is
kappa-dependent while the analytic (Laplace) singular core stays fixed. Trains on kappa in {1,2,4,6} and
evaluates on held-out interpolation {1.5,3,5} and out-of-range {0.5,8}, at bulk (margin 0.12) and
near-boundary (0.02) queries, for the split (Laplace prior + kappa-corr) and no-split (fully-learned
kappa-corr) variants. Per-kappa controls (unconditioned NBIO trained at a single kappa) confirm the frozen
failure is operator mismatch, not a broken pipeline. Stage 3 = the split-vs-no-split bulk/near-boundary
comparison in the conditioned model."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.nbio import NBIO                                   # noqa: E402
from src.data.bie import sample_coeffs, boundary, sample_charges, sample_queries, eval_u  # noqa: E402

N_B = 128
TRAIN_K = [1.0, 2.0, 4.0, 6.0]
INTERP_K = [1.5, 3.0, 5.0]
EXTRA_K = [0.5, 8.0]
OUT = ROOT / "results" / "tmlr_revision"


def gen(kappas, n_geom, seed, margin=0.12, n_q=256):
    rng = np.random.default_rng(seed)
    R = {k: [] for k in ["Xb", "Nb", "wb", "curv", "g", "Xq", "uq", "kappa"]}
    fine = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    for kap in kappas:
        made = 0
        while made < n_geom:
            co = sample_coeffs(rng)
            from src.data.bie import radius_at
            if radius_at(co, fine).min() <= 0.05:
                continue
            bd = boundary(co, N_B); ch = sample_charges(rng, "interior", co)
            Xq = sample_queries(rng, co, "interior", n_q, margin=margin)
            R["Xb"].append(bd["X"]); R["Nb"].append(bd["normal"]); R["wb"].append(bd["w"])
            R["curv"].append(bd["curv"]); R["g"].append(eval_u(bd["X"], ch, kap))
            R["Xq"].append(Xq); R["uq"].append(eval_u(Xq, ch, kap)); R["kappa"].append(kap)
            made += 1
    T = {k: torch.tensor(np.asarray(v), dtype=torch.float32) for k, v in R.items() if k != "kappa"}
    T["kappa"] = torch.tensor(R["kappa"], dtype=torch.float32)
    return T


def rel_l2(pred, uq):
    return (torch.linalg.vector_norm(pred - uq, dim=1) / (torch.linalg.vector_norm(uq, dim=1) + 1e-9))


def batches(T, bs, dev, shuffle, gen_):
    n = T["kappa"].shape[0]
    idx = torch.randperm(n, generator=gen_) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        j = idx[s:s + bs]
        yield {k: v[j].to(dev) for k, v in T.items()}


def train_model(model, tr, va, dev, epochs=60, bs=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        for b in batches(tr, bs, dev, True, g):
            opt.zero_grad(); loss = rel_l2(model(b), b["uq"]).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vr = torch.cat([rel_l2(model(b), b["uq"]) for b in batches(va, 64, dev, False, g)]).mean().item()
        if vr < best[0]:
            best = (vr, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return best[0]


@torch.no_grad()
def eval_at(model, dev, kappas, margin, n_geom=48, seed=777):
    out = {}
    for kap in kappas:
        T = gen([kap], n_geom, seed + int(kap * 100), margin=margin)
        errs = torch.cat([rel_l2(model(b), b["uq"]) for b in batches(T, 64, dev, False,
                          torch.Generator().manual_seed(0))]).cpu().numpy()
        out[kap] = dict(mean=float(errs.mean()), median=float(np.median(errs)))
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    t0 = time.time()
    print(f"device {dev}; generating multi-kappa data...", flush=True)
    tr = gen(TRAIN_K, 150, seed=1, margin=0.12)
    va = gen(TRAIN_K, 30, seed=2, margin=0.12)
    print(f"train {tr['kappa'].shape[0]} val {va['kappa'].shape[0]} ({time.time()-t0:.0f}s)", flush=True)

    results = {"train_kappa": TRAIN_K, "interp_kappa": INTERP_K, "extra_kappa": EXTRA_K}
    for kind, split in (("cond_split", True), ("cond_nosplit", False)):
        m = NBIO(hidden=64, n_layers=3, region="interior", split=split, solve="direct",
                 kappa_cond=True).to(dev)
        bv = train_model(m, tr, va, dev)
        print(f"[{kind}] best_val {bv:.3e} ({time.time()-t0:.0f}s)", flush=True)
        results[kind] = {"best_val": bv}
        for mtag, mv in (("bulk", 0.12), ("near", 0.02)):
            results[kind][mtag] = {"interp": eval_at(m, dev, INTERP_K, mv),
                                   "extra": eval_at(m, dev, EXTRA_K, mv),
                                   "train_k": eval_at(m, dev, TRAIN_K, mv)}

    # per-kappa positive controls (unconditioned, single-kappa training)
    results["controls"] = {}
    for kap in [1.0, 2.0, 6.0]:
        trc = gen([kap], 150, seed=10 + int(kap), margin=0.12)
        vac = gen([kap], 30, seed=20 + int(kap), margin=0.12)
        mc = NBIO(hidden=64, n_layers=3, region="interior", split=True, solve="direct",
                  kappa_cond=False).to(dev)
        train_model(mc, trc, vac, dev, epochs=50)
        e = eval_at(mc, dev, [kap], 0.12)
        results["controls"][kap] = e[kap]
        print(f"[control kappa={kap}] rel-L2 {e[kap]['mean']:.3e} ({time.time()-t0:.0f}s)", flush=True)

    (OUT / "kappa_conditioned.json").write_text(json.dumps(results, indent=2))
    print("\n=== SUMMARY (rel-L2 mean) ===")
    for kind in ("cond_split", "cond_nosplit"):
        r = results[kind]
        print(f"[{kind}] best_val {r['best_val']:.3e}")
        for mtag in ("bulk", "near"):
            i = r[mtag]["interp"]; x = r[mtag]["extra"]
            print(f"   {mtag}: interp " + " ".join(f"k{k}={i[k]['mean']:.2e}" for k in INTERP_K)
                  + " | extra " + " ".join(f"k{k}={x[k]['mean']:.2e}" for k in EXTRA_K))
    print("controls (single-kappa trained, eval same kappa):",
          {k: f"{v['mean']:.2e}" for k, v in results["controls"].items()})
    print("wrote results/tmlr_revision/kappa_conditioned.json")


if __name__ == "__main__":
    main()
