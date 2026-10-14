#!/usr/bin/env python
"""(A) Out-of-distribution GENERALIZATION study for NBIO.

The paper already shows zero-shot RESOLUTION transfer. This study tests a harder, main-conference
expectation: GEOMETRY / operator distribution shift. We train NBIO (split=True, solve=direct) and a
black-box baseline on the standard IN-DISTRIBUTION interior-Laplace family (5-mode smooth stars, as in
gen_data.py), then evaluate them ZERO-SHOT on several genuinely shifted OOD test families:

  (1) highmode  -- higher Fourier mode content (n_modes=9, slower decay): rougher boundaries.
  (2) rough     -- even more high-frequency energy (n_modes=8, decay=1.0, larger amp/dev cap).
  (3) deform    -- near-self-intersecting / high-deformation stars (dev pushed toward the star-shaped
                   limit r->0): very high curvature, small min-radius.
  (4) margin06  -- (optional) tighter query margin (queries closer to the boundary), in-dist geometry.

All OOD shards use the EXACT manufactured GT (point charges outside Omega) and the SAME shard format as
the training data, at the SAME boundary resolution n_b=128 as training -> this isolates DISTRIBUTION
SHIFT (not resolution). For each OOD set we also store the classical Nystrom-BEM prediction at n_b=128
(`bem_uq`) so the table shows NBIO TRACKING the classical solver even OOD.

Story under test: NBIO degrades GRACEFULLY and stays close to the BEM reference under shift (it is a
DISCRETIZED OPERATOR, not a fitted map), while the black-box collapses. We report HONESTLY if NBIO also
degrades -- BEM itself degrades at fixed n_b on rough boundaries (under-resolved quadrature), and NBIO,
being the same discretization, is expected to track that.

Modes:
  --mode gen   : (CPU) build the OOD test shards into data_cache/ (manufactured GT + BEM ref).
  --mode eval  : (GPU) load trained checkpoints, eval in-dist + OOD, write results/ood_generalization.json
                 and figures/ood_generalization.png.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.bie import boundary, radius_at, sample_charges, eval_u, sample_queries, bem_solve
from src.data.loader import BIEDataset, to_device
from src.training.trainer import rel_l2
from src.utils.common import configure_torch, get_device, save_json, load_json


# ----------------------------- OOD geometry sampler -----------------------------
def ood_coeffs(rng, n_modes, decay, amp, dev_cap, r0=1.0):
    """Star-shaped Fourier coeffs with a CONFIGURABLE deviation cap (cf. bie.sample_coeffs which fixes
    n_modes=5, decay=1.6, amp=0.32, dev_cap=0.55). Larger n_modes / amp / dev_cap -> rougher / more
    deformed (toward the star-shaped limit). dev_cap<1 keeps 1+sum(.)>0 so r(theta)>0 (single-valued
    radius => non-self-intersecting)."""
    k = np.arange(1, n_modes + 1)
    scale = amp / k ** decay
    a = rng.normal(0, 1, n_modes) * scale
    b = rng.normal(0, 1, n_modes) * scale
    dev = np.sum(np.abs(a) + np.abs(b))
    if dev > dev_cap:
        a *= dev_cap / dev
        b *= dev_cap / dev
    return {"r0": float(r0), "a": a, "b": b}


# in-distribution settings mirror src.data.bie.sample_coeffs defaults
IN_DIST = dict(n_modes=5, decay=1.6, amp=0.32, dev_cap=0.55, margin=0.12)

OOD_FAMILIES = {
    # tag        : (n_modes, decay, amp, dev_cap, margin, seed_off) -- seed_off makes gen deterministic
    "highmode":   dict(n_modes=9, decay=1.2, amp=0.45, dev_cap=0.55, margin=0.12, seed_off=1),
    "rough":      dict(n_modes=8, decay=1.0, amp=0.60, dev_cap=0.75, margin=0.12, seed_off=2),
    "deform":     dict(n_modes=5, decay=1.6, amp=0.60, dev_cap=0.85, margin=0.12, seed_off=3),
    "margin06":   dict(n_modes=5, decay=1.6, amp=0.32, dev_cap=0.55, margin=0.06, seed_off=4),
}


def _make_shard(n, fam, n_b, n_q, n_charges, seed):
    """Build one OOD shard with exact manufactured GT + classical-BEM reference at this n_b."""
    rng = np.random.default_rng(seed)
    rows = []
    th_fine = np.linspace(0, 2 * np.pi, 1024)
    n_reject = 0
    while len(rows) < n:
        c = ood_coeffs(rng, fam["n_modes"], fam["decay"], fam["amp"], fam["dev_cap"])
        if radius_at(c, th_fine).ravel().min() <= 1e-3:   # guard: stay strictly star-shaped
            n_reject += 1
            continue
        bd = boundary(c, n_b)
        ch = sample_charges(rng, "interior", c, n_charges)
        g = eval_u(bd["X"], ch)
        Xq = sample_queries(rng, c, "interior", n_q, margin=fam["margin"])
        uq = eval_u(Xq, ch)
        d = {"Xb": bd["X"], "Nb": bd["normal"], "wb": bd["w"], "curv": bd["curv"], "speed": bd["speed"],
             "g": g, "Xq": Xq, "uq": uq, "a": c["a"], "b": c["b"]}
        try:
            mu, ev = bem_solve(bd, g, "interior")
            d["bem_uq"] = ev(Xq); d["mu"] = mu
        except np.linalg.LinAlgError:
            d["bem_uq"] = np.full_like(uq, np.nan); d["mu"] = np.full_like(g, np.nan)
        rows.append({k: np.asarray(v, np.float32) for k, v in d.items()})
    out = {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0].keys()}
    return out, n_reject


def gen(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("=== generating OOD test shards (exact manufactured GT + BEM ref @ n_b=%d) ===" % args.n_b)
    summary = {}
    for tag, fam in OOD_FAMILIES.items():
        d, nrej = _make_shard(args.n_test, fam, args.n_b, args.n_q, args.n_charges, seed=args.seed_base + fam["seed_off"])
        path = os.path.join(args.out_dir, f"ood_{tag}.pt")
        torch.save(d, path)
        bem_rel = (torch.linalg.vector_norm(d["bem_uq"] - d["uq"], dim=1)
                   / (torch.linalg.vector_norm(d["uq"], dim=1) + 1e-9))
        curv_max = d["curv"].abs().amax(dim=1).median().item()
        rmin = (d["Xb"].norm(dim=-1).amin(dim=1)).median().item()
        summary[tag] = {"path": path, "n": int(d["uq"].shape[0]), "n_b": int(d["Xb"].shape[1]),
                        "bem_rel_l2_mean": float(bem_rel.mean()), "bem_rel_l2_med": float(bem_rel.median()),
                        "curv_max_med": curv_max, "rmin_med": rmin, "n_reject": nrej, **fam}
        print(f"  ood_{tag:9s} N={summary[tag]['n']} n_b={summary[tag]['n_b']}  "
              f"BEM rel-L2 mean={summary[tag]['bem_rel_l2_mean']:.2e}  curv_max(med)={curv_max:.1f}  "
              f"rmin(med)={rmin:.3f}  rejects={nrej}")
    save_json(summary, os.path.join("results", "ood_shards_summary.json"))
    print("[ood gen] DONE")


# ----------------------------- evaluation -----------------------------
def _load_model(run_dir, device):
    from src.models import build_model
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"]


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
    return {"rel_l2": sum(errs) / max(n, 1),
            "bem_rel_l2": (sum(bems) / max(n, 1)) if bems else float("nan"),
            "n_b": int(ds.d["Xb"].shape[1]), "n": n}


def evaluate(args):
    configure_torch()
    device = get_device(True)
    shards = [("in_dist", os.path.join(args.data_dir, "interior_test.pt"))]
    for tag in OOD_FAMILIES:
        shards.append((f"ood_{tag}", os.path.join(args.data_dir, f"ood_{tag}.pt")))

    models = {"nbio": args.nbio_dir, args.bb_name: args.bb_dir}
    out = {"in_dist_settings": IN_DIST, "ood_settings": OOD_FAMILIES, "models": {}}
    bem_row = {}
    for mname, mdir in models.items():
        model, cfg = _load_model(mdir, device)
        npar = int(sum(p.numel() for p in model.parameters()))
        rows = {}
        for tag, path in shards:
            r = _shard_eval(model, path, device)
            rows[tag] = r
            bem_row[tag] = r["bem_rel_l2"]   # same shard -> same BEM ref regardless of model
        out["models"][mname] = {"run_dir": mdir, "n_params": npar, "model_name": cfg["model"]["name"],
                                "kwargs": cfg["model"].get("kwargs", {}), "shards": rows}
        print(f"[{mname:10s}] params={npar}")
        for tag, _ in shards:
            print(f"    {tag:14s}: rel_l2 {rows[tag]['rel_l2']:.4e}")
    out["bem_reference"] = bem_row

    save_json(out, os.path.join("results", "ood_generalization.json"))
    print("[ood eval] wrote results/ood_generalization.json")
    _figure(out, shards)
    return out


def _figure(out, shards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [t for t, _ in shards]
    labels = {"in_dist": "in-dist", "ood_highmode": "OOD\nhighmode", "ood_rough": "OOD\nrough",
              "ood_deform": "OOD\ndeform", "ood_margin06": "OOD\nmargin0.06"}
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(9, 5))
    # BEM reference
    bem = [out["bem_reference"][t] for t in tags]
    ax.plot(x, bem, "k--o", lw=2, ms=7, label="classical BEM (ref, n_b=128)")
    colors = {"nbio": "tab:blue"}
    for mname, md in out["models"].items():
        y = [md["shards"][t]["rel_l2"] for t in tags]
        style = dict(marker="s", lw=2, ms=7)
        if mname == "nbio":
            ax.plot(x, y, color="tab:blue", label=f"NBIO ({md['n_params']} params)", **style)
        else:
            ax.plot(x, y, color="tab:red", label=f"{mname} ({md['n_params']} params)", **style)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([labels.get(t, t) for t in tags])
    ax.set_ylabel("rel-L2 error (log)")
    ax.set_title("OOD geometry generalization (zero-shot, trained on 5-mode smooth stars)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/ood_generalization.png", dpi=150)
    print("[ood eval] wrote figures/ood_generalization.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "eval"], required=True)
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--out-dir", default="data_cache")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-b", type=int, default=128)
    ap.add_argument("--n-q", type=int, default=256)
    ap.add_argument("--n-charges", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=4000)
    ap.add_argument("--nbio-dir", default="results/ood_nbio_s0")
    ap.add_argument("--bb-dir", default="results/ood_deeponet_s0")
    ap.add_argument("--bb-name", default="deeponet")
    args = ap.parse_args()
    if args.mode == "gen":
        gen(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
