#!/usr/bin/env python
"""Stage 1: frozen-checkpoint kappa-family audit (NO retraining).

The trained kappa=4 checkpoints receive geometry, boundary data, and query points but NOT kappa. We fix a
kappa-INDEPENDENT boundary trace g (a normalized 6-mode Fourier trace, identical at every kappa, so the
model cannot infer kappa from the data generator), take the model prediction u_hat once (kappa-independent),
and compare it to the reference solution u_kappa=BEM(Gamma,g,kappa) of (Delta-kappa^2)u=0 across a kappa grid.
Reports rel-L2 vs kappa, degradation relative to kappa=4, the exact-target separation ||u_kappa-u_4||, the
half-separation lower bound (1/2)||u_kappa-u_4|| (max error of ANY kappa-free predictor), and the empirical
transfer radius (kappa interval where rel-L2 stays within 2x its trained-kappa value)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model                       # noqa: E402
from src.data.bie import bem_solve                       # noqa: E402

DATA = ROOT / "data_cache" / "tmlr_revision"
CKPTS = {
    ("split", 0): ROOT / "results/tmlr_revision/nbio_yuk_split_s0/best.pt",
    ("split", 1): ROOT / "results/nbio_yuk_split_s1/best.pt",
    ("split", 2): ROOT / "results/nbio_yuk_split_s2/best.pt",
    ("nosplit", 0): ROOT / "results/tmlr_revision/nbio_yuk_nosplit_s0/best.pt",
    ("nosplit", 1): ROOT / "results/nbio_yuk_nosplit_s1/best.pt",
    ("nosplit", 2): ROOT / "results/nbio_yuk_nosplit_s2/best.pt",
}
FAMILIES = ["in_dist", "rough", "deform", "highmode"]
NBS = [128, 256]
MARGINS = [("m120", 0.12), ("m020", 0.02)]
KAPPAS = [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0, 6.0, 8.0]
KREF = 4.0


def fourier_g(n_b, gidx):
    """kappa-independent boundary trace at nodes t_i=2pi i/n_b; fixed coeffs per geometry index."""
    rng = np.random.default_rng(10_000 + gidx)
    a = rng.standard_normal(7); b = rng.standard_normal(7)
    t = 2 * np.pi * np.arange(n_b) / n_b
    g = np.full(n_b, a[0])
    for m in range(1, 7):
        g += a[m] * np.cos(m * t) + b[m] * np.sin(m * t)
    return g.astype(np.float64)


def load_shard(fam, mtag, n_b):
    d = torch.load(DATA / f"yukawa_k4_{fam}_{mtag}_nb{n_b}.pt", map_location="cpu", weights_only=False)
    return {k: d[k].double() for k in ["Xb", "Nb", "wb", "curv", "Xq"]}


def main():
    dev = torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    models = {}
    for key, ck in CKPTS.items():
        if not ck.exists():
            print(f"# missing {key}: {ck}"); continue
        c = torch.load(ck, map_location=dev, weights_only=False)
        m = build_model(c["config"]).double(); m.load_state_dict(c["state_dict"]); m.eval()
        models[key] = m
    # accumulate rel-L2 per (kind, kappa) and separation per kappa
    rel = {("split", k): {kap: [] for kap in KAPPAS} for k in (0, 1, 2)}
    rel.update({("nosplit", k): {kap: [] for kap in KAPPAS} for k in (0, 1, 2)})
    sep = {kap: [] for kap in KAPPAS}     # ||u_kappa-u_4||/||u_4||
    gidx = 0
    with torch.no_grad():
        for fam in FAMILIES:
            for mtag, _mv in MARGINS:
                for n_b in NBS:
                    sh = load_shard(fam, mtag, n_b)
                    B = min(24, sh["Xb"].shape[0])
                    gs = np.stack([fourier_g(n_b, gidx + i) for i in range(B)])   # (B,n_b)
                    sh = {k: v[:B] for k, v in sh.items()}   # subsample geometries
                    gt = torch.from_numpy(gs)
                    batch = {"Xb": sh["Xb"], "Nb": sh["Nb"], "wb": sh["wb"], "curv": sh["curv"],
                             "g": gt, "Xq": sh["Xq"]}
                    # model predictions (kappa-independent), per checkpoint
                    preds = {key: models[key](batch).cpu().numpy() for key in models}   # (B,Q)
                    # reference u_kappa per geometry per kappa
                    Xb = sh["Xb"].numpy(); Nb = sh["Nb"].numpy(); wb = sh["wb"].numpy()
                    cv = sh["curv"].numpy(); Xq = sh["Xq"].numpy()
                    for i in range(B):
                        bd = {"X": Xb[i], "normal": Nb[i], "w": wb[i], "curv": cv[i]}
                        uk = {}
                        for kap in KAPPAS:
                            _, ev = bem_solve(bd, gs[i], "interior", kappa=kap)
                            uk[kap] = ev(Xq[i])
                        u4 = uk[KREF]; n4 = np.linalg.norm(u4) + 1e-30
                        for kap in KAPPAS:
                            sep[kap].append(np.linalg.norm(uk[kap] - u4) / n4)
                            for key in models:
                                e = np.linalg.norm(preds[key][i] - uk[kap]) / (np.linalg.norm(uk[kap]) + 1e-30)
                                rel[key][kap].append(e)
                    gidx += B
                    print(f"[stage1] {fam} {mtag} nb{n_b} B={B} done", flush=True)

    def agg(kind):
        seeds = [(kind, s) for s in (0, 1, 2) if (kind, s) in models]
        out = {}
        for kap in KAPPAS:
            vals = np.concatenate([np.asarray(rel[k][kap]) for k in seeds])
            out[kap] = dict(mean=float(vals.mean()), median=float(np.median(vals)),
                            p90=float(np.percentile(vals, 90)))
        return out

    summary = {"kappas": KAPPAS, "kref": KREF,
               "separation_mean": {kap: float(np.mean(sep[kap])) for kap in KAPPAS},
               "half_separation_mean": {kap: float(0.5 * np.mean(sep[kap])) for kap in KAPPAS}}
    print("\n=== rel-L2 vs kappa (mean over 3 seeds; kref=4) ===")
    print(f"{'kappa':>6} | {'split mean':>10} {'deg/k4':>7} | {'nosplit mean':>12} {'deg/k4':>7} | "
          f"{'sep/u4':>7} {'half':>6}")
    for kind in ("split", "nosplit"):
        summary[kind] = agg(kind)
    s4 = summary["split"][KREF]["mean"]; n4 = summary["nosplit"][KREF]["mean"]
    for kap in KAPPAS:
        sm = summary["split"][kap]["mean"]; nm = summary["nosplit"][kap]["mean"]
        print(f"{kap:6.2f} | {sm:10.3e} {sm/s4:7.2f} | {nm:12.3e} {nm/n4:7.2f} | "
              f"{summary['separation_mean'][kap]:7.3f} {summary['half_separation_mean'][kap]:6.3f}")

    def radius(kind):
        base = summary[kind][KREF]["mean"]; lo = hi = KREF
        for kap in sorted(k for k in KAPPAS if k <= KREF)[::-1]:
            if summary[kind][kap]["mean"] <= 2 * base: lo = kap
            else: break
        for kap in sorted(k for k in KAPPAS if k >= KREF):
            if summary[kind][kap]["mean"] <= 2 * base: hi = kap
            else: break
        return lo, hi
    for kind in ("split", "nosplit"):
        lo, hi = radius(kind); summary[kind + "_transfer_radius_2x"] = [lo, hi]
        print(f"  {kind}: 2x-error transfer radius kappa in [{lo}, {hi}] (trained kref={KREF})")
    (ROOT / "results/tmlr_revision/kappa_sweep.json").write_text(json.dumps(summary, indent=2))
    print("wrote results/tmlr_revision/kappa_sweep.json")


if __name__ == "__main__":
    main()
