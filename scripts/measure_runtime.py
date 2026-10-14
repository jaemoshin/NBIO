#!/usr/bin/env python
"""Properly-controlled same-GPU runtime audit: warm-up + per-repetition cuda.synchronize timing ->
median and IQR per component (assembly / solve / potential), for structure-only vs learned-split NBIO.
Reports the GPU model. Writes results/tmlr_revision/runtime_controlled.csv."""
from __future__ import annotations
import sys, time, csv
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model                       # noqa: E402
from src.data.loader import BIEDataset, to_device        # noqa: E402
from src.models.layers import double_layer_kernel        # noqa: E402

DATA = ROOT / "data_cache" / "tmlr_revision"
CKPT = ROOT / "results" / "tmlr_revision" / "nbio_yuk_split_s0" / "best.pt"
OUT = ROOT / "results" / "tmlr_revision" / "runtime_controlled.csv"
NBS = [64, 128, 256]
BATCH = 8
WARMUP = 15
REPS = 80


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, bs):
    for _ in range(WARMUP):
        fn()
    sync(dev)
    ts = []
    for _ in range(REPS):
        sync(dev); t0 = time.perf_counter()
        fn(); sync(dev)
        ts.append((time.perf_counter() - t0) * 1000.0 / bs)     # ms per sample
    return np.array(ts)


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    print(f"device: {dev}  gpu: {gpu}  batch: {BATCH}  warmup: {WARMUP}  reps: {REPS}")
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = build_model(ck["config"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    rows = []
    for n_b in NBS:
        ds = BIEDataset(str(DATA / f"yukawa_k4_in_dist_m120_nb{n_b}.pt"))
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False)))
        b = to_device(batch, dev); jump = model.jump

        def la():
            return model.bio.assemble(b["Xb"], b["Nb"], b["wb"], b["curv"], jump)
        A = la()
        def ls():
            return model.solver(A, b["g"], jump)
        mu = ls()
        def lp():
            P = model.bio.potential(b["Xq"], b["Xb"], b["Nb"], b["wb"])
            return torch.bmm(P, mu.unsqueeze(-1)).squeeze(-1)

        def sa():
            k = double_layer_kernel(b["Xb"], b["Xb"], b["Nb"], curv_s=b["curv"], self_kernel=True)
            return jump * torch.eye(n_b, device=dev).unsqueeze(0) + k * b["wb"][:, None, :]
        A0 = sa()
        def ss():
            return torch.linalg.solve(A0, b["g"].unsqueeze(-1)).squeeze(-1)
        mu0 = ss()
        def sp():
            P = double_layer_kernel(b["Xq"], b["Xb"], b["Nb"]) * b["wb"][:, None, :]
            return torch.bmm(P, mu0.unsqueeze(-1)).squeeze(-1)

        for method, (fa, fs, fp) in (("structure_only", (sa, ss, sp)),
                                     ("learned_split", (la, ls, lp))):
            comp = {c: timed(fn, dev, BATCH) for c, fn in (("assembly", fa), ("solve", fs), ("potential", fp))}
            tot = comp["assembly"] + comp["solve"] + comp["potential"]
            row = {"method": method, "gpu": gpu, "n_b": n_b, "batch": BATCH,
                   "total_median_ms": float(np.median(tot)),
                   "total_iqr_ms": float(np.percentile(tot, 75) - np.percentile(tot, 25))}
            for c in ("assembly", "solve", "potential"):
                row[f"{c}_median_ms"] = float(np.median(comp[c]))
                row[f"{c}_iqr_ms"] = float(np.percentile(comp[c], 75) - np.percentile(comp[c], 25))
            rows.append(row)
            print(f"  n_b={n_b:4d} {method:14s} total {row['total_median_ms']:.3f} "
                  f"(IQR {row['total_iqr_ms']:.3f}) ms/sample | "
                  f"asm {row['assembly_median_ms']:.3f} solve {row['solve_median_ms']:.3f} "
                  f"pot {row['potential_median_ms']:.3f}")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
