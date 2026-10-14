#!/usr/bin/env python
"""OBSOLETE pre-lock draft; retained only as provenance.

Use eval_screening_confirmatory_family.py from the locked protocol.

Confirmatory-v1 blinded evaluation: all frozen checkpoints over the full confirmation matrix.
Runs ONLY after all 12 trainings complete; writes raw.csv + checkpoint hashes. No criteria here —
the report script applies them. Also computes stability diagnostics (residual, cond ratio,
perturbation margin ||A_exact^{-1}(A_model - A_exact)||_2) on nb=128 bulk shards."""
from __future__ import annotations
raise SystemExit(
    "obsolete pre-lock draft; use scripts/eval_screening_confirmatory_family.py"
)
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model  # noqa: E402
from src.data.bie import _yukawa_double_layer_kernel  # noqa: E402

DATA = ROOT / "data_cache/screening_confirmatory_v1"
OUT = ROOT / "results/screening_law/confirmatory_v1"
CK = OUT / "checkpoints"
K_ALL = (0.15, 0.45, 0.85, 1.35, 1.9, 2.55, 3.25, 3.9, 4.5, 5.25, 6.0)
NBS = (64, 128, 256)
MARGINS = (("m120", 0.12), ("m020", 0.02))
FAMS = ("in_dist", "highmode", "rough", "deform")
METHODS = ("mix8", "radial_mlp", "generic_full", "generic_none")


def tagk(k):
    return str(k).replace(".", "p")


def rel_l2(p, t):
    return torch.linalg.vector_norm(p - t, dim=-1) / torch.linalg.vector_norm(t, dim=-1).clamp_min(1e-9)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, hashes = {}, {}
    for m in METHODS:
        for s in (0, 1, 2):
            p = CK / f"data_s{s}_{m}" / "best.pt"
            ck = torch.load(p, map_location="cpu", weights_only=False)
            mod = build_model(ck["config"]); mod.load_state_dict(ck["state_dict"])
            models[(m, s)] = mod.to(dev).eval()
            hashes[f"{m}_s{s}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2))
    print(f"loaded {len(models)} checkpoints", flush=True)

    rows = []
    for fam in FAMS:
        for kap in K_ALL:
            for mt, mv in MARGINS:
                for nb in NBS:
                    for load in ("point_source", "fourier"):
                        d = torch.load(DATA / "confirmation" / f"{load}_{fam}_k{tagk(kap)}_{mt}_nb{nb}.pt",
                                       map_location="cpu", weights_only=False)
                        base = dict(load=load, family=fam, kappa=kap, margin=mv, n_b=nb)
                        errs_bem = rel_l2(d["bem_uq"], d["uq"]).tolist()
                        for sid, e in enumerate(errs_bem):
                            rows.append({**base, "method": "correct_bie", "data_seed": -1,
                                         "sample_id": sid, "rel_l2": e})
                        b_all = {k: v.to(dev) for k, v in d.items() if k != "bem_uq"}
                        for (m, s), mod in models.items():
                            with torch.no_grad():
                                errs = rel_l2(mod(b_all), b_all["uq"]).cpu().tolist()
                            for sid, e in enumerate(errs):
                                rows.append({**base, "method": m, "data_seed": s,
                                             "sample_id": sid, "rel_l2": e})
            print(f"[eval] {fam} kappa={kap:g}", flush=True)

    with (OUT / "raw.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows", flush=True)

    # stability diagnostics on point_source in_dist nb=128 bulk shards
    stab = {}
    for (m, s), mod in models.items():
        worst = {"residual": 0.0, "cond_ratio": 0.0, "pert_margin": 0.0, "nonfinite": 0}
        for kap in (0.45, 1.9, 3.9, 5.25):
            d = torch.load(DATA / f"confirmation/point_source_in_dist_k{tagk(kap)}_m120_nb128.pt",
                           map_location="cpu", weights_only=False)
            b = {k: v[:12].to(dev) for k, v in d.items() if k != "bem_uq"}
            with torch.no_grad():
                if hasattr(mod, "_kernel") and not hasattr(mod, "bio"):
                    K = mod._kernel(b["Xb"], b["Xb"], b["Nb"], b["kappa"], curv_s=b["curv"], self_kernel=True)
                    A = mod.jump * torch.eye(K.shape[-1], device=dev).unsqueeze(0) + K * b["wb"][:, None, :]
                else:
                    A = mod.bio.assemble(b["Xb"], b["Nb"], b["wb"], b["curv"], b["kappa"], mod.jump)
                mu = torch.linalg.solve(A, b["g"].unsqueeze(-1)).squeeze(-1)
                r = (torch.linalg.vector_norm(torch.bmm(A, mu.unsqueeze(-1)).squeeze(-1) - b["g"], dim=1)
                     / torch.linalg.vector_norm(b["g"], dim=1)).max().item()
                worst["nonfinite"] += int((~torch.isfinite(mu)).sum().item())
                # exact assembled matrix
                Xb = b["Xb"].cpu().numpy(); Nb = b["Nb"].cpu().numpy()
                wb = b["wb"].cpu().numpy(); cv = b["curv"].cpu().numpy()
                conds, perts = [], []
                for i in range(Xb.shape[0]):
                    Ke = _yukawa_double_layer_kernel(Xb[i], Xb[i], Nb[i], kap)
                    np.fill_diagonal(Ke, -cv[i] / (4 * np.pi))
                    Ae = -0.5 * np.eye(Xb.shape[1]) + Ke * wb[i][None, :]
                    Am = A[i].cpu().numpy()
                    conds.append(np.linalg.cond(Am) / np.linalg.cond(Ae))
                    perts.append(np.linalg.norm(np.linalg.solve(Ae, Am - Ae), 2))
                worst["residual"] = max(worst["residual"], r)
                worst["cond_ratio"] = max(worst["cond_ratio"], float(max(conds)))
                worst["pert_margin"] = max(worst["pert_margin"], float(max(perts)))
        stab[f"{m}_s{s}"] = worst
    (OUT / "stability.json").write_text(json.dumps(stab, indent=2))
    print("stability written", flush=True)


if __name__ == "__main__":
    main()
