#!/usr/bin/env python
"""Confirmatory-v1 data generation for Screening-Law NBIO (BLINDED; run before any training).

Training/validation: 3 fresh splits with the established continuous-kappa distribution law
(generated_split of gen_kappa_continuous), NEW seeds train 91001+d / val 92001+d; pipeline-check
yukawa_test.pt from fresh seed 93001 (declared NON-confirmatory; only feeds train.py's automatic printout).

Blinded confirmation: kappa_test {0.15,0.45,0.85,1.35,1.9,2.55,3.25,3.9} + kappa_extra {4.5,5.25,6.0};
4 OOD families x 24 fresh geometries (coefficients shared across kappa/margins/nb for pairing);
loads: point_source (exact manufactured reference) and fourier (kappa-independent 6-mode trace;
reference = 512-node correct-Yukawa BEM at the queries); margins {0.12, 0.02}; nb {64,128,256}.
Each shard also carries bem_uq = correct-Yukawa BIE prediction AT THE SHARD'S nb (ceiling control).
Confirmation values are written, hashed, and NOT read until all trainings complete.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.bie import (boundary, sample_charges, sample_queries, eval_u, bem_solve, radius_at)  # noqa: E402
from gen_kappa_continuous import generated_split                                                    # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs                                           # noqa: E402
from tmlr_revision2_conditioned_transfer import fourier_coefficients, fourier_trace                 # noqa: E402

K_TEST = (0.15, 0.45, 0.85, 1.35, 1.9, 2.55, 3.25, 3.9)
K_EXTRA = (4.5, 5.25, 6.0)
NBS = (64, 128, 256)
MARGINS = (("m120", 0.12), ("m020", 0.02))
N_GEOM = 24
N_Q = 256
REF_NB = 512
DATA = ROOT / "data_cache" / "screening_confirmatory_v1"


def tagk(k):
    return str(k).replace(".", "p")


def gen_train():
    for d in range(3):
        out = DATA / f"data_s{d}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(generated_split(2400, 91001 + d), out / "yukawa_train.pt")
        torch.save(generated_split(400, 92001 + d), out / "yukawa_val.pt")
        print(f"train split d={d} done", flush=True)
    check = generated_split(200, 93001)
    for d in range(3):
        torch.save(check, DATA / f"data_s{d}" / "yukawa_test.pt")
    print("pipeline-check test written (non-confirmatory)", flush=True)


def gen_confirmation():
    conf = DATA / "confirmation"
    conf.mkdir(parents=True, exist_ok=True)
    fine = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    for fi, (fam, settings) in enumerate(FAMILIES.items()):
        rng = np.random.default_rng(95001 + fi)
        geoms = []
        while len(geoms) < N_GEOM:
            co = ood_coeffs(rng, settings)
            if radius_at(co, fine).min() > 0.05:
                geoms.append({
                    "co": co,
                    "charges": sample_charges(rng, "interior", co, n_charges=6),
                    "fourier": fourier_coefficients(rng),
                    "queries": {mv: sample_queries(rng, co, "interior", N_Q, margin=mv)
                                for _, mv in MARGINS},
                })
        for kap in K_TEST + K_EXTRA:
            # reference solves at REF_NB for fourier loads (one per geometry)
            for g in geoms:
                bd_ref = boundary(g["co"], REF_NB)
                trace = fourier_trace(bd_ref["theta"], g["fourier"])
                _, ev = bem_solve(bd_ref, trace, "interior", kappa=kap)
                g["_fref"] = {mv: ev(g["queries"][mv]) for _, mv in MARGINS}
            for mt, mv in MARGINS:
                for nb in NBS:
                    for load in ("point_source", "fourier"):
                        R = {k: [] for k in ("Xb", "Nb", "wb", "curv", "g", "Xq", "uq", "bem_uq")}
                        for g in geoms:
                            bd = boundary(g["co"], nb)
                            if load == "point_source":
                                trace = eval_u(bd["X"], g["charges"], kap)
                                uq = eval_u(g["queries"][mv], g["charges"], kap)
                            else:
                                trace = fourier_trace(bd["theta"], g["fourier"])
                                uq = g["_fref"][mv]
                            _, ev = bem_solve(bd, trace, "interior", kappa=kap)
                            R["Xb"].append(bd["X"]); R["Nb"].append(bd["normal"])
                            R["wb"].append(bd["w"]); R["curv"].append(bd["curv"])
                            R["g"].append(trace); R["Xq"].append(g["queries"][mv])
                            R["uq"].append(uq); R["bem_uq"].append(ev(g["queries"][mv]))
                        T = {k: torch.tensor(np.asarray(v), dtype=torch.float32) for k, v in R.items()}
                        T["kappa"] = torch.full((N_GEOM,), float(kap))
                        torch.save(T, conf / f"{load}_{fam}_k{tagk(kap)}_{mt}_nb{nb}.pt")
            print(f"[conf] {fam} kappa={kap:g} done", flush=True)


def main():
    gen_train()
    gen_confirmation()
    n = len(list((DATA / "confirmation").glob("*.pt")))
    manifest = {"train_seeds": [91001, 91002, 91003], "val_seeds": [92001, 92002, 92003],
                "check_seed": 93001, "family_seeds": [95001, 95002, 95003, 95004],
                "k_test": list(K_TEST), "k_extra": list(K_EXTRA), "n_shards": n}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {n} confirmation shards + 3 train splits", flush=True)


if __name__ == "__main__":
    main()
