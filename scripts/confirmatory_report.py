#!/usr/bin/env python
"""OBSOLETE pre-lock draft; retained only as provenance.

Use aggregate_screening_confirmatory.py from the locked protocol.

Confirmatory-v1 automated report: applies the pre-registered criteria of PROTOCOL.md to raw.csv.
One run, one report; humans inspect only this output. Criteria (verbatim from protocol):
 H1 (factorization): E_radial_mlp,bulk[1,4] <= 0.8 * min(E_generic_full, E_generic_none), paired CI vs the
    better control excludes zero. (J8 reported on the same bar as secondary.)
 H2 (monotone competitiveness): E_mix8,bulk[1,4] <= 1.5 * E_radial_mlp,bulk[1,4]; safer degradation:
    mean_k_extra[E_mix8(k)/E_mix8,in] < mean_k_extra[E_mlp(k)/E_mlp,in]  (bulk).
 NEAR: E_mix8,near@256[1,4] <= 0.6 * E_generic_none,near@256[1,4]; E_mix8,near@256 <= E_mix8,near@128
    (aggregate [1,4]). nb=64 near = stress test, reported not gated.
 OOR guardrail: for each k in {4.5,5.25,6.0} (bulk): E_mix8(k) <= 2 * min(E_full(k), E_none(k)).
 STABILITY: zero nonfinite; residual < 1e-5; cond ratio <= 1.5; no in-range cell (bulk or near@128/256)
    where mix8 > 2x BOTH generic controls; direction of H1/H2/NEAR quantities consistent across 3 seeds.
 Outcomes: A (H1+H2 pass), B (H1 passes, H2 fails), C (H2/mix8 clauses pass, H1 fails), D (neither).
"""
from __future__ import annotations
raise SystemExit(
    "obsolete pre-lock draft; use scripts/aggregate_screening_confirmatory.py"
)
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/screening_law/confirmatory_v1"
K14 = (1.35, 1.9, 2.55, 3.25, 3.9)
K_IN = (0.15, 0.45, 0.85, 1.35, 1.9, 2.55, 3.25, 3.9)
K_EXTRA = (4.5, 5.25, 6.0)
GEN = ("generic_full", "generic_none")


def ci(d, seed=0, n=10000):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    cells = defaultdict(lambda: defaultdict(dict))  # (load,fam,kappa,margin,nb) -> method -> seed -> arr
    with (OUT / "raw.csv").open() as f:
        acc = defaultdict(list)
        for r in csv.DictReader(f):
            acc[(r["load"], r["family"], float(r["kappa"]), float(r["margin"]), int(r["n_b"]),
                 r["method"], int(r["data_seed"]))].append((int(r["sample_id"]), float(r["rel_l2"])))
    for (load, fam, k, mg, nb, m, s), v in acc.items():
        v.sort()
        cells[(load, fam, k, mg, nb)][m][s] = np.array([x for _, x in v])

    def inst(m, kappas, mg, nb, seed=None, loads=("point_source", "fourier")):
        """Concatenated paired per-instance vector over (load,family,kappa) at margin/nb."""
        out = []
        for load in loads:
            for fam in ("in_dist", "highmode", "rough", "deform"):
                for k in kappas:
                    c = cells[(load, fam, k, mg, nb)][m]
                    out.append(np.mean([c[s2] for s2 in (0, 1, 2)], axis=0) if seed is None else c[seed])
        return np.concatenate(out)

    rep = {"aggregates": {}, "criteria": {}}
    agg = {}
    for m in ("mix8", "radial_mlp", "generic_full", "generic_none", "correct_bie"):
        seed = -1 if m == "correct_bie" else None
        if m == "correct_bie":
            b14 = np.concatenate([cells[(l, f, k, 0.12, 128)][m][-1]
                                  for l in ("point_source", "fourier")
                                  for f in ("in_dist", "highmode", "rough", "deform") for k in K14])
            agg[m] = {"bulk14": float(b14.mean())}
            continue
        agg[m] = {
            "bulk14": float(inst(m, K14, 0.12, 128).mean()),
            "near14_256": float(inst(m, K14, 0.02, 256).mean()),
            "near14_128": float(inst(m, K14, 0.02, 128).mean()),
            "bulk_in": float(inst(m, K_IN, 0.12, 128).mean()),
            "oor_bulk": {k: float(inst(m, (k,), 0.12, 128).mean()) for k in K_EXTRA},
        }
    rep["aggregates"] = agg

    # H1
    better = min(agg["generic_full"]["bulk14"], agg["generic_none"]["bulk14"])
    better_m = "generic_full" if agg["generic_full"]["bulk14"] <= agg["generic_none"]["bulk14"] else "generic_none"
    h1_point = agg["radial_mlp"]["bulk14"] <= 0.8 * better
    h1_ci = ci(inst("radial_mlp", K14, 0.12, 128) - inst(better_m, K14, 0.12, 128))
    rep["criteria"]["H1"] = {"radial_mlp": agg["radial_mlp"]["bulk14"], "better_generic": better,
                             "bar_0p8": 0.8 * better, "ci_vs_better": h1_ci,
                             "mix8_same_bar": bool(agg["mix8"]["bulk14"] <= 0.8 * better),
                             "pass": bool(h1_point and h1_ci[1] < 0)}
    # H2
    h2_comp = agg["mix8"]["bulk14"] <= 1.5 * agg["radial_mlp"]["bulk14"]
    deg_mix = np.mean([agg["mix8"]["oor_bulk"][k] / agg["mix8"]["bulk_in"] for k in K_EXTRA])
    deg_mlp = np.mean([agg["radial_mlp"]["oor_bulk"][k] / agg["radial_mlp"]["bulk_in"] for k in K_EXTRA])
    rep["criteria"]["H2"] = {"mix8_bulk14": agg["mix8"]["bulk14"],
                             "bar_1p5xmlp": 1.5 * agg["radial_mlp"]["bulk14"],
                             "degradation_mix8": float(deg_mix), "degradation_mlp": float(deg_mlp),
                             "competitive": bool(h2_comp), "safer_degradation": bool(deg_mix < deg_mlp),
                             "pass": bool(h2_comp and deg_mix < deg_mlp)}
    # NEAR
    near_ok = agg["mix8"]["near14_256"] <= 0.6 * agg["generic_none"]["near14_256"]
    res_ok = agg["mix8"]["near14_256"] <= agg["mix8"]["near14_128"]
    rep["criteria"]["NEAR"] = {"mix8": agg["mix8"]["near14_256"],
                               "bar_0p6xnone": 0.6 * agg["generic_none"]["near14_256"],
                               "near128": agg["mix8"]["near14_128"],
                               "pass": bool(near_ok and res_ok)}
    # OOR safety diagnostic (per amended protocol): report ratios; gate only on catastrophic cells
    # (handled in STABILITY scan extended to OOR) and safer degradation (in H2).
    oor = {str(k): {"mix8": agg["mix8"]["oor_bulk"][k],
                    "ratio_vs_2xmin": agg["mix8"]["oor_bulk"][k] /
                        (2 * min(agg["generic_full"]["oor_bulk"][k], agg["generic_none"]["oor_bulk"][k]))}
           for k in K_EXTRA}
    oor_cat = []
    for (load, fam, k, mg, nb), mm in cells.items():
        if k <= 4.0 or (mg == 0.02 and nb == 64):
            continue
        c = float(np.mean([mm["mix8"][s] for s in (0, 1, 2)]))
        f = float(np.mean([mm["generic_full"][s] for s in (0, 1, 2)]))
        n = float(np.mean([mm["generic_none"][s] for s in (0, 1, 2)]))
        if c > 2 * f and c > 2 * n:
            oor_cat.append([load, fam, k, mg, nb, c, f, n])
    rep["criteria"]["OOR_safety"] = {"ratios": oor, "oor_catastrophic_cells": oor_cat,
                                     "pass": bool(not oor_cat)}
    # STABILITY
    stab = json.load(open(OUT / "stability.json"))
    cat = []
    for (load, fam, k, mg, nb), mm in cells.items():
        if k > 4.0 or (mg == 0.02 and nb == 64):
            continue
        c = float(np.mean([mm["mix8"][s] for s in (0, 1, 2)]))
        f = float(np.mean([mm["generic_full"][s] for s in (0, 1, 2)]))
        n = float(np.mean([mm["generic_none"][s] for s in (0, 1, 2)]))
        if c > 2 * f and c > 2 * n:
            cat.append([load, fam, k, mg, nb, c, f, n])
    dir_ok = True
    for s in (0, 1, 2):
        b_m = inst("mix8", K14, 0.12, 128, s).mean(); b_r = inst("radial_mlp", K14, 0.12, 128, s).mean()
        b_f = inst("generic_full", K14, 0.12, 128, s).mean(); b_n = inst("generic_none", K14, 0.12, 128, s).mean()
        n_m = inst("mix8", K14, 0.02, 256, s).mean(); n_n = inst("generic_none", K14, 0.02, 256, s).mean()
        dir_ok &= (b_r < min(b_f, b_n)) and (b_m < min(b_f, b_n)) and (n_m < 0.6 * n_n) \
                  and (b_m <= 1.5 * b_r)
    stab_ok = (all(v["nonfinite"] == 0 and v["residual"] < 1e-5 and v["cond_ratio"] <= 1.5
                   for v in stab.values()) and not cat and dir_ok)
    rep["criteria"]["STABILITY"] = {"per_ckpt_worst": stab, "catastrophic_cells": cat,
                                    "direction_all_seeds": bool(dir_ok), "pass": bool(stab_ok)}
    # outcome
    H1, H2 = rep["criteria"]["H1"]["pass"], rep["criteria"]["H2"]["pass"]
    NEAR, OORs, ST = rep["criteria"]["NEAR"]["pass"], rep["criteria"]["OOR_safety"]["pass"], stab_ok
    mix8_clauses = H2 and NEAR and OORs
    if H1 and mix8_clauses and ST:
        outcome = "A"
    elif H1 and ST:
        outcome = "B"
    elif mix8_clauses and ST:
        outcome = "C"
    else:
        outcome = "D"
    rep["OUTCOME"] = outcome
    rep["PAPERIZE"] = bool(outcome in ("A", "B", "C"))
    (OUT / "REPORT.json").write_text(json.dumps(rep, indent=2))
    lines = [f"# Confirmatory v1 report — OUTCOME {outcome} (paperize={rep['PAPERIZE']})", ""]
    for name, c in rep["criteria"].items():
        lines.append(f"- {name}: pass={c['pass']}")
    lines.append(f"- aggregates: {json.dumps(agg, indent=1)}")
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({k: v["pass"] for k, v in rep["criteria"].items()}, indent=2))
    print("OUTCOME:", outcome)
    print("wrote", OUT / "REPORT.json")


if __name__ == "__main__":
    main()
