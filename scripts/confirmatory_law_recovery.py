#!/usr/bin/env python
"""PRE-LOCK DRAFT retained as provenance; paths do not match the locked pipeline.

Do not run this version. A compatible diagnostic is created only if confirmation passes.

Priority 2: recover the physical screening law from solution-trained models (evaluation-only diagnostic).

For every trained seed of mixture-J8 and radial_mlp: compare m_theta(s) against the exact diagnostic law
m*(s) = s K_1(s) (NEVER used in training). Reports median/max relative law error on the in-range s-interval,
cross-seed variation, out-of-range behavior, error binned by training s-density, and the correlation between
law-recovery error and confirmation bulk[1,4] accuracy (joined from raw.csv). Writes law_recovery.json +
law_curves.csv."""
from __future__ import annotations
raise SystemExit(
    "pre-lock draft is incompatible with confirmatory_v1; use the post-confirmation diagnostic"
)
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.special import kn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model  # noqa: E402

OUT = ROOT / "results/screening_law/confirmatory_v1"
CK = OUT / "checkpoints"
K14 = (1.35, 1.9, 2.55, 3.25, 3.9)
S_IN_MAX = 4.0 * 3.3          # kappa_max * typical max pair distance
S_GRID = np.linspace(1e-3, 1.5 * S_IN_MAX, 800)


def train_s_density():
    """Empirical s=kappa*r density from the actual training pairs (data_s0)."""
    d = torch.load(ROOT / "data_cache/screening_confirmatory_v1/data_s0/yukawa_train.pt",
                   map_location="cpu", weights_only=False)
    Xb, kap = d["Xb"][:400].numpy(), d["kappa"][:400].numpy()
    ss = []
    for i in range(Xb.shape[0]):
        r = np.linalg.norm(Xb[i][:, None, :] - Xb[i][None, :, :], axis=-1)
        ss.append((kap[i] * r[np.triu_indices(r.shape[0], 1)])[::37])
    return np.concatenate(ss)


def main():
    dev = torch.device("cpu")
    exact = S_GRID * kn(1, np.maximum(S_GRID, 1e-300))
    sdens = train_s_density()
    hist, edges = np.histogram(sdens, bins=40, range=(0, S_GRID.max()))
    dense_bins = hist > np.percentile(hist[hist > 0], 50)
    res = {"models": {}}
    rows = []
    for m in ("mix8", "radial_mlp"):
        errs_in, laws = [], []
        for s in (0, 1, 2):
            ck = torch.load(CK / f"data_s{s}_{m}" / "best.pt", map_location="cpu", weights_only=False)
            mod = build_model(ck["config"]); mod.load_state_dict(ck["state_dict"]); mod.eval()
            with torch.no_grad():
                law = mod.m_law(torch.tensor(S_GRID, dtype=torch.float32)).numpy().astype(float)
            laws.append(law)
            for sv, lv, ev in zip(S_GRID, law, exact):
                rows.append({"model": m, "seed": s, "s": sv, "m_theta": lv, "m_star": ev})
        laws = np.array(laws)
        in_mask = S_GRID <= S_IN_MAX
        big = exact >= 1e-2
        rel = np.abs(laws - exact[None]) / np.maximum(exact[None], 1e-12)
        bin_idx = np.clip(np.digitize(S_GRID, edges) - 1, 0, 39)
        dmask = dense_bins[bin_idx] & in_mask & big
        smask = ~dense_bins[bin_idx] & in_mask & big
        res["models"][m] = {
            "rel_median_inrange": float(np.median(rel[:, in_mask & big])),
            "rel_max_inrange": float(rel[:, in_mask & big].max()),
            "cross_seed_std_of_law": float(laws[:, in_mask].std(axis=0).max()),
            "rel_median_dense_s": float(np.median(rel[:, dmask])) if dmask.any() else None,
            "rel_median_sparse_s": float(np.median(rel[:, smask])) if smask.any() else None,
            "oor_max_law_value": float(laws[:, ~in_mask].max()),
            "oor_min_law_value": float(laws[:, ~in_mask].min()),
            "per_seed_rel_median": [float(np.median(rel[i, in_mask & big])) for i in range(3)],
        }
    # correlation with confirmation accuracy (bulk[1,4], per seed)
    acc = defaultdict(list)
    with (OUT / "raw.csv").open() as f:
        for r in csv.DictReader(f):
            if (r["method"] in ("mix8", "radial_mlp") and float(r["kappa"]) in K14
                    and float(r["margin"]) == 0.12 and int(r["n_b"]) == 128):
                acc[(r["method"], int(r["data_seed"]))].append(float(r["rel_l2"]))
    for m in ("mix8", "radial_mlp"):
        sol = [float(np.mean(acc[(m, s)])) for s in (0, 1, 2)]
        law = res["models"][m]["per_seed_rel_median"]
        res["models"][m]["per_seed_bulk14"] = sol
        if len(set(law)) > 1 and len(set(sol)) > 1:
            res["models"][m]["law_vs_solution_corr"] = float(np.corrcoef(law, sol)[0, 1])
    with (OUT / "law_curves.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    (OUT / "law_recovery.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
