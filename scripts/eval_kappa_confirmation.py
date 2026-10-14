#!/usr/bin/env python
"""Confirm full-prior family utility and evaluate a validation-selected router."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from eval_kappa_conditioned import (  # noqa: E402
    checkpoint_model, instance_rel, paired_regime, save_csv, save_json, summarize,
)
from gen_kappa_confirmation import (  # noqa: E402
    EXTRAPOLATION_KAPPAS, INTERPOLATION_KAPPAS, MARGINS, RESOLUTIONS, TEST_KAPPAS,
)


THRESHOLDS = tuple(np.arange(0.0, 4.0001, 0.25))
REGIME = {
    kappa: ("interpolation" if kappa in INTERPOLATION_KAPPAS
            else "extrapolation" if kappa in EXTRAPOLATION_KAPPAS
            else "diagnostic")
    for kappa in TEST_KAPPAS
}


def load_models(run_root, device):
    output = {"full_norm": [], "none_norm": []}
    for method in output:
        for seed in (0, 1, 2):
            model, _ = checkpoint_model(run_root / f"{method}_s{seed}/best.pt", device)
            output[method].append(model)
    return output


@torch.no_grad()
def select_threshold(models, validation_path, device, batch_size):
    dataset = BIEDataset(validation_path)
    predictions = {method: [[] for _ in range(3)] for method in models}
    targets = []
    kappas = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        batch = to_device(batch, device)
        targets.append(batch["uq"].cpu())
        kappas.append(batch["kappa"].cpu())
        for method, method_models in models.items():
            for seed, model in enumerate(method_models):
                predictions[method][seed].append(model(batch).cpu())
    target = torch.cat(targets)
    kappa = torch.cat(kappas)
    errors = {
        method: torch.stack([
            instance_rel(torch.cat(chunks), target) for chunks in seed_chunks
        ])
        for method, seed_chunks in predictions.items()
    }
    table = []
    for threshold in THRESHOLDS:
        use_full = kappa[None, :] <= threshold
        routed = torch.where(use_full, errors["full_norm"], errors["none_norm"])
        table.append({
            "threshold": float(threshold),
            "validation_mean_rel_l2": float(routed.mean()),
            "fraction_full_prior": float(use_full.float().mean()),
        })
    table.sort(key=lambda row: (row["validation_mean_rel_l2"], row["threshold"]))
    return table[0]["threshold"], table


def shard_path(data_dir, kappa, margin, n_b):
    kappa_tag = str(kappa).replace(".", "p")
    margin_tag = f"m{int(round(1000 * margin)):03d}"
    return data_dir / f"test_k{kappa_tag}_{margin_tag}_nb{n_b}.pt"


@torch.no_grad()
def evaluate(models, threshold, data_dir, device, batch_size):
    output = []
    for kappa in TEST_KAPPAS:
        for margin in MARGINS:
            for n_b in RESOLUTIONS:
                dataset = BIEDataset(shard_path(data_dir, kappa, margin, n_b))
                for method, key in (
                    ("correct_yukawa_bie", "bem_uq"),
                    ("fixed_laplace_bie", "laplace_bem_uq"),
                ):
                    errors = instance_rel(dataset.d[key], dataset.d["uq"]).numpy()
                    for sample_id, error in enumerate(errors):
                        output.append({
                            "method": method, "seed": -1, "kappa": kappa,
                            "kappa_regime": REGIME[kappa], "query_margin": margin,
                            "n_b": n_b, "sample_id": sample_id, "rel_l2": float(error),
                        })
                loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )
                for seed in range(3):
                    offset = 0
                    for batch in loader:
                        batch = to_device(batch, device)
                        full = models["full_norm"][seed](batch)
                        none = models["none_norm"][seed](batch)
                        predictions = {
                            "full_norm": full,
                            "none_norm": none,
                            "router": full if kappa <= threshold else none,
                        }
                        for method, prediction in predictions.items():
                            errors = instance_rel(prediction, batch["uq"]).cpu().numpy()
                            for index, error in enumerate(errors):
                                output.append({
                                    "method": method, "seed": seed, "kappa": kappa,
                                    "kappa_regime": REGIME[kappa], "query_margin": margin,
                                    "n_b": n_b, "sample_id": offset + index,
                                    "rel_l2": float(error),
                                })
                        offset += len(errors)
                print(
                    f"[confirm] kappa={kappa:g} regime={REGIME[kappa]} "
                    f"margin={margin:g} nb={n_b}", flush=True
                )
    return output


def analyze(rows, threshold):
    comparisons = []
    seed = 20261001
    for regime in ("interpolation", "extrapolation"):
        for margin, n_b in ((0.12, 128), (0.02, 256)):
            comparisons.append(paired_regime(
                rows, "full_norm", "none_norm", regime, margin, n_b, seed
            ))
            seed += 1
            for control in ("none_norm", "full_norm"):
                comparisons.append(paired_regime(
                    rows, "router", control, regime, margin, n_b, seed
                ))
                seed += 1
    return {
        "protocol": {
            "primary_target": "full_norm", "primary_control": "none_norm",
            "router_threshold": threshold,
            "threshold_selection": "minimum validation mean over fixed 0.25 grid",
            "test_geometry_seed": 9303,
            "paired_unit": "geometry cluster; each cluster averages kappas and three seeds",
        },
        "paired_comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/kappa_continuous")
    parser.add_argument("--validation", default="data_cache/kappa_continuous/yukawa_val.pt")
    parser.add_argument("--data-dir", default="data_cache/kappa_confirmation")
    parser.add_argument("--out-dir", default="results/kappa_confirmation")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    configure_torch()
    device = get_device(True)
    models = load_models(ROOT / args.run_root, device)
    threshold, selection = select_threshold(
        models, ROOT / args.validation, device, args.batch_size
    )
    print(f"[selection] validation-only router threshold={threshold:g}", flush=True)
    output = evaluate(models, threshold, ROOT / args.data_dir, device, args.batch_size)
    summary_rows = summarize(output)
    analysis = analyze(output, threshold)
    out = ROOT / args.out_dir
    save_csv(selection, out / "router_validation_selection.csv")
    save_csv(output, out / "raw.csv")
    save_csv(summary_rows, out / "summary.csv")
    save_json(analysis, out / "analysis.json")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
