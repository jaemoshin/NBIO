#!/usr/bin/env python
"""Evaluate continuous-kappa family models on the fresh holdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from eval_kappa_conditioned import (  # noqa: E402
    checkpoint_model, instance_rel, paired_regime, save_csv, save_json, summarize,
)
from gen_kappa_continuous import (  # noqa: E402
    PRIMARY_EXTRAPOLATION, PRIMARY_INTERPOLATION, QUERY_MARGINS,
    RESOLUTIONS, TEST_KAPPAS,
)


RUNS = (
    ("none_norm", "none_norm"),
    ("full_norm", "full_norm"),
    ("full_sat", "full_sat"),
    ("adaptive_norm", "adaptive_norm"),
    ("adaptive_sat", "adaptive_sat"),
)
REGIME = {
    kappa: ("interpolation" if kappa in PRIMARY_INTERPOLATION
            else "extrapolation" if kappa in PRIMARY_EXTRAPOLATION
            else "endpoint_diagnostic")
    for kappa in TEST_KAPPAS
}


def shard_path(data_dir, kappa, margin, n_b):
    kappa_tag = str(kappa).replace(".", "p")
    margin_tag = f"m{int(round(1000 * margin)):03d}"
    return data_dir / f"test_k{kappa_tag}_{margin_tag}_nb{n_b}.pt"


@torch.no_grad()
def evaluate(run_root, data_dir, seeds, batch_size, device):
    models = []
    training = []
    for method, directory_prefix in RUNS:
        for seed in seeds:
            directory = run_root / f"{directory_prefix}_s{seed}"
            model, checkpoint = checkpoint_model(directory / "best.pt", device)
            models.append((method, seed, model))
            history = json.loads((directory / "history.json").read_text())
            gate_rate = float("nan")
            if getattr(model.bio, "log_gate_rate", None) is not None:
                gate_rate = float(torch.nn.functional.softplus(
                    model.bio.log_gate_rate.detach()
                ).cpu())
            kwargs = checkpoint["config"]["model"]["kwargs"]
            training.append({
                "method": method, "seed": seed,
                "prior_mode": kwargs["prior_mode"],
                "residual_scale": kwargs.get("residual_scale", "normalized"),
                "n_params": int(history["n_params"]),
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_rel_l2": float(history["best_val_rel_l2"]),
                "learned_gate_rate": gate_rate,
                "checkpoint": str((directory / "best.pt").relative_to(ROOT)),
            })

    output = []
    for kappa in TEST_KAPPAS:
        for margin in QUERY_MARGINS:
            for n_b in RESOLUTIONS:
                dataset = BIEDataset(shard_path(data_dir, kappa, margin, n_b))
                for method, tensor_key in (
                    ("correct_yukawa_bie", "bem_uq"),
                    ("fixed_laplace_bie", "laplace_bem_uq"),
                ):
                    errors = instance_rel(dataset.d[tensor_key], dataset.d["uq"]).numpy()
                    for sample_id, error in enumerate(errors):
                        output.append({
                            "method": method, "seed": -1, "kappa": kappa,
                            "kappa_regime": REGIME[kappa], "query_margin": margin,
                            "n_b": n_b, "sample_id": sample_id, "rel_l2": float(error),
                        })
                loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )
                for method, seed, model in models:
                    offset = 0
                    for batch in loader:
                        batch = to_device(batch, device)
                        errors = instance_rel(model(batch), batch["uq"]).cpu().numpy()
                        for index, error in enumerate(errors):
                            output.append({
                                "method": method, "seed": seed, "kappa": kappa,
                                "kappa_regime": REGIME[kappa], "query_margin": margin,
                                "n_b": n_b, "sample_id": offset + index,
                                "rel_l2": float(error),
                            })
                        offset += len(errors)
                print(
                    f"[eval] kappa={kappa:g} regime={REGIME[kappa]} "
                    f"margin={margin:g} nb={n_b}", flush=True
                )
    return output, training


def analyze(rows):
    comparisons = []
    seed = 20260901
    for regime in ("interpolation", "extrapolation"):
        for margin in QUERY_MARGINS:
            for n_b in (128, 256):
                for control in ("none_norm", "full_norm", "full_sat", "adaptive_norm"):
                    comparisons.append(paired_regime(
                        rows, "adaptive_sat", control, regime, margin, n_b, seed
                    ))
                    seed += 1
    return {
        "protocol": {
            "train_distribution": "80% stratified uniform kappa in (0,4), 10% each at 0 and 4",
            "primary_interpolation_kappas": list(PRIMARY_INTERPOLATION),
            "primary_extrapolation_kappas": list(PRIMARY_EXTRAPOLATION),
            "target_method": "adaptive_sat",
            "paired_unit": "geometry cluster; each cluster averages kappas and three seeds",
            "fresh_test_seed": 9203,
        },
        "paired_regime_comparisons": comparisons,
    }


def make_figure(summary_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = (
        ("correct_yukawa_bie", "Correct BIE", "black", "--"),
        ("fixed_laplace_bie", "Fixed Laplace BIE", "#999999", ":"),
        ("none_norm", "No prior", "#7570b3", "-"),
        ("full_norm", "Full prior", "#d95f02", "-"),
        ("full_sat", "Full + sat. residual", "#e6ab02", "-"),
        ("adaptive_norm", "Adaptive prior", "#66a61e", "-"),
        ("adaptive_sat", "Adaptive + sat. residual", "#1b9e77", "-"),
    )
    lookup = {}
    for row in summary_rows:
        lookup[(float(row["query_margin"]), int(row["n_b"]), row["method"],
                float(row["kappa"]))] = float(row["mean_rel_l2"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, margin, n_b, title in (
        (axes[0], 0.12, 128, "Bulk, train resolution"),
        (axes[1], 0.12, 256, "Bulk, refined boundary"),
        (axes[2], 0.02, 256, "Near boundary"),
    ):
        for method, label, color, linestyle in methods:
            values = [lookup[(margin, n_b, method, kappa)] for kappa in TEST_KAPPAS]
            ax.plot(TEST_KAPPAS, values, marker="o", markersize=3, linewidth=1.4,
                    color=color, linestyle=linestyle, label=label)
        ax.axvspan(4.0, 5.5, color="#eeeeee", zorder=-2)
        ax.axvline(4.0, color="#bbbbbb", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"$\kappa$")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("relative L2 error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/kappa_continuous")
    parser.add_argument("--data-dir", default="data_cache/kappa_continuous")
    parser.add_argument("--out-dir", default="results/kappa_continuous_eval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    configure_torch()
    device = get_device(True)
    output, training = evaluate(
        ROOT / args.run_root, ROOT / args.data_dir, args.seeds, args.batch_size, device
    )
    summary_rows = summarize(output)
    analysis = analyze(output)
    out = ROOT / args.out_dir
    save_csv(output, out / "raw.csv")
    save_csv(summary_rows, out / "summary.csv")
    save_csv(training, out / "training_summary.csv")
    save_json(analysis, out / "analysis.json")
    make_figure(summary_rows, ROOT / "figures/kappa_continuous_transfer.png")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
