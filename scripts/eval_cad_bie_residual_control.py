#!/usr/bin/env python
"""Paired same-case control for the CAD BIE residual experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cad_overnight_tranche import build_training_cases
from scripts.mcb_thermal_benchmark import save_json
from scripts.top5_benchmark_screen import matrix_summary
from scripts.train_cad_sota_headtohead import (
    BUILDERS,
    coarse_single_layer_potential,
    make_batch,
    rel_l2_t,
    split_cases,
)


OUT = ROOT / "results" / "official_sota_headtohead"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fusion", "abc"], required=True)
    parser.add_argument("--model", default="bie_tiny_residual_ngf")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result_path = OUT / f"{args.dataset}_{args.model}_seed{args.seed}.json"
    checkpoint_path = OUT / f"{args.dataset}_{args.model}_seed{args.seed}.pt"
    source = json.loads(result_path.read_text())
    cfg = argparse.Namespace(**source["args"])
    data_seed = cfg.seed if getattr(cfg, "data_seed", None) is None else cfg.data_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BUILDERS[args.model]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    if args.smoke:
        generator = torch.Generator(device=device).manual_seed(0)
        xb = torch.randn(1, 32, 3, generator=generator, device=device)
        xb = xb / torch.linalg.vector_norm(xb, dim=-1, keepdim=True)
        wb = torch.full((1, 32), 4.0 * np.pi / 32, device=device)
        gb = xb[..., 0]
        xq = torch.randn(1, 16, 3, generator=generator, device=device)
        xq = 1.5 * xq / torch.linalg.vector_norm(xq, dim=-1, keepdim=True)
        with torch.no_grad():
            coarse = coarse_single_layer_potential(xb, wb, gb, xq)
            learned = model(xb, wb, xb, gb, xq)
        assert torch.isfinite(coarse).all() and torch.isfinite(learned).all()
        print("paired-control smoke passed")
        return

    cases = build_training_cases(
        cfg.dataset,
        cfg.n_meshes,
        cfg.families,
        cfg.n_surface,
        cfg.n_ref,
        cfg.n_query,
        data_seed,
    )
    _, test_cases = split_cases(cases)

    rng = np.random.default_rng(data_seed + 999)
    rows = []
    with torch.no_grad():
        for case in test_cases:
            batch = make_batch([case], rng, device, cfg.eval_surface, cfg.eval_query)
            u0 = coarse_single_layer_potential(batch["xb"], batch["wb"], batch["gb"], batch["xq"])
            learned = model(batch["xb"], batch["wb"], batch["nb"], batch["gb"], batch["xq"])
            coarse_error = float(rel_l2_t(u0, batch["u"])[0].cpu())
            learned_error = float(rel_l2_t(learned, batch["u"])[0].cpu())
            rows.append({
                "mesh": case.mesh,
                "family": case.family,
                "coarse_bie_rel_l2": coarse_error,
                "learned_residual_rel_l2": learned_error,
                "learned_minus_coarse": learned_error - coarse_error,
            })

    coarse = [row["coarse_bie_rel_l2"] for row in rows]
    learned = [row["learned_residual_rel_l2"] for row in rows]
    coarse_summary = matrix_summary(coarse)
    learned_summary = matrix_summary(learned)
    improves_median = learned_summary["median"] < coarse_summary["median"]
    improves_p90 = learned_summary["p90"] < coarse_summary["p90"]
    output = {
        "status": "ok",
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "data_seed": data_seed,
        "source_result": str(result_path.relative_to(ROOT)),
        "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "coarse_bie": coarse_summary,
        "learned_residual": learned_summary,
        "paired_win_fraction": float(np.mean(np.asarray(learned) < np.asarray(coarse))),
        "rows": rows,
        "kill_criterion": "Do not claim learned CAD improvement unless median or p90 improves on this paired control.",
        "verdict": "CONTINUE" if (improves_median or improves_p90) else "KILL_LEARNED_CAD_CLAIM",
        "exact_command": (
            f"python scripts/eval_cad_bie_residual_control.py --dataset {args.dataset} "
            f"--model {args.model} --seed {args.seed}"
        ),
    }
    output_path = OUT / f"{args.dataset}_{args.model}_seed{args.seed}_paired_control.json"
    save_json(output, output_path)


if __name__ == "__main__":
    main()
