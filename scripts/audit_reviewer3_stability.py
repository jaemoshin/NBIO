#!/usr/bin/env python
"""Full-grid discrete perturbation and solve-stability audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get("NBIO_SCREENING_SOURCE", ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reviewer3_exact_screened import ExactScreenedBIE, laplace_matrix  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402

DATA_ROOT = ROOT / "data_cache" / "radial_factorization_ablation_v1" / "holdout"
DATA_MANIFEST = ROOT / "protocols" / "radial_factorization_ablation_data_manifest.json"
CHECKPOINT_MANIFEST = ROOT / "results" / "screening_law" / "radial_paperization_v1" / "checkpoint_manifest.json"
RESULT_ROOT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1"
KAPPAS = (1.2, 1.75, 2.4, 3.1, 3.8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def load_models(device):
    manifest = json.loads(CHECKPOINT_MANIFEST.read_text())
    output = []
    for record in manifest["records"]:
        if record["method"] != "radial_normal":
            continue
        path = ROOT / record["path"]
        if not path.exists():
            path = FROZEN_ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        output.append((record["seed_index"], model.to(device).eval()))
    if len(output) != 3:
        raise RuntimeError("expected three primary radial checkpoints")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("in_dist", "highmode", "rough", "deform"), required=True)
    parser.add_argument("--n-boundary", type=int, choices=(64, 128, 256), required=True)
    args = parser.parse_args()
    output = RESULT_ROOT / "raw" / f"stability_{args.family}_nb{args.n_boundary}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    exact_model = ExactScreenedBIE().to(device).eval()
    models = load_models(device)
    data_manifest = json.loads(DATA_MANIFEST.read_text())
    data_hashes = {row["path"]: row["sha256"] for row in data_manifest["files"]}
    rows = []
    with torch.no_grad():
        for kappa in KAPPAS:
            path = DATA_ROOT / f"point_source_{args.family}_k{tag(kappa)}_nb{args.n_boundary}.pt"
            relative = str(path.relative_to(ROOT))
            if relative not in data_hashes or sha256(path) != data_hashes[relative]:
                raise RuntimeError(f"unsealed registered shard: {path}")
            shard = torch.load(path, map_location="cpu", weights_only=False)
            batch = {
                key: value.to(device) for key, value in shard.items()
                if key != "bem_uq"
            }
            laplace = laplace_matrix(batch)
            exact = exact_model.assemble(batch)
            exact_condition = torch.linalg.cond(exact)
            for seed_index, model in models:
                matrix = model.assemble(batch)
                density = torch.linalg.solve(matrix, batch["g"].unsqueeze(-1)).squeeze(-1)
                residual = (
                    torch.linalg.vector_norm(
                        torch.bmm(matrix, density.unsqueeze(-1)).squeeze(-1) - batch["g"],
                        dim=-1,
                    ) / torch.linalg.vector_norm(batch["g"], dim=-1).clamp_min(1e-12)
                )
                condition = torch.linalg.cond(matrix)
                perturbation = torch.linalg.solve(laplace, matrix - laplace)
                margin = torch.linalg.matrix_norm(perturbation, ord=2)
                finite = (
                    torch.isfinite(residual) & torch.isfinite(condition)
                    & torch.isfinite(exact_condition) & torch.isfinite(margin)
                )
                for sample_id in range(len(shard["Xb"])):
                    rows.append({
                        "method": "radial_normal",
                        "seed_index": seed_index,
                        "family": args.family,
                        "kappa": kappa,
                        "n_boundary": args.n_boundary,
                        "sample_id": sample_id,
                        "laplace_perturbation_margin": float(margin[sample_id]),
                        "margin_below_one": bool(margin[sample_id] < 1.0),
                        "condition_number": float(condition[sample_id]),
                        "exact_condition_number": float(exact_condition[sample_id]),
                        "condition_ratio": float(condition[sample_id] / exact_condition[sample_id]),
                        "relative_solve_residual": float(residual[sample_id]),
                        "finite": bool(finite[sample_id]),
                    })
            print(
                f"[{args.family} nb={args.n_boundary}] kappa={kappa:g}", flush=True,
            )
    if len(rows) != 5 * 32 * 3:
        raise RuntimeError(f"coverage mismatch: {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
