#!/usr/bin/env python
"""Blind family-sharded evaluation for the matched radial factorization ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import kn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402

DATA_ROOT = ROOT / "data_cache" / "radial_factorization_ablation_v1" / "holdout"
RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_paperization_v1"
DATA_MANIFEST_PATH = ROOT / "protocols" / "radial_factorization_ablation_data_manifest.json"
KAPPAS = (1.2, 1.75, 2.4, 3.1, 3.8)
LOADS = ("point_source", "fourier")
RESOLUTIONS = (64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError("refusing to write empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_models(device: torch.device):
    manifest_path = RESULT_ROOT / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT":
        raise RuntimeError("checkpoints are not sealed")
    models = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        models.append((record["method"], record["seed_index"], model.to(device).eval()))
    return models


def exact_matrix(shard: dict[str, torch.Tensor], index: int, kappa: float) -> np.ndarray:
    boundary = shard["Xb"][index].double().numpy()
    normals = shard["Nb"][index].double().numpy()
    weights = shard["wb"][index].double().numpy()
    curvature = shard["curv"][index].double().numpy()
    difference = boundary[None] - boundary[:, None]
    radius = np.linalg.norm(difference, axis=-1)
    numerator = np.sum(difference * normals[None], axis=-1)
    safe = np.maximum(radius, 1e-12)
    if kappa == 0.0:
        kernel = -(1.0 / (2.0 * np.pi)) * numerator / np.square(safe)
    else:
        kernel = -(kappa / (2.0 * np.pi)) * kn(1, kappa * safe) * numerator / safe
    np.fill_diagonal(kernel, -curvature / (4.0 * np.pi))
    return -0.5 * np.eye(len(boundary)) + kernel * weights[None]


def assembled_matrix(model, batch):
    if hasattr(model, "assemble"):
        return model.assemble(batch)
    return model.bio.assemble(
        batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
        batch["kappa"], model.jump,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("in_dist", "highmode", "rough", "deform"), required=True)
    args = parser.parse_args()
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    data_manifest = json.loads(DATA_MANIFEST_PATH.read_text())
    data_records = {record["path"]: record["sha256"] for record in data_manifest["files"]}
    models = load_models(device)
    rows = []
    stability = []
    for kappa in KAPPAS:
        for load in LOADS:
            for n_boundary in RESOLUTIONS:
                path = DATA_ROOT / f"{load}_{args.family}_k{tag(kappa)}_nb{n_boundary}.pt"
                relative = str(path.relative_to(ROOT))
                if relative not in data_records or sha256(path) != data_records[relative]:
                    raise RuntimeError(f"unsealed or changed holdout shard: {path}")
                shard = torch.load(path, map_location="cpu", weights_only=False)
                target = shard["uq"]
                exact_errors = (
                    torch.linalg.vector_norm(shard["bem_uq"] - target, dim=-1)
                    / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
                ).numpy()
                for sample_id, error in enumerate(exact_errors):
                    rows.append({
                        "method": "exact_yukawa_bie",
                        "seed_index": -1,
                        "family": args.family,
                        "load_family": load,
                        "kappa": kappa,
                        "n_boundary": n_boundary,
                        "sample_id": sample_id,
                        "rel_l2": float(error),
                        "finite": bool(np.isfinite(error)),
                    })
                batch = {key: value.to(device) for key, value in shard.items() if key != "bem_uq"}
                for method, seed_index, model in models:
                    try:
                        prediction = model(batch)
                        errors = (
                            torch.linalg.vector_norm(prediction - batch["uq"], dim=-1)
                            / torch.linalg.vector_norm(batch["uq"], dim=-1).clamp_min(1e-12)
                        ).cpu().numpy()
                    except (RuntimeError, torch.linalg.LinAlgError):
                        errors = np.full(len(target), np.nan)
                    for sample_id, error in enumerate(errors):
                        rows.append({
                            "method": method,
                            "seed_index": seed_index,
                            "family": args.family,
                            "load_family": load,
                            "kappa": kappa,
                            "n_boundary": n_boundary,
                            "sample_id": sample_id,
                            "rel_l2": float(error),
                            "finite": bool(np.isfinite(error)),
                        })
                    if args.family == "in_dist" and load == "point_source" and n_boundary == 128:
                        diagnostic = {key: value[:8] for key, value in batch.items()}
                        matrix = assembled_matrix(model, diagnostic)
                        density = torch.linalg.solve(
                            matrix, diagnostic["g"].unsqueeze(-1),
                        ).squeeze(-1)
                        residual = (
                            torch.linalg.vector_norm(
                                torch.bmm(matrix, density.unsqueeze(-1)).squeeze(-1)
                                - diagnostic["g"], dim=-1,
                            )
                            / torch.linalg.vector_norm(diagnostic["g"], dim=-1).clamp_min(1e-12)
                        ).cpu().numpy()
                        condition = torch.linalg.cond(matrix).cpu().numpy()
                        exact = np.stack([
                            exact_matrix(shard, sample_id, kappa) for sample_id in range(8)
                        ])
                        exact_condition = np.linalg.cond(exact)
                        laplace = torch.as_tensor(
                            np.stack([
                                exact_matrix(shard, sample_id, 0.0)
                                for sample_id in range(8)
                            ]),
                            device=matrix.device, dtype=matrix.dtype,
                        )
                        perturbation = torch.linalg.solve(laplace, matrix - laplace)
                        perturbation_margin = torch.linalg.matrix_norm(
                            perturbation, ord=2,
                        ).cpu().numpy()
                        for sample_id in range(8):
                            stability.append({
                                "method": method,
                                "seed_index": seed_index,
                                "kappa": kappa,
                                "sample_id": sample_id,
                                "relative_residual": float(residual[sample_id]),
                                "condition_number": float(condition[sample_id]),
                                "exact_condition_number": float(exact_condition[sample_id]),
                                "condition_ratio": float(
                                    condition[sample_id] / exact_condition[sample_id]
                                ),
                                "laplace_perturbation_margin": float(
                                    perturbation_margin[sample_id]
                                ),
                            })
                print(
                    f"[{args.family}] kappa={kappa:g} load={load} n_b={n_boundary}",
                    flush=True,
                )
    write_csv(rows, RESULT_ROOT / "raw" / f"{args.family}.csv")
    if stability:
        write_csv(stability, RESULT_ROOT / "raw" / "stability.csv")
    (RESULT_ROOT / "raw" / f"{args.family}_manifest.json").write_text(
        json.dumps({
            "family": args.family,
            "row_count": len(rows),
            "stability_row_count": len(stability),
            "checkpoint_manifest_sha256": sha256(RESULT_ROOT / "checkpoint_manifest.json"),
        }, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
