#!/usr/bin/env python
"""Blind family-sharded evaluation for Screening-Law NBIO confirmatory v1.

Per-instance rows are written without aggregate metrics. Only the locked final
aggregator is allowed to summarize the confirmation matrix.
"""
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

DATA_ROOT = ROOT / "data_cache" / "screening_confirmatory_v1" / "confirmation"
RESULT_ROOT = ROOT / "results" / "screening_law" / "confirmatory_v1"
KAPPAS = (0.15, 0.45, 0.85, 1.35, 1.9, 2.55, 3.25, 3.9, 4.5, 5.25, 6.0)
MARGINS = (("m120", 0.12), ("m020", 0.02))
RESOLUTIONS = (64, 128, 256)
LOADS = ("point_source", "fourier")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kappa_tag(kappa: float) -> str:
    return str(kappa).replace(".", "p")


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"write-once evaluation artifact exists: {path}")
    if not rows:
        raise ValueError("refusing to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified_data(path: Path, records: dict) -> dict[str, torch.Tensor]:
    relative = str(path.relative_to(ROOT))
    if relative not in records:
        raise RuntimeError(f"unregistered confirmation shard: {relative}")
    if sha256(path) != records[relative]:
        raise RuntimeError(f"confirmation shard hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_models(device: torch.device):
    manifest_path = RESULT_ROOT / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "CHECKPOINTS_FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("checkpoints were not sealed")
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


def exact_yukawa_matrix(shard: dict[str, torch.Tensor], index: int, kappa: float):
    boundary = shard["Xb"][index].double().numpy()
    normals = shard["Nb"][index].double().numpy()
    weights = shard["wb"][index].double().numpy()
    curvature = shard["curv"][index].double().numpy()
    difference = boundary[None] - boundary[:, None]
    radius = np.linalg.norm(difference, axis=-1)
    numerator = np.sum(difference * normals[None], axis=-1)
    safe_radius = np.maximum(radius, 1e-12)
    kernel = -(kappa / (2 * np.pi)) * kn(1, kappa * safe_radius) * numerator / safe_radius
    np.fill_diagonal(kernel, -curvature / (4 * np.pi))
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
    parser.add_argument("--family", required=True, choices=("in_dist", "highmode", "rough", "deform"))
    args = parser.parse_args()
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    data_manifest = json.loads(
        (ROOT / "protocols" / "screening_law_confirmatory_data_manifest.json").read_text()
    )
    data_records = {record["path"]: record["sha256"] for record in data_manifest["files"]}
    models = load_models(device)
    raw_rows = []
    stability_rows = []
    for kappa in KAPPAS:
        regime = "in_range" if kappa <= 4.0 else "out_of_range"
        for margin_tag, margin in MARGINS:
            for n_boundary in RESOLUTIONS:
                for load in LOADS:
                    path = DATA_ROOT / (
                        f"{load}_{args.family}_k{kappa_tag(kappa)}_"
                        f"{margin_tag}_nb{n_boundary}.pt"
                    )
                    shard = verified_data(path, data_records)
                    target = shard["uq"]
                    correct_errors = (
                        torch.linalg.vector_norm(shard["bem_uq"] - target, dim=-1)
                        / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
                    ).numpy()
                    for sample_id, error in enumerate(correct_errors):
                        raw_rows.append({
                            "method": "correct_yukawa_bie", "seed_index": -1,
                            "family": args.family, "load_family": load,
                            "kappa": kappa, "regime": regime,
                            "query_margin": margin, "n_boundary": n_boundary,
                            "sample_id": sample_id, "rel_l2": float(error),
                            "finite": bool(np.isfinite(error)),
                        })
                    batch = {
                        key: value.to(device)
                        for key, value in shard.items() if key != "bem_uq"
                    }
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
                            raw_rows.append({
                                "method": method, "seed_index": seed_index,
                                "family": args.family, "load_family": load,
                                "kappa": kappa, "regime": regime,
                                "query_margin": margin, "n_boundary": n_boundary,
                                "sample_id": sample_id, "rel_l2": float(error),
                                "finite": bool(np.isfinite(error)),
                            })
                        if (args.family == "in_dist" and load == "point_source"
                                and margin == 0.12 and n_boundary == 128):
                            diagnostic = {key: value[:8] for key, value in batch.items()}
                            matrix = assembled_matrix(model, diagnostic)
                            density = torch.linalg.solve(matrix, diagnostic["g"].unsqueeze(-1)).squeeze(-1)
                            residual = (
                                torch.linalg.vector_norm(
                                    torch.bmm(matrix, density.unsqueeze(-1)).squeeze(-1)
                                    - diagnostic["g"], dim=-1,
                                )
                                / torch.linalg.vector_norm(diagnostic["g"], dim=-1).clamp_min(1e-12)
                            ).cpu().numpy()
                            learned_condition = torch.linalg.cond(matrix).cpu().numpy()
                            exact = np.stack([
                                exact_yukawa_matrix(shard, sample_id, kappa)
                                for sample_id in range(8)
                            ])
                            exact_tensor = torch.from_numpy(exact).to(matrix)
                            exact_condition = np.linalg.cond(exact)
                            perturbation_margin = torch.linalg.matrix_norm(
                                torch.linalg.solve(exact_tensor, matrix - exact_tensor), ord=2,
                            ).cpu().numpy()
                            for sample_id in range(8):
                                stability_rows.append({
                                    "method": method, "seed_index": seed_index,
                                    "kappa": kappa, "sample_id": sample_id,
                                    "relative_residual": float(residual[sample_id]),
                                    "condition_number": float(learned_condition[sample_id]),
                                    "exact_condition_number": float(exact_condition[sample_id]),
                                    "condition_ratio": float(
                                        learned_condition[sample_id] / exact_condition[sample_id]
                                    ),
                                    "discrete_perturbation_margin": float(
                                        perturbation_margin[sample_id]
                                    ),
                                })
                    print(
                        f"[{args.family}] kappa={kappa:g} load={load} "
                        f"margin={margin:g} n_b={n_boundary}", flush=True,
                    )
    raw_path = RESULT_ROOT / "raw" / f"{args.family}.csv"
    write_csv(raw_rows, raw_path)
    if stability_rows:
        write_csv(stability_rows, RESULT_ROOT / "raw" / "stability.csv")
    manifest_path = RESULT_ROOT / "raw" / f"{args.family}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps({
        "family": args.family,
        "row_count": len(raw_rows),
        "stability_row_count": len(stability_rows),
        "checkpoint_manifest_sha256": sha256(RESULT_ROOT / "checkpoint_manifest.json"),
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote sealed family rows: {len(raw_rows)}")


if __name__ == "__main__":
    main()
