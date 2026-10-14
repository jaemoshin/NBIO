#!/usr/bin/env python3
"""Evaluate fresh bulk holdouts for asymptotic and transformer work packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models import build_model  # noqa: E402
from src.models.layers import double_layer_kernel  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from tmlr_revision_validation import FAMILIES  # noqa: E402

RESULT_ROOT = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
)
DATA_ROOT = ROOT / "data_cache" / "radial_transfer_core_v1"
DATA_MANIFEST = ROOT / "protocols" / "radial_transfer_core_data_manifest.json"
FROZEN_MANIFEST = (
    ROOT / "results" / "screening_law" / "radial_quadratic_v1"
    / "checkpoint_manifest.json"
)
SPECIFICATIONS = {
    "asymptotic": {
        "data_study": "asymptotic",
        "kappas": (1.05, 1.65, 2.35, 3.05, 3.75),
        "resolutions": (128, 256, 512),
        "loads": ("point_source", "fourier_bessel"),
        "new_manifest": RESULT_ROOT / "asymptotic_smoke" / "checkpoint_manifest.json",
        "frozen_methods": ("radial_normal",),
        "new_method": "radial_asymptotic",
        "output": RESULT_ROOT / "asymptotic_smoke",
    },
    "transformer": {
        "data_study": "transformer",
        "kappas": (1.1, 1.7, 2.3, 3.0, 3.7),
        "resolutions": (64, 128, 256),
        "loads": ("point_source", "fourier_bessel"),
        "new_manifest": RESULT_ROOT / "transformer" / "checkpoint_manifest.json",
        "frozen_methods": (
            "radial_normal", "generic_matched", "generic_no_prior_matched",
        ),
        "new_method": "kappa_boundary_transformer",
        "output": RESULT_ROOT / "transformer",
    },
    "transformer_v2": {
        "data_study": "transformer",
        "kappas": (1.1, 1.7, 2.3, 3.0, 3.7),
        "resolutions": (64, 128, 256),
        "loads": ("point_source", "fourier_bessel"),
        "new_manifest": RESULT_ROOT / "transformer_v2" / "checkpoint_manifest.json",
        "frozen_methods": (
            "radial_normal", "generic_matched", "generic_no_prior_matched",
        ),
        "new_method": "kappa_boundary_transformer_v2",
        "output": RESULT_ROOT / "transformer_v2",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def load_verified_models(
    manifest_path: Path, methods: tuple[str, ...], device: torch.device,
) -> list[tuple[str, int, torch.nn.Module]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT":
        raise RuntimeError(f"unsealed checkpoint manifest: {manifest_path}")
    models = []
    for record in manifest["records"]:
        if record["method"] not in methods:
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"]).to(
            device=device, dtype=torch.float64,
        ).eval()
        model.load_state_dict(checkpoint["state_dict"])
        models.append((record["method"], record["seed_index"], model))
    expected = 3 * len(methods)
    if len(models) != expected:
        raise RuntimeError(
            f"{manifest_path}: expected {expected} frozen models, found {len(models)}"
        )
    return models


def load_new_models(
    manifest_path: Path, method: str, device: torch.device,
) -> list[tuple[str, int, torch.nn.Module]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT":
        raise RuntimeError(f"unsealed checkpoint manifest: {manifest_path}")
    models = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"]).to(
            device=device, dtype=torch.float64,
        ).eval()
        model.load_state_dict(checkpoint["state_dict"])
        models.append((method, record["seed_index"], model))
    return models


def exact_operators(batch: dict[str, torch.Tensor]):
    boundary = batch["Xb"]
    known = double_layer_kernel(
        boundary, boundary, batch["Nb"],
        curv_s=batch["curv"], self_kernel=True,
    )
    radius = torch.cdist(boundary, boundary).clamp_min(1e-14)
    scaled = batch["kappa"].reshape(-1, 1, 1) * radius
    multiplier = scaled * torch.special.modified_bessel_k1(scaled)
    diagonal = torch.arange(boundary.shape[1], device=boundary.device)
    multiplier[:, diagonal, diagonal] = 1.0
    kernel = known * multiplier
    identity = torch.eye(
        boundary.shape[1], device=boundary.device, dtype=boundary.dtype,
    ).unsqueeze(0)
    matrix = -0.5 * identity + kernel * batch["wb"][:, None]

    query_known = double_layer_kernel(
        batch["Xq"], boundary, batch["Nb"],
    )
    query_radius = torch.cdist(batch["Xq"], boundary).clamp_min(1e-14)
    query_scaled = batch["kappa"].reshape(-1, 1, 1) * query_radius
    query_multiplier = (
        query_scaled * torch.special.modified_bessel_k1(query_scaled)
    )
    potential = query_known * query_multiplier * batch["wb"][:, None]
    return matrix, potential


def model_matrix(model, batch):
    if hasattr(model, "assemble"):
        return model.assemble(batch)
    if hasattr(model, "bio"):
        return model.bio.assemble(
            batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
            batch["kappa"], model.jump,
        )
    return None


def relative_errors(prediction, target):
    return (
        torch.linalg.vector_norm(prediction - target, dim=-1)
        / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-14)
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=tuple(SPECIFICATIONS), required=True)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    specification = SPECIFICATIONS[args.study]
    output = specification["output"] / "bulk_raw" / f"{args.family}.csv"
    stability_output = (
        specification["output"] / "bulk_raw" / f"{args.family}_stability.csv"
    )
    if output.exists() or stability_output.exists():
        raise FileExistsError(output)

    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    models = load_verified_models(
        FROZEN_MANIFEST, specification["frozen_methods"], device,
    )
    models += load_new_models(
        specification["new_manifest"], specification["new_method"], device,
    )
    data_manifest = json.loads(DATA_MANIFEST.read_text())
    data_hashes = {
        record["path"]: record["sha256"]
        for record in data_manifest["files"]
        if record["study"] == specification["data_study"]
    }

    rows = []
    stability = []
    for kappa in specification["kappas"]:
        for load in specification["loads"]:
            for n_boundary in specification["resolutions"]:
                path = (
                    DATA_ROOT / specification["data_study"] / "holdout"
                    / f"{load}_{args.family}_k{tag(kappa)}_nb{n_boundary}.pt"
                )
                relative_path = str(path.relative_to(ROOT))
                if data_hashes.get(relative_path) != sha256(path):
                    raise RuntimeError(f"unsealed data shard: {path}")
                shard = torch.load(path, map_location="cpu", weights_only=False)
                n_samples = len(shard["uq"])
                for start in range(0, n_samples, args.batch_size):
                    stop = min(start + args.batch_size, n_samples)
                    batch = {
                        key: value[start:stop].to(
                            device=device, dtype=torch.float64,
                        )
                        for key, value in shard.items()
                    }
                    target = batch["uq"]
                    exact_matrix, exact_potential = exact_operators(batch)
                    exact_density = torch.linalg.solve(
                        exact_matrix, batch["g"].unsqueeze(-1),
                    ).squeeze(-1)
                    exact_prediction = torch.bmm(
                        exact_potential, exact_density.unsqueeze(-1),
                    ).squeeze(-1)
                    exact_error = relative_errors(exact_prediction, target)
                    for offset, value in enumerate(exact_error):
                        rows.append({
                            "study": args.study,
                            "method": "exact_yukawa_bie",
                            "seed_index": -1,
                            "family": args.family,
                            "load_family": load,
                            "kappa": kappa,
                            "n_boundary": n_boundary,
                            "sample_id": start + offset,
                            "rel_l2": float(value),
                            "finite": bool(torch.isfinite(value)),
                        })
                    for method, seed, model in models:
                        prediction = model(batch)
                        error = relative_errors(prediction, target)
                        for offset, value in enumerate(error):
                            rows.append({
                                "study": args.study,
                                "method": method,
                                "seed_index": seed,
                                "family": args.family,
                                "load_family": load,
                                "kappa": kappa,
                                "n_boundary": n_boundary,
                                "sample_id": start + offset,
                                "rel_l2": float(value),
                                "finite": bool(torch.isfinite(value)),
                            })
                        matrix = model_matrix(model, batch)
                        if matrix is not None:
                            residual = torch.linalg.vector_norm(
                                torch.bmm(
                                    matrix,
                                    model.solver(
                                        matrix, batch["g"], model.jump,
                                    ).unsqueeze(-1),
                                ).squeeze(-1) - batch["g"],
                                dim=-1,
                            ) / torch.linalg.vector_norm(
                                batch["g"], dim=-1,
                            ).clamp_min(1e-14)
                            if n_boundary == 128:
                                exact_condition = torch.linalg.cond(exact_matrix)
                                condition = torch.linalg.cond(matrix)
                            else:
                                exact_condition = torch.full_like(
                                    residual, float("nan"),
                                )
                                condition = torch.full_like(
                                    residual, float("nan"),
                                )
                            for offset, value in enumerate(residual):
                                stability.append({
                                    "study": args.study,
                                    "method": method,
                                    "seed_index": seed,
                                    "family": args.family,
                                    "load_family": load,
                                    "kappa": kappa,
                                    "n_boundary": n_boundary,
                                    "sample_id": start + offset,
                                    "relative_residual": float(value),
                                    "condition_number": float(condition[offset]),
                                    "exact_condition_number": float(
                                        exact_condition[offset]
                                    ),
                                    "condition_ratio": float(
                                        condition[offset] / exact_condition[offset]
                                    ),
                                })
                print(
                    f"[{args.study}/{args.family}] {load} k={kappa:g} "
                    f"nb={n_boundary}",
                    flush=True,
                )
    if not rows or not all(row["finite"] for row in rows):
        raise RuntimeError("empty or nonfinite bulk evaluation")
    write_csv(rows, output)
    write_csv(stability, stability_output)


if __name__ == "__main__":
    main()
