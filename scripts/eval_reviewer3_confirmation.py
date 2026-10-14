#!/usr/bin/env python
"""Blind evaluation on the untouched reviewer-3 confirmation bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get("NBIO_SCREENING_SOURCE", ROOT))
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402

RESULT_ROOT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1"
DATA_ROOT = ROOT / "data_cache" / "reviewer3_confirmation_v1"
RADIAL_MANIFEST = ROOT / "results" / "screening_law" / "radial_paperization_v1" / "checkpoint_manifest.json"
EFFICIENCY_MANIFEST = ROOT / "results" / "screening_law" / "radial_data_efficiency_v1" / "checkpoint_manifest.json"
KAPPAS = (0.85, 1.45, 2.15, 2.85, 3.55)
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


def verified_records(manifest_path: Path, *, methods: set[str] | None = None):
    manifest = json.loads(manifest_path.read_text())
    output = []
    for record in manifest["records"]:
        if methods is not None and record["method"] not in methods:
            continue
        path = ROOT / record["path"]
        if not path.exists():
            path = FROZEN_ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        output.append((record, path))
    return output


def load_models(records, device: torch.device, *, with_condition: bool):
    output = []
    for record, path in records:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        output.append({
            "method": record["method"],
            "condition": (
                record.get("condition", {}).get("name", "main")
                if with_condition else "main"
            ),
            "seed_index": int(record["seed_index"]),
            "model": model.to(device).eval(),
        })
    return output


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
    parser.add_argument("--family", choices=("in_dist", "highmode", "rough", "deform"), required=True)
    args = parser.parse_args()
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)

    confirmation_manifest_path = RESULT_ROOT / "confirmation_data_manifest.json"
    confirmation_manifest = json.loads(confirmation_manifest_path.read_text())
    if confirmation_manifest["status"] != "UNTOUCHED_CONFIRMATION_DATA_SEALED":
        raise RuntimeError("confirmation data are not sealed")
    data_hashes = {
        row["path"]: row["sha256"] for row in confirmation_manifest["files"]
    }
    spline_manifest = RESULT_ROOT / "spline_checkpoint_manifest.json"
    main_records = []
    main_records.extend(verified_records(spline_manifest))
    main_records.extend(verified_records(
        RADIAL_MANIFEST, methods={"radial_normal", "generic_matched"},
    ))
    main_models = load_models(main_records, device, with_condition=False)
    efficiency_models = load_models(
        verified_records(EFFICIENCY_MANIFEST), device, with_condition=True,
    )

    main_rows = []
    efficiency_rows = []
    for kappa in KAPPAS:
        for load in LOADS:
            for n_boundary in RESOLUTIONS:
                path = DATA_ROOT / f"{load}_{args.family}_k{tag(kappa)}_nb{n_boundary}.pt"
                relative = str(path.relative_to(ROOT))
                if relative not in data_hashes or sha256(path) != data_hashes[relative]:
                    raise RuntimeError(f"unsealed or changed confirmation shard: {path}")
                shard = torch.load(path, map_location="cpu", weights_only=False)
                target = shard["uq"]
                exact_errors = (
                    torch.linalg.vector_norm(shard["bem_uq"] - target, dim=-1)
                    / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
                ).numpy()
                for sample_id, error in enumerate(exact_errors):
                    main_rows.append({
                        "method": "exact_screened_bie", "condition": "exact",
                        "seed_index": -1, "family": args.family,
                        "load_family": load, "kappa": kappa,
                        "n_boundary": n_boundary, "sample_id": sample_id,
                        "rel_l2": float(error), "finite": bool(np.isfinite(error)),
                    })
                batch = {
                    key: value.to(device) for key, value in shard.items()
                    if key != "bem_uq"
                }
                for record in main_models:
                    try:
                        prediction = record["model"](batch)
                        errors = (
                            torch.linalg.vector_norm(prediction - batch["uq"], dim=-1)
                            / torch.linalg.vector_norm(batch["uq"], dim=-1).clamp_min(1e-12)
                        ).cpu().numpy()
                    except (RuntimeError, torch.linalg.LinAlgError):
                        errors = np.full(len(target), np.nan)
                    for sample_id, error in enumerate(errors):
                        main_rows.append({
                            "method": record["method"], "condition": "main",
                            "seed_index": record["seed_index"], "family": args.family,
                            "load_family": load, "kappa": kappa,
                            "n_boundary": n_boundary, "sample_id": sample_id,
                            "rel_l2": float(error), "finite": bool(np.isfinite(error)),
                        })
                if n_boundary == 128:
                    for record in efficiency_models:
                        try:
                            prediction = record["model"](batch)
                            errors = (
                                torch.linalg.vector_norm(prediction - batch["uq"], dim=-1)
                                / torch.linalg.vector_norm(batch["uq"], dim=-1).clamp_min(1e-12)
                            ).cpu().numpy()
                        except (RuntimeError, torch.linalg.LinAlgError):
                            errors = np.full(len(target), np.nan)
                        for sample_id, error in enumerate(errors):
                            efficiency_rows.append({
                                "method": record["method"],
                                "condition": record["condition"],
                                "seed_index": record["seed_index"],
                                "family": args.family, "load_family": load,
                                "kappa": kappa, "n_boundary": n_boundary,
                                "sample_id": sample_id, "rel_l2": float(error),
                                "finite": bool(np.isfinite(error)),
                            })
                print(
                    f"[{args.family}] kappa={kappa:g} load={load} nb={n_boundary}",
                    flush=True,
                )
    expected_main = 5 * 2 * 3 * 32 * 10
    expected_efficiency = 5 * 2 * 32 * 42
    if len(main_rows) != expected_main or len(efficiency_rows) != expected_efficiency:
        raise RuntimeError(
            f"coverage mismatch main={len(main_rows)}/{expected_main} "
            f"efficiency={len(efficiency_rows)}/{expected_efficiency}"
        )
    write_csv(main_rows, RESULT_ROOT / "raw" / f"confirmation_main_{args.family}.csv")
    write_csv(
        efficiency_rows,
        RESULT_ROOT / "raw" / f"confirmation_efficiency_{args.family}.csv",
    )


if __name__ == "__main__":
    main()
