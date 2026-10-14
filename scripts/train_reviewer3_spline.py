#!/usr/bin/env python
"""Train the preregistered capacity-matched cubic B-spline control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.trainer import evaluate_rel_l2, rel_l2  # noqa: E402
from src.utils.common import configure_torch, count_params, get_device, set_seed  # noqa: E402

PROTOCOL_PATH = ROOT / "protocols" / "reviewer3_evidence_v1.json"
DATA_MANIFEST_PATH = ROOT / "protocols" / "radial_factorization_ablation_data_manifest.json"
RESULT_ROOT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1"
MODEL_SEEDS = (171001, 171002, 171003)
BATCH_SEEDS = (111101, 111102, 111103)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(manifest: dict, suffix: str) -> Path:
    matches = [row for row in manifest["files"] if row["path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one data record ending in {suffix!r}")
    path = ROOT / matches[0]["path"]
    if sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"data hash mismatch: {path}")
    return path


def model_config() -> dict:
    return {"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_spline",
        "n_spline_coefficients": 2241,
        "spline_s_max": 16.0,
        "region": "interior",
        "solve": "direct",
        "kappa_scale": 4.0,
    }}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args()
    max_updates = args.smoke_updates or 7500
    if args.smoke_updates and not 1 <= args.smoke_updates <= 5:
        raise ValueError("smoke updates must be between one and five")

    suffix = f"spline_s{args.seed_index}"
    if args.smoke_updates:
        suffix += "_smoke"
    output = RESULT_ROOT / "training" / suffix
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    protocol = json.loads(PROTOCOL_PATH.read_text())
    data_manifest = json.loads(DATA_MANIFEST_PATH.read_text())
    if protocol["status"] != "LOCKED_BEFORE_EXECUTION":
        raise RuntimeError("protocol is not locked")
    expected_hash = protocol["frozen_inputs"]["radial_data_manifest_sha256"]
    if sha256(DATA_MANIFEST_PATH) != expected_hash:
        raise RuntimeError("radial data manifest differs from preregistration")
    train_path = verified_path(data_manifest, f"data_s{args.seed_index}/yukawa_train.pt")
    validation_path = verified_path(data_manifest, f"data_s{args.seed_index}/yukawa_val.pt")
    train_set = BIEDataset(str(train_path))
    validation_set = BIEDataset(str(validation_path))
    if len(train_set) != 2400 or len(validation_set) != 400:
        raise RuntimeError("frozen dataset size mismatch")

    set_seed(MODEL_SEEDS[args.seed_index])
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    config = model_config()
    config.update({
        "method": "radial_spline",
        "seed": MODEL_SEEDS[args.seed_index],
        "seed_index": args.seed_index,
        "protocol": "reviewer3_evidence_v1",
    })
    model = build_model(config).to(device)
    if count_params(model) != 2241:
        raise RuntimeError(f"parameter contract failed: {count_params(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=0)
    updates_per_epoch = len(train_set) // 32
    curve_path = output / "training_curve.csv"
    with curve_path.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            ("update", "examples_seen", "train_rel_l2", "val_rel_l2", "lr")
        )
    best = float("inf")
    history = []
    running_loss = 0.0
    running_count = 0
    permutation = None
    validation_every = max_updates if args.smoke_updates else 375
    for update in range(1, max_updates + 1):
        batch_in_epoch = (update - 1) % updates_per_epoch
        if batch_in_epoch == 0:
            epoch = (update - 1) // updates_per_epoch
            generator = torch.Generator().manual_seed(BATCH_SEEDS[args.seed_index] + epoch)
            permutation = torch.randperm(len(train_set), generator=generator).tolist()
        start = batch_in_epoch * 32
        indices = permutation[start:start + 32]
        batch = to_device(default_collate([train_set[index] for index in indices]), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rel_l2(model(batch), batch["uq"])
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("nonfinite training gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        factor = 0.5 * (1.0 + math.cos(math.pi * update / max_updates))
        optimizer.param_groups[0]["lr"] = 1e-3 * factor
        running_loss += float(loss) * len(indices)
        running_count += len(indices)
        if update % validation_every == 0 or update == max_updates:
            validation_error = evaluate_rel_l2(model, validation_loader, device)
            row = {
                "update": update,
                "examples_seen": update * 32,
                "train_rel_l2": running_loss / running_count,
                "val_rel_l2": validation_error,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            with curve_path.open("a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(row.values())
            if validation_error < best:
                best = validation_error
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": config,
                    "update": update,
                    "validation_rel_l2": validation_error,
                    "protocol_sha256": sha256(PROTOCOL_PATH),
                    "data_manifest_sha256": sha256(DATA_MANIFEST_PATH),
                }, output / "best.pt")
            print(
                f"[spline s{args.seed_index}] {update}/{max_updates} "
                f"train={row['train_rel_l2']:.4e} val={validation_error:.4e}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    manifest = {
        "method": "radial_spline",
        "seed_index": args.seed_index,
        "model_seed": MODEL_SEEDS[args.seed_index],
        "batch_seed": BATCH_SEEDS[args.seed_index],
        "n_parameters": count_params(model),
        "max_updates": max_updates,
        "batch_size": 32,
        "examples_seen": max_updates * 32,
        "best_validation_rel_l2": best,
        "train_data_sha256": sha256(train_path),
        "validation_data_sha256": sha256(validation_path),
        "checkpoint_sha256": sha256(output / "best.pt"),
        "history": history,
        "smoke": bool(args.smoke_updates),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
