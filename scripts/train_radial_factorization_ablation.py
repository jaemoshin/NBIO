#!/usr/bin/env python
"""Equal-update training for the fresh parameter-matched factorization ablation."""
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

METHODS = ("generic_matched", "radial_separate", "radial_normal")
MODEL_SEEDS = (111001, 111002, 111003)
BATCH_SEEDS = (111101, 111102, 111103)
RESULT_ROOT = ROOT / "results" / "screening_law" / "radial_paperization_v1"
PROTOCOL_PATH = ROOT / "protocols" / "radial_factorization_ablation_v1.json"
TRAIN_DATA_MANIFEST_PATH = (
    ROOT / "protocols" / "radial_factorization_ablation_data_manifest.json"
)
EXPECTED_PARAMETERS = {
    "generic_matched": 2221,
    "radial_separate": 2241,
    "radial_normal": 2241,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config(method: str) -> dict:
    if method == "generic_matched":
        return {"model": {"name": "nbio_kappa", "kwargs": {
            "hidden": 30,
            "n_layers": 3,
            "feat_scale": 1.0,
            "region": "interior",
            "solve": "direct",
            "kappa_scale": 4.0,
            "prior_mode": "full",
            "residual_order": 2.0,
            "residual_scale": "normalized",
            "feature_mode": "smooth",
        }}}
    return {"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_separate" if method == "radial_separate" else "radial_mlp",
        "hidden": 32,
        "n_layers": 3,
        "region": "interior",
        "solve": "direct",
        "kappa_scale": 4.0,
    }}}


def verified_path(manifest: dict, suffix: str) -> Path:
    matches = [record for record in manifest["files"] if record["path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one data record ending with {suffix!r}")
    path = ROOT / matches[0]["path"]
    if sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"data hash mismatch: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--max-updates", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=375)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if (args.max_updates, args.batch_size, args.validation_every, args.lr) != (
        7500, 32, 375, 1e-3,
    ):
        raise ValueError("optimizer contract is immutable")

    output = RESULT_ROOT / "training" / f"{args.method}_s{args.seed_index}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol_path = PROTOCOL_PATH
    data_manifest_path = TRAIN_DATA_MANIFEST_PATH
    protocol = json.loads(protocol_path.read_text())
    data_manifest = json.loads(data_manifest_path.read_text())
    if protocol["status"] != "LOCKED_BEFORE_EXECUTION":
        raise RuntimeError("protocol is not locked")
    if data_manifest["status"] != "SEALED_BEFORE_TRAINING":
        raise RuntimeError("data are not sealed")

    train_path = verified_path(
        data_manifest, f"data_s{args.seed_index}/yukawa_train.pt",
    )
    validation_path = verified_path(
        data_manifest, f"data_s{args.seed_index}/yukawa_val.pt",
    )
    train_set = BIEDataset(str(train_path))
    validation_set = BIEDataset(str(validation_path))
    if len(train_set) != 2400 or len(validation_set) != 400:
        raise RuntimeError("frozen dataset size mismatch")

    set_seed(MODEL_SEEDS[args.seed_index])
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    config = model_config(args.method)
    config.update({
        "seed": MODEL_SEEDS[args.seed_index],
        "method": args.method,
        "seed_index": args.seed_index,
        "train": {
            "max_updates": args.max_updates,
            "batch_size": args.batch_size,
            "validation_every": args.validation_every,
            "lr": args.lr,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "batch_seed": BATCH_SEEDS[args.seed_index],
        },
    })
    model = build_model(config).to(device)
    expected_parameters = EXPECTED_PARAMETERS[args.method]
    if count_params(model) != expected_parameters:
        raise RuntimeError(
            f"parameter mismatch for {args.method}: {count_params(model)} != {expected_parameters}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    updates_per_epoch = len(train_set) // args.batch_size
    if updates_per_epoch * args.batch_size != len(train_set):
        raise RuntimeError("training size must divide batch size")

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
    for update in range(1, args.max_updates + 1):
        batch_in_epoch = (update - 1) % updates_per_epoch
        if batch_in_epoch == 0:
            epoch = (update - 1) // updates_per_epoch
            generator = torch.Generator().manual_seed(BATCH_SEEDS[args.seed_index] + epoch)
            permutation = torch.randperm(len(train_set), generator=generator).tolist()
        start = batch_in_epoch * args.batch_size
        indices = permutation[start:start + args.batch_size]
        batch = to_device(default_collate([train_set[index] for index in indices]), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rel_l2(model(batch), batch["uq"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        factor = 0.5 * (1.0 + math.cos(math.pi * update / args.max_updates))
        optimizer.param_groups[0]["lr"] = args.lr * factor
        running_loss += float(loss) * len(indices)
        running_count += len(indices)

        if update % args.validation_every == 0 or update == args.max_updates:
            validation_error = evaluate_rel_l2(model, validation_loader, device)
            row = {
                "update": update,
                "examples_seen": update * args.batch_size,
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
                    "protocol_sha256": sha256(protocol_path),
                    "data_manifest_sha256": sha256(data_manifest_path),
                }, output / "best.pt")
            print(
                f"[{args.method} s{args.seed_index}] {update}/{args.max_updates} "
                f"train={row['train_rel_l2']:.4e} val={validation_error:.4e}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    manifest = {
        "method": args.method,
        "seed_index": args.seed_index,
        "model_seed": MODEL_SEEDS[args.seed_index],
        "batch_seed": BATCH_SEEDS[args.seed_index],
        "n_parameters": count_params(model),
        "max_updates": args.max_updates,
        "batch_size": args.batch_size,
        "examples_seen": args.max_updates * args.batch_size,
        "best_validation_rel_l2": best,
        "train_data_sha256": sha256(train_path),
        "validation_data_sha256": sha256(validation_path),
        "checkpoint_sha256": sha256(output / "best.pt"),
        "history": history,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
