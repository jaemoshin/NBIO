#!/usr/bin/env python
"""Equal-update training for the locked Screening-Law NBIO confirmation."""
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

METHODS = ("full_prior", "no_prior", "screening_mix8", "radial_mlp")
MODEL_SEEDS = (51001, 51002, 51003)
BATCH_SEEDS = (52001, 52002, 52003)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config(method: str) -> dict:
    if method in {"full_prior", "no_prior"}:
        return {
            "model": {
                "name": "nbio_kappa",
                "kwargs": {
                    "hidden": 64,
                    "n_layers": 3,
                    "feat_scale": 1.0,
                    "region": "interior",
                    "solve": "direct",
                    "kappa_scale": 4.0,
                    "prior_mode": "full" if method == "full_prior" else "none",
                    "local_tau": 0.32,
                    "local_power": 4.0,
                    "residual_order": 2.0,
                    "local_scope": "both",
                    "residual_scale": "normalized",
                    "feature_mode": "smooth",
                },
            }
        }
    return {
        "model": {
            "name": "nbio_screening",
            "kwargs": {
                "variant": "mixture" if method == "screening_mix8" else "radial_mlp",
                "n_atoms": 8,
                "hidden": 32,
                "n_layers": 3,
                "region": "interior",
                "solve": "direct",
            },
        }
    }


def verified_data_path(manifest: dict, suffix: str) -> Path:
    matches = [entry for entry in manifest["files"] if entry["path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest entry ending in {suffix!r}")
    path = ROOT / matches[0]["path"]
    if sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"dataset hash mismatch: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-updates", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=375)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out = ROOT / args.out_dir
    expected_out = (
        ROOT / "results" / "screening_law" / "confirmatory_v1" / "training"
        / f"{args.method}_s{args.data_seed_index}"
    )
    if out.resolve() != expected_out.resolve():
        raise ValueError(f"confirmatory output must be {expected_out}")
    if (args.max_updates, args.batch_size, args.validation_every, args.lr) != (
        7500, 32, 375, 1e-3,
    ):
        raise ValueError("confirmatory optimizer contract is immutable")
    if out.exists():
        raise FileExistsError(f"write-once training directory exists: {out}")
    out.mkdir(parents=True)
    protocol_path = ROOT / "protocols" / "screening_law_confirmatory_v1.json"
    data_manifest_path = ROOT / "protocols" / "screening_law_confirmatory_data_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(data_manifest_path.read_text())
    if protocol["status"] != "LOCKED_BEFORE_TRAINING":
        raise RuntimeError("protocol is not locked")

    train_path = verified_data_path(
        manifest, f"data_s{args.data_seed_index}/yukawa_train.pt"
    )
    validation_path = verified_data_path(
        manifest, f"data_s{args.data_seed_index}/yukawa_val.pt"
    )
    train_set = BIEDataset(str(train_path))
    validation_set = BIEDataset(str(validation_path))
    if len(train_set) != 2400 or len(validation_set) != 400:
        raise RuntimeError("frozen dataset size mismatch")

    model_seed = MODEL_SEEDS[args.data_seed_index]
    batch_seed = BATCH_SEEDS[args.data_seed_index]
    set_seed(model_seed)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    config = model_config(args.method)
    config.update({
        "seed": model_seed,
        "method": args.method,
        "data_seed_index": args.data_seed_index,
        "train": {
            "max_updates": args.max_updates,
            "batch_size": args.batch_size,
            "validation_every": args.validation_every,
            "lr": args.lr,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "batch_seed": batch_seed,
        },
    })
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    updates_per_epoch = len(train_set) // args.batch_size
    if updates_per_epoch * args.batch_size != len(train_set):
        raise RuntimeError("training size must divide batch size for equal exposure")

    curve_path = out / "training_curve.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("update", "examples_seen", "train_rel_l2", "val_rel_l2", "lr"))

    best = float("inf")
    history = []
    running_loss = 0.0
    running_count = 0
    permutation = None
    for update in range(1, args.max_updates + 1):
        batch_in_epoch = (update - 1) % updates_per_epoch
        if batch_in_epoch == 0:
            epoch = (update - 1) // updates_per_epoch
            generator = torch.Generator().manual_seed(batch_seed + epoch)
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
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        running_loss += float(loss) * len(indices)
        running_count += len(indices)

        if update % args.validation_every == 0 or update == args.max_updates:
            validation_error = evaluate_rel_l2(model, validation_loader, device)
            training_error = running_loss / running_count
            row = {
                "update": update,
                "examples_seen": update * args.batch_size,
                "train_rel_l2": training_error,
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
                }, out / "best.pt")
            print(
                f"[{args.method} s{args.data_seed_index}] update={update}/{args.max_updates} "
                f"train={training_error:.4e} val={validation_error:.4e}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    run_manifest = {
        "method": args.method,
        "data_seed_index": args.data_seed_index,
        "model_seed": model_seed,
        "batch_seed": batch_seed,
        "n_parameters": count_params(model),
        "max_updates": args.max_updates,
        "batch_size": args.batch_size,
        "examples_seen": args.max_updates * args.batch_size,
        "validation_every": args.validation_every,
        "best_validation_rel_l2": best,
        "train_data_sha256": sha256(train_path),
        "validation_data_sha256": sha256(validation_path),
        "checkpoint_sha256": sha256(out / "best.pt"),
        "history": history,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"completed {args.method} seed index {args.data_seed_index}; best val={best:.4e}")


if __name__ == "__main__":
    main()
