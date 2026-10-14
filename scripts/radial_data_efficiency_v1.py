#!/usr/bin/env python
"""Prospective sample, sensor, and label-noise efficiency study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.trainer import evaluate_rel_l2, rel_l2  # noqa: E402
from src.utils.common import configure_torch, count_params, get_device, set_seed  # noqa: E402
from train_radial_factorization_ablation import model_config  # noqa: E402

PROTOCOL = ROOT / "protocols" / "radial_data_efficiency_v1.json"
DATA_MANIFEST = ROOT / "protocols" / "radial_factorization_ablation_data_manifest.json"
DATA_ROOT = ROOT / "data_cache" / "radial_factorization_ablation_v1"
RESULT = ROOT / "results" / "screening_law" / "radial_data_efficiency_v1"
REPORT = ROOT / "reports" / "radial_data_efficiency_v1.md"
METHODS = ("generic_matched", "radial_normal")
MODEL_SEEDS = (121001, 121002, 121003)
BATCH_SEEDS = (121101, 121102, 121103)
SUBSET_SEEDS = (121201, 121202, 121203)
NOISE_SEEDS = (121301, 121302, 121303)
FAMILIES = ("in_dist", "highmode", "rough", "deform")
KAPPAS = (1.2, 1.75, 2.4, 3.1, 3.8)
LOADS = ("point_source", "fourier")


def conditions() -> list[dict]:
    output = [
        {"name": f"sample_n{count}", "n_unique": count, "n_query": 256, "noise": 0.0}
        for count in (100, 300, 800, 2400)
    ]
    output.extend(
        {"name": f"sensor_q32_noise{str(noise).replace('.', 'p')}",
         "n_unique": 300, "n_query": 32, "noise": noise}
        for noise in (0.0, 0.01, 0.05)
    )
    return output


def condition_by_name(name: str) -> dict:
    matches = [condition for condition in conditions() if condition["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown condition {name}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(manifest: dict, suffix: str) -> Path:
    matches = [record for record in manifest["files"] if record["path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one data record ending in {suffix!r}")
    path = ROOT / matches[0]["path"]
    if sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"data hash mismatch: {path}")
    return path


class LimitedObservationDataset(Dataset):
    """A fixed subset with fixed query sensors and fixed relative label noise."""

    def __init__(self, base: BIEDataset, *, count: int, n_query: int,
                 noise: float, subset_seed: int, noise_seed: int):
        if count > len(base):
            raise ValueError("subset exceeds base data")
        self.base = base
        generator = torch.Generator().manual_seed(subset_seed)
        self.indices = torch.randperm(len(base), generator=generator)[:count].tolist()
        self.n_query = int(n_query)
        self.noise = float(noise)
        self.subset_seed = int(subset_seed)
        self.noise_seed = int(noise_seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        item = dict(self.base[base_index])
        total_queries = item["uq"].shape[0]
        if self.n_query < total_queries:
            generator = torch.Generator().manual_seed(
                self.subset_seed + 1000003 * (base_index + 1)
            )
            query_indices = torch.randperm(
                total_queries, generator=generator,
            )[:self.n_query]
            item["Xq"] = item["Xq"][query_indices]
            item["uq"] = item["uq"][query_indices]
        if self.noise:
            generator = torch.Generator().manual_seed(
                self.noise_seed + 1000033 * (base_index + 1)
            )
            noise = torch.randn(item["uq"].shape, generator=generator)
            scale = item["uq"].square().mean().sqrt().clamp_min(1e-12)
            item["uq"] = item["uq"] + self.noise * scale * noise
        return item


def train(method: str, condition_name: str, seed_index: int) -> None:
    condition = condition_by_name(condition_name)
    output = RESULT / "training" / f"{method}__{condition_name}__s{seed_index}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(DATA_MANIFEST.read_text())
    if protocol["status"] != "LOCKED_BEFORE_EXECUTION":
        raise RuntimeError("protocol is not locked")
    train_path = verified_path(
        manifest, f"data_s{seed_index}/yukawa_train.pt",
    )
    validation_path = verified_path(
        manifest, f"data_s{seed_index}/yukawa_val.pt",
    )
    train_set = LimitedObservationDataset(
        BIEDataset(str(train_path)),
        count=condition["n_unique"],
        n_query=condition["n_query"],
        noise=condition["noise"],
        subset_seed=SUBSET_SEEDS[seed_index],
        noise_seed=NOISE_SEEDS[seed_index],
    )
    validation_set = BIEDataset(str(validation_path))
    set_seed(MODEL_SEEDS[seed_index])
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    config = model_config(method)
    config.update({
        "method": method,
        "condition": condition,
        "seed_index": seed_index,
        "seed": MODEL_SEEDS[seed_index],
    })
    model = build_model(config).to(device)
    expected_parameters = 2221 if method == "generic_matched" else 2241
    if count_params(model) != expected_parameters:
        raise RuntimeError("parameter contract failed")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    validation_loader = DataLoader(
        validation_set, batch_size=32, shuffle=False, num_workers=0,
    )
    batch_generator = torch.Generator().manual_seed(BATCH_SEEDS[seed_index])
    best = float("inf")
    history = []
    curve = output / "training_curve.csv"
    with curve.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            ("update", "examples_seen", "unique_instances", "query_observations",
             "train_rel_l2", "val_rel_l2", "lr")
        )
    running = 0.0
    running_count = 0
    for update in range(1, 7501):
        indices = torch.randint(
            0, len(train_set), (32,), generator=batch_generator,
        ).tolist()
        batch = to_device(
            default_collate([train_set[index] for index in indices]), device,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rel_l2(model(batch), batch["uq"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        factor = 0.5 * (1.0 + math.cos(math.pi * update / 7500))
        optimizer.param_groups[0]["lr"] = 1e-3 * factor
        running += float(loss) * 32
        running_count += 32
        if update % 375 == 0:
            validation = evaluate_rel_l2(model, validation_loader, device)
            row = {
                "update": update,
                "examples_seen": update * 32,
                "unique_instances": condition["n_unique"],
                "query_observations": condition["n_unique"] * condition["n_query"],
                "train_rel_l2": running / running_count,
                "val_rel_l2": validation,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            with curve.open("a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(row.values())
            if validation < best:
                best = validation
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": config,
                    "update": update,
                    "validation_rel_l2": validation,
                }, output / "best.pt")
            print(
                f"[{method} {condition_name} s{seed_index}] "
                f"{update}/7500 val={validation:.4e}",
                flush=True,
            )
            running = 0.0
            running_count = 0
    run_manifest = {
        "method": method,
        "condition": condition,
        "seed_index": seed_index,
        "model_seed": MODEL_SEEDS[seed_index],
        "batch_seed": BATCH_SEEDS[seed_index],
        "subset_seed": SUBSET_SEEDS[seed_index],
        "noise_seed": NOISE_SEEDS[seed_index],
        "n_parameters": count_params(model),
        "updates": 7500,
        "examples_seen": 240000,
        "unique_instances": condition["n_unique"],
        "query_observations": condition["n_unique"] * condition["n_query"],
        "best_validation_rel_l2": best,
        "checkpoint_sha256": sha256(output / "best.pt"),
        "protocol_sha256": sha256(PROTOCOL),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "history": history,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )


def seal() -> None:
    output = RESULT / "checkpoint_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    records = []
    for condition in conditions():
        for method in METHODS:
            for seed_index in range(3):
                run = (
                    RESULT / "training"
                    / f"{method}__{condition['name']}__s{seed_index}"
                )
                checkpoint = run / "best.pt"
                manifest = json.loads((run / "run_manifest.json").read_text())
                if manifest["updates"] != 7500 or manifest["examples_seen"] != 240000:
                    raise RuntimeError(f"training contract failed: {run}")
                if manifest["checkpoint_sha256"] != sha256(checkpoint):
                    raise RuntimeError(f"checkpoint hash mismatch: {run}")
                records.append({
                    "method": method,
                    "condition": condition,
                    "seed_index": seed_index,
                    "path": str(checkpoint.relative_to(ROOT)),
                    "sha256": sha256(checkpoint),
                    "n_parameters": manifest["n_parameters"],
                    "best_validation_rel_l2": manifest["best_validation_rel_l2"],
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT",
        "records": records,
    }, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(records)} checkpoints")


def load_models(device: torch.device):
    manifest = json.loads((RESULT / "checkpoint_manifest.json").read_text())
    if manifest["status"] != "CHECKPOINTS_FROZEN_BEFORE_HOLDOUT":
        raise RuntimeError("checkpoints are not sealed")
    output = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        output.append((
            record["method"], record["condition"]["name"],
            record["seed_index"], model.to(device).eval(),
        ))
    return output


def kappa_tag(value: float) -> str:
    return str(value).replace(".", "p")


@torch.no_grad()
def evaluate(family: str) -> None:
    output = RESULT / "raw" / f"{family}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    models = load_models(device)
    manifest = json.loads(DATA_MANIFEST.read_text())
    records = {record["path"]: record["sha256"] for record in manifest["files"]}
    rows = []
    for kappa in KAPPAS:
        for load in LOADS:
            path = (
                DATA_ROOT / "holdout"
                / f"{load}_{family}_k{kappa_tag(kappa)}_nb128.pt"
            )
            relative = str(path.relative_to(ROOT))
            if relative not in records or sha256(path) != records[relative]:
                raise RuntimeError(f"unsealed holdout shard: {path}")
            shard = torch.load(path, map_location="cpu", weights_only=False)
            target = shard["uq"]
            exact = (
                torch.linalg.vector_norm(shard["bem_uq"] - target, dim=-1)
                / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
            ).numpy()
            for sample_id, error in enumerate(exact):
                rows.append({
                    "method": "exact_yukawa_bie",
                    "condition": "exact",
                    "seed_index": -1,
                    "family": family,
                    "load_family": load,
                    "kappa": kappa,
                    "sample_id": sample_id,
                    "rel_l2": float(error),
                    "finite": bool(np.isfinite(error)),
                })
            batch = {
                key: value.to(device)
                for key, value in shard.items() if key != "bem_uq"
            }
            for method, condition, seed_index, model in models:
                try:
                    prediction = model(batch)
                    errors = (
                        torch.linalg.vector_norm(
                            prediction - batch["uq"], dim=-1,
                        )
                        / torch.linalg.vector_norm(
                            batch["uq"], dim=-1,
                        ).clamp_min(1e-12)
                    ).cpu().numpy()
                except (RuntimeError, torch.linalg.LinAlgError):
                    errors = np.full(len(target), np.nan)
                for sample_id, error in enumerate(errors):
                    rows.append({
                        "method": method,
                        "condition": condition,
                        "seed_index": seed_index,
                        "family": family,
                        "load_family": load,
                        "kappa": kappa,
                        "sample_id": sample_id,
                        "rel_l2": float(error),
                        "finite": bool(np.isfinite(error)),
                    })
            print(f"[{family}] kappa={kappa:g} load={load}", flush=True)
    expected = len(KAPPAS) * len(LOADS) * 32 * (
        len(conditions()) * len(METHODS) * 3 + 1
    )
    if len(rows) != expected:
        raise RuntimeError(f"coverage failure: {len(rows)} != {expected}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate() -> None:
    outputs = (RESULT / "summary.csv", RESULT / "verdict.json", REPORT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("write-once aggregate output exists")
    rows = []
    for family in FAMILIES:
        with (RESULT / "raw" / f"{family}.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        row["seed_index"] = int(row["seed_index"])
        row["kappa"] = float(row["kappa"])
        row["sample_id"] = int(row["sample_id"])
        row["rel_l2"] = float(row["rel_l2"])
        row["finite"] = row["finite"].lower() == "true"
    expected = len(FAMILIES) * len(KAPPAS) * len(LOADS) * 32 * (
        len(conditions()) * len(METHODS) * 3 + 1
    )
    all_finite = (
        len(rows) == expected
        and all(row["finite"] and np.isfinite(row["rel_l2"]) for row in rows)
    )

    def values(method: str, condition: str, seed_index=None):
        return [
            row["rel_l2"] for row in rows
            if row["method"] == method
            and row["condition"] == condition
            and (seed_index is None or row["seed_index"] == seed_index)
        ]

    means = {
        condition["name"]: {
            method: float(np.mean(values(method, condition["name"])))
            for method in METHODS
        }
        for condition in conditions()
    }
    seed_means = {
        condition["name"]: {
            method: {
                str(seed): float(np.mean(values(method, condition["name"], seed)))
                for seed in range(3)
            }
            for method in METHODS
        }
        for condition in conditions()
    }
    exact = float(np.mean([
        row["rel_l2"] for row in rows if row["method"] == "exact_yukawa_bie"
    ]))
    gate_values = {
        "radial_at_100_over_generic": (
            means["sample_n100"]["radial_normal"]
            / means["sample_n100"]["generic_matched"]
        ),
        "radial_at_300_over_2400": (
            means["sample_n300"]["radial_normal"]
            / means["sample_n2400"]["radial_normal"]
        ),
        "radial_q32_noise5pct_over_generic": (
            means["sensor_q32_noise0p05"]["radial_normal"]
            / means["sensor_q32_noise0p05"]["generic_matched"]
        ),
    }
    gates = {
        "E1_small_sample_advantage": gate_values[
            "radial_at_100_over_generic"
        ] <= 0.5,
        "E2_300_sample_efficiency": gate_values[
            "radial_at_300_over_2400"
        ] <= 2.0,
        "E3_sensor_noise_advantage": gate_values[
            "radial_q32_noise5pct_over_generic"
        ] <= 0.5,
        "E4_all_finite": all_finite,
    }
    verdict = "PAPERIZE" if all(gates.values()) else "NARROW"

    summary_rows = []
    for condition in conditions():
        for method in METHODS:
            chosen = values(method, condition["name"])
            summary_rows.append({
                "method": method,
                **condition,
                "n_rows": len(chosen),
                "mean_rel_l2": float(np.mean(chosen)),
                "median_rel_l2": float(np.median(chosen)),
                "p90_rel_l2": float(np.quantile(chosen, 0.9)),
                "seed0_mean": seed_means[condition["name"]][method]["0"],
                "seed1_mean": seed_means[condition["name"]][method]["1"],
                "seed2_mean": seed_means[condition["name"]][method]["2"],
            })
    with (RESULT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary_rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "verdict": verdict,
        "gates": gates,
        "gate_values": gate_values,
        "all_finite": all_finite,
        "coverage_rows": len(rows),
        "exact_yukawa_bie": exact,
        "means": means,
        "seed_means": seed_means,
    }
    (RESULT / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    sample_lines = "\n".join(
        f"| {count} | `{means[f'sample_n{count}']['radial_normal']:.4e}` "
        f"| `{means[f'sample_n{count}']['generic_matched']:.4e}` |"
        for count in (100, 300, 800, 2400)
    )
    noise_lines = []
    for noise in (0.0, 0.01, 0.05):
        name = f"sensor_q32_noise{str(noise).replace('.', 'p')}"
        noise_lines.append(
            f"| {noise:g} | `{means[name]['radial_normal']:.4e}` "
            f"| `{means[name]['generic_matched']:.4e}` |"
        )
    noise_lines = "\n".join(noise_lines)
    REPORT.write_text(f"""# Radial normal-form data efficiency

Verdict: **{verdict}**

All models receive 7,500 updates and are evaluated on the same clean holdout.

| unique training instances | radial normal form | generic matched |
|---:|---:|---:|
{sample_lines}

With 300 unique instances and 32 query observations per instance:

| relative label noise | radial normal form | generic matched |
|---:|---:|---:|
{noise_lines}

The exact modified-Helmholtz BIE ceiling is `{exact:.4e}`. Gate values are:
`{json.dumps(gate_values, sort_keys=True)}`.

The experiment measures solution-supervision efficiency at a fixed optimizer
budget. It does not establish efficiency under every noise model or acquisition
process.
""")
    print(json.dumps(payload, indent=2))


def smoke() -> None:
    manifest = json.loads(DATA_MANIFEST.read_text())
    train_path = verified_path(manifest, "data_s0/yukawa_train.pt")
    base = BIEDataset(str(train_path))
    device = get_device(True)
    for condition in conditions():
        dataset = LimitedObservationDataset(
            base,
            count=condition["n_unique"],
            n_query=condition["n_query"],
            noise=condition["noise"],
            subset_seed=SUBSET_SEEDS[0],
            noise_seed=NOISE_SEEDS[0],
        )
        batch = to_device(default_collate([dataset[0], dataset[1]]), device)
        for method in METHODS:
            model = build_model(model_config(method)).to(device)
            prediction = model(batch)
            loss = rel_l2(prediction, batch["uq"])
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite smoke loss: {condition} {method}")
    print(f"CUDA smoke passed on {device}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--method", choices=METHODS, required=True)
    train_parser.add_argument(
        "--condition", choices=[condition["name"] for condition in conditions()],
        required=True,
    )
    train_parser.add_argument(
        "--seed-index", type=int, choices=(0, 1, 2), required=True,
    )
    sub.add_parser("seal")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--family", choices=FAMILIES, required=True)
    sub.add_parser("aggregate")
    sub.add_parser("smoke")
    args = parser.parse_args()
    if args.command == "train":
        train(args.method, args.condition, args.seed_index)
    elif args.command == "seal":
        seal()
    elif args.command == "evaluate":
        evaluate(args.family)
    elif args.command == "aggregate":
        aggregate()
    else:
        smoke()


if __name__ == "__main__":
    main()
