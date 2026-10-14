#!/usr/bin/env python
"""Prospective sparse-kappa interpolation test for radial screening-law NBIO."""
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
from torch.utils.data import DataLoader, default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_kappa_conditioned import make_row, stack  # noqa: E402
from screening_query_resolution_phase import fourier_solution  # noqa: E402
from src.data.bie import boundary, bem_solve, eval_u, radius_at, sample_charges, sample_coeffs, sample_queries  # noqa: E402
from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.trainer import evaluate_rel_l2, rel_l2  # noqa: E402
from src.utils.common import configure_torch, count_params, get_device, set_seed  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402
from train_screening_confirmatory import model_config  # noqa: E402

PROTOCOL = ROOT / "protocols" / "screening_law_sparse_k_v1.json"
DATA = ROOT / "data_cache" / "screening_sparse_k_v1"
RESULT = ROOT / "results" / "screening_law" / "sparse_k_v1"
REPORT = ROOT / "reports" / "screening_law_sparse_k_v1.md"
TRAIN_KAPPAS = (0.0, 1.0, 2.0, 4.0)
TEST_KAPPAS = (0.25, 0.6, 1.4, 1.7, 2.5, 3.0, 3.6)
RESOLUTIONS = (64, 128, 256)
METHODS = ("full_prior", "no_prior", "radial_mlp")
MODEL_SEEDS = (71001, 71002, 71003)
BATCH_SEEDS = (72001, 72002, 72003)
FAMILY_NAMES = ("in_dist", "highmode", "rough", "deform")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_split(n_samples: int, seed: int) -> dict[str, torch.Tensor]:
    if n_samples % len(TRAIN_KAPPAS):
        raise ValueError("sample count must divide across sparse kappa anchors")
    rng = np.random.default_rng(seed)
    kappas = np.repeat(TRAIN_KAPPAS, n_samples // len(TRAIN_KAPPAS))
    rng.shuffle(kappas)
    rows = []
    for kappa in kappas:
        coefficients = sample_coeffs(rng)
        charges = sample_charges(rng, "interior", coefficients, n_charges=6)
        query = sample_queries(rng, coefficients, "interior", 256, margin=0.12)
        rows.append(make_row(coefficients, charges, query, float(kappa), 128, False,
                             query_margin=0.12))
    return stack(rows)


def generate_data() -> None:
    if DATA.exists():
        raise FileExistsError(DATA)
    for seed_index in range(3):
        directory = DATA / f"data_s{seed_index}"
        directory.mkdir(parents=True)
        torch.save(sparse_split(2400, 99011 + seed_index), directory / "train.pt")
        torch.save(sparse_split(400, 99111 + seed_index), directory / "val.pt")
        print(f"generated sparse train/val seed {seed_index}", flush=True)

    confirmation = DATA / "confirmation"
    confirmation.mkdir(parents=True)
    fine = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    for family_index, family in enumerate(FAMILY_NAMES):
        rng = np.random.default_rng(99201 + family_index)
        geometries = []
        while len(geometries) < 24:
            coefficients = ood_coeffs(rng, FAMILIES[family])
            if radius_at(coefficients, fine).min() <= 0.2:
                continue
            geometries.append({
                "coefficients": coefficients,
                "charges": sample_charges(rng, "interior", coefficients, n_charges=6),
                "fourier": rng.normal(size=(6, 2)) / np.arange(1, 7)[:, None],
                "query": sample_queries(rng, coefficients, "interior", 256, margin=0.12),
            })
        for kappa in TEST_KAPPAS:
            for n_boundary in RESOLUTIONS:
                for load in ("point_source", "fourier"):
                    values = {key: [] for key in ("Xb", "Nb", "wb", "curv", "g", "Xq", "uq", "bem_uq")}
                    for geometry in geometries:
                        curve = boundary(geometry["coefficients"], n_boundary)
                        if load == "point_source":
                            trace = eval_u(curve["X"], geometry["charges"], kappa)
                            target = eval_u(geometry["query"], geometry["charges"], kappa)
                        else:
                            trace = fourier_solution(curve["X"], kappa, geometry["fourier"])
                            target = fourier_solution(geometry["query"], kappa, geometry["fourier"])
                        _, evaluator = bem_solve(curve, trace, "interior", kappa=kappa)
                        for key, value in (
                            ("Xb", curve["X"]), ("Nb", curve["normal"]), ("wb", curve["w"]),
                            ("curv", curve["curv"]), ("g", trace), ("Xq", geometry["query"]),
                            ("uq", target), ("bem_uq", evaluator(geometry["query"])),
                        ):
                            values[key].append(value)
                    shard = {key: torch.tensor(np.asarray(value), dtype=torch.float32)
                             for key, value in values.items()}
                    shard["kappa"] = torch.full((24,), float(kappa))
                    tag = str(kappa).replace(".", "p")
                    torch.save(shard, confirmation / f"{load}_{family}_k{tag}_nb{n_boundary}.pt")
            print(f"generated {family} kappa={kappa:g}", flush=True)

    records = []
    for path in sorted(DATA.rglob("*.pt")):
        records.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                        "bytes": path.stat().st_size})
    manifest = {"status": "DATA_FROZEN_BEFORE_TRAINING", "records": records}
    (DATA / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"sealed {len(records)} sparse-k data files", flush=True)


def verified_data(manifest: dict, suffix: str) -> Path:
    matches = [record for record in manifest["records"] if record["path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one data file ending in {suffix}")
    path = ROOT / matches[0]["path"]
    if sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"data hash mismatch: {path}")
    return path


def train(method: str, seed_index: int) -> None:
    out = RESULT / "training" / f"{method}_s{seed_index}"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest_path = DATA / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path = verified_data(manifest, f"data_s{seed_index}/train.pt")
    val_path = verified_data(manifest, f"data_s{seed_index}/val.pt")
    train_set, val_set = BIEDataset(str(train_path)), BIEDataset(str(val_path))
    set_seed(MODEL_SEEDS[seed_index]); configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    config = model_config(method)
    config.update({"method": method, "seed": MODEL_SEEDS[seed_index], "seed_index": seed_index})
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    updates_per_epoch = len(train_set) // 32
    best = float("inf"); history = []
    curve = out / "training_curve.csv"
    with curve.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            ("update", "examples_seen", "train_rel_l2", "val_rel_l2", "lr")
        )
    running = 0.0; count = 0; permutation = None
    for update in range(1, 7501):
        batch_index = (update - 1) % updates_per_epoch
        if batch_index == 0:
            epoch = (update - 1) // updates_per_epoch
            generator = torch.Generator().manual_seed(BATCH_SEEDS[seed_index] + epoch)
            permutation = torch.randperm(len(train_set), generator=generator).tolist()
        indices = permutation[batch_index * 32:(batch_index + 1) * 32]
        batch = to_device(default_collate([train_set[index] for index in indices]), device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = rel_l2(model(batch), batch["uq"]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        factor = 0.5 * (1 + math.cos(math.pi * update / 7500))
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * factor
        running += float(loss) * 32; count += 32
        if update % 375 == 0:
            validation = evaluate_rel_l2(model, val_loader, device)
            record = {"update": update, "examples_seen": update * 32,
                      "train_rel_l2": running / count, "val_rel_l2": validation,
                      "lr": optimizer.param_groups[0]["lr"]}
            history.append(record)
            with curve.open("a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(record.values())
            if validation < best:
                best = validation
                torch.save({"state_dict": model.state_dict(), "config": config,
                            "update": update, "validation_rel_l2": validation}, out / "best.pt")
            print(f"[{method} s{seed_index}] {update}/7500 val={validation:.4e}", flush=True)
            running = 0.0; count = 0
    run_manifest = {
        "method": method, "seed_index": seed_index, "model_seed": MODEL_SEEDS[seed_index],
        "batch_seed": BATCH_SEEDS[seed_index], "n_parameters": count_params(model),
        "updates": 7500, "examples_seen": 240000, "best_validation_rel_l2": best,
        "checkpoint_sha256": sha256(out / "best.pt"), "data_manifest_sha256": sha256(manifest_path),
        "history": history,
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")


def seal() -> None:
    path = RESULT / "checkpoint_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    records = []
    for method in METHODS:
        for seed in range(3):
            run = RESULT / "training" / f"{method}_s{seed}"
            manifest = json.loads((run / "run_manifest.json").read_text())
            checkpoint = run / "best.pt"
            if manifest["updates"] != 7500 or manifest["checkpoint_sha256"] != sha256(checkpoint):
                raise RuntimeError(f"invalid run: {run}")
            records.append({"method": method, "seed_index": seed,
                            "path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint),
                            "n_parameters": manifest["n_parameters"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "CHECKPOINTS_FROZEN", "records": records},
                               indent=2, sort_keys=True) + "\n")


def load_models(device: torch.device):
    manifest = json.loads((RESULT / "checkpoint_manifest.json").read_text())
    output = []
    for record in manifest["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"]); model.load_state_dict(checkpoint["state_dict"])
        output.append((record["method"], record["seed_index"], model.to(device).eval()))
    return output


@torch.no_grad()
def evaluate_family(family: str) -> None:
    output = RESULT / "raw" / f"{family}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch(); torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True); models = load_models(device)
    data_manifest = json.loads((DATA / "data_manifest.json").read_text())
    rows = []
    for kappa in TEST_KAPPAS:
        for n_boundary in RESOLUTIONS:
            for load in ("point_source", "fourier"):
                tag = str(kappa).replace(".", "p")
                path = verified_data(data_manifest, f"{load}_{family}_k{tag}_nb{n_boundary}.pt")
                shard = torch.load(path, map_location="cpu", weights_only=False)
                target = shard["uq"]
                exact_errors = torch.linalg.vector_norm(shard["bem_uq"] - target, dim=-1) \
                    / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
                for sample_id, error in enumerate(exact_errors):
                    rows.append({"method": "correct_yukawa_bie", "seed_index": -1,
                                 "family": family, "kappa": kappa, "load_family": load,
                                 "n_boundary": n_boundary, "sample_id": sample_id,
                                 "rel_l2": float(error), "finite": bool(torch.isfinite(error))})
                batch = {key: value.to(device) for key, value in shard.items() if key != "bem_uq"}
                for method, seed, model in models:
                    try:
                        prediction = model(batch)
                        errors = torch.linalg.vector_norm(prediction - batch["uq"], dim=-1) \
                            / torch.linalg.vector_norm(batch["uq"], dim=-1).clamp_min(1e-12)
                        errors = errors.cpu().numpy()
                    except (RuntimeError, torch.linalg.LinAlgError):
                        errors = np.full(24, np.nan)
                    for sample_id, error in enumerate(errors):
                        rows.append({"method": method, "seed_index": seed, "family": family,
                                     "kappa": kappa, "load_family": load,
                                     "n_boundary": n_boundary, "sample_id": sample_id,
                                     "rel_l2": float(error), "finite": bool(np.isfinite(error))})
                print(f"[{family}] k={kappa:g} load={load} nb={n_boundary}", flush=True)
    expected = 7 * 3 * 2 * 24 * 10
    if len(rows) != expected:
        raise RuntimeError(f"coverage failure {len(rows)} != {expected}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def select(rows, method, *, seed=None, n_boundary=128):
    return [row for row in rows if row["method"] == method and int(row["n_boundary"]) == n_boundary
            and (seed is None or int(row["seed_index"]) == seed)]


def aggregate() -> None:
    rows = []
    for family in FAMILY_NAMES:
        with (RESULT / "raw" / f"{family}.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 40320 or any(row["finite"] != "True" for row in rows):
        raise RuntimeError("coverage or finite failure")
    means = {method: float(np.mean([float(row["rel_l2"]) for row in select(rows, method)]))
             for method in METHODS + ("correct_yukawa_bie",)}
    best = min(("full_prior", "no_prior"), key=means.get)
    ratio = means["radial_mlp"] / means[best]
    seed_ratios = []
    for seed in range(3):
        radial = np.mean([float(row["rel_l2"]) for row in select(rows, "radial_mlp", seed=seed)])
        full = np.mean([float(row["rel_l2"]) for row in select(rows, "full_prior", seed=seed)])
        none = np.mean([float(row["rel_l2"]) for row in select(rows, "no_prior", seed=seed)])
        seed_ratios.append(float(radial / min(full, none)))
    clusters = {}
    for method in ("radial_mlp", best):
        grouped = defaultdict(list)
        for row in select(rows, method):
            grouped[(row["family"], row["sample_id"])].append(float(row["rel_l2"]))
        clusters[method] = {key: float(np.mean(value)) for key, value in grouped.items()}
    keys = sorted(clusters["radial_mlp"])
    target = np.asarray([clusters["radial_mlp"][key] for key in keys])
    control = np.asarray([clusters[best][key] for key in keys])
    rng = np.random.default_rng(73001); indices = rng.integers(0, len(keys), (5000, len(keys)))
    ratios = target[indices].mean(1) / control[indices].mean(1)
    ci = [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]
    pass_gate = ratio <= 0.8 and max(seed_ratios) < 1 and ci[1] < 1

    per_kappa = {}
    for kappa in TEST_KAPPAS:
        per_kappa[str(kappa)] = {
            method: float(np.mean([float(row["rel_l2"]) for row in select(rows, method)
                                   if float(row["kappa"]) == kappa]))
            for method in METHODS + ("correct_yukawa_bie",)
        }
    result = {"verdict": "PAPERIZE" if pass_gate else "KILL", "pass": pass_gate,
              "means": means, "best_generic_control": best, "radial_control_ratio": ratio,
              "seed_ratios": seed_ratios, "cluster_bootstrap_ci95": ci,
              "per_kappa": per_kappa, "n_rows": len(rows)}
    (RESULT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(f"""# Sparse-kappa screening-law interpolation

Models were trained only at `kappa={{0,1,2,4}}` and tested at seven unseen interior values across four fresh
geometry families, point/Fourier loads, and resolutions 64/128/256.

At the primary `n_b=128` protocol:

| method | mean rel-L2 |
|---|---:|
| exact Yukawa BIE | `{means['correct_yukawa_bie']:.4e}` |
| generic full prior | `{means['full_prior']:.4e}` |
| generic no prior | `{means['no_prior']:.4e}` |
| radial MLP | `{means['radial_mlp']:.4e}` |

Radial/better-generic ratio: `{ratio:.4f}`; 95% geometry-cluster bootstrap interval
`{ci[0]:.4f}-{ci[1]:.4f}`; paired seed ratios `{seed_ratios}`.

Verdict: **{'PAPERIZE' if pass_gate else 'KILL'}**. This is a prospective secondary interpolation test and
does not replace the main blinded confirmation.
""")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--method", choices=METHODS, required=True)
    train_parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    sub.add_parser("seal")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--family", choices=FAMILY_NAMES, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "LOCKED_BEFORE_EXECUTION":
        raise RuntimeError("sparse-k protocol is not locked")
    if args.command == "generate": generate_data()
    elif args.command == "train": train(args.method, args.seed_index)
    elif args.command == "seal": seal()
    elif args.command == "evaluate": evaluate_family(args.family)
    else: aggregate()


if __name__ == "__main__":
    main()
