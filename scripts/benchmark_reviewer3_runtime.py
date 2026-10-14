#!/usr/bin/env python
"""Unified exact-BIE, radial-NBIO, and Transformer runtime harness."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get("NBIO_SCREENING_SOURCE", ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reviewer3_exact_screened import ExactScreenedBIE  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, count_params, get_device  # noqa: E402

RESULT_ROOT = ROOT / "results" / "screening_law" / "reviewer3_evidence_v1"
RADIAL_MANIFEST = ROOT / "results" / "screening_law" / "radial_paperization_v1" / "checkpoint_manifest.json"
TRANSFORMER_MANIFEST = ROOT / "results" / "screening_law" / "radial_transfer_core_v1" / "transformer_v2" / "checkpoint_manifest.json"
SHARD = ROOT / "data_cache" / "radial_transfer_core_v1" / "transformer" / "holdout" / "point_source_in_dist_k2p3_nb128.pt"
WARMUP = 30
REPEATS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(manifest_path: Path, method: str, device: torch.device):
    manifest = json.loads(manifest_path.read_text())
    record = next(
        row for row in manifest["records"]
        if row["method"] == method and row["seed_index"] == 0
    )
    path = ROOT / record["path"]
    if not path.exists():
        path = FROZEN_ROOT / record["path"]
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), record


def events(callable_):
    with torch.inference_mode():
        for _ in range(WARMUP):
            callable_()
        torch.cuda.synchronize()
        values = []
        for _ in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            callable_()
            stop.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(stop)))
    return values


def summarize(method, contract, batch_size, parameters, values, memory_bytes):
    ordered = sorted(values)
    median = statistics.median(values)
    return {
        "method": method,
        "contract": contract,
        "batch_size": batch_size,
        "n_boundary": 128,
        "n_query": 256,
        "parameters": parameters,
        "median_ms_per_batch": median,
        "median_ms_per_sample": median / batch_size,
        "mean_ms_per_sample": statistics.mean(values) / batch_size,
        "p90_ms_per_sample": ordered[int(0.9 * len(ordered))] / batch_size,
        "throughput_samples_per_second": 1000.0 * batch_size / median,
        "peak_incremental_memory_mb": memory_bytes / (1024 ** 2),
        "warmup": WARMUP,
        "repeats": REPEATS,
    }


def main() -> None:
    csv_output = RESULT_ROOT / "raw" / "unified_runtime.csv"
    json_output = RESULT_ROOT / "raw" / "unified_runtime.json"
    if csv_output.exists() or json_output.exists():
        raise FileExistsError("unified runtime output exists")
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    shard = torch.load(SHARD, map_location="cpu", weights_only=False)
    radial, radial_record = load_model(RADIAL_MANIFEST, "radial_normal", device)
    transformer, transformer_record = load_model(
        TRANSFORMER_MANIFEST, "kappa_boundary_transformer_v2", device,
    )
    exact = ExactScreenedBIE().to(device).eval()
    models = {
        "exact_screened_bie": (exact, 0, "new_geometry_assembly+solve+potential"),
        "radial_normal": (radial, count_params(radial), "new_geometry_assembly+solve+potential"),
        "kappa_boundary_transformer_v2": (
            transformer, count_params(transformer), "amortized_full_forward",
        ),
    }
    rows = []
    raw = {}
    for batch_size in (1, 8, 32):
        batch = {key: value[:batch_size].to(device) for key, value in shard.items()}
        for method, (model, parameters, contract) in models.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            with torch.inference_mode():
                model(batch)
            torch.cuda.synchronize()
            memory = max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
            values = events(lambda model=model, batch=batch: model(batch))
            raw[f"{method}__batch{batch_size}__new_geometry"] = values
            rows.append(summarize(
                method, contract, batch_size, parameters, values, memory,
            ))

        for method, model, parameters in (
            ("exact_screened_bie", exact, 0),
            ("radial_normal", radial, count_params(radial)),
        ):
            with torch.inference_mode():
                matrix = model.assemble(batch)
                potential = model.potential(batch)

            def cached_call(matrix=matrix, potential=potential, batch=batch):
                density = torch.linalg.solve(matrix, batch["g"].unsqueeze(-1))
                return torch.bmm(potential, density).squeeze(-1)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            with torch.inference_mode():
                cached_call()
            torch.cuda.synchronize()
            memory = max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
            values = events(cached_call)
            raw[f"{method}__batch{batch_size}__cached_geometry"] = values
            rows.append(summarize(
                method, "cached_geometry_repeated_rhs_solve+potential",
                batch_size, parameters, values, memory,
            ))
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps({
        "rows": rows,
        "raw_ms_per_batch": raw,
        "device": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "allow_tf32": False,
        "input_transfer_excluded": True,
        "checkpoint_loading_excluded": True,
        "radial_checkpoint": radial_record,
        "transformer_checkpoint": transformer_record,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
