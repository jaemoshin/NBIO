#!/usr/bin/env python3
"""Same-GPU online runtime for radial NBIO and the normalized Transformer."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, count_params, get_device  # noqa: E402

RESULT_ROOT = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "runtime"
)
RADIAL_MANIFEST = (
    ROOT / "results" / "screening_law" / "radial_quadratic_v1"
    / "checkpoint_manifest.json"
)
TRANSFORMER_MANIFEST = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "transformer_v2" / "checkpoint_manifest.json"
)
SHARD = (
    ROOT / "data_cache" / "radial_transfer_core_v1" / "transformer" / "holdout"
    / "point_source_in_dist_k2p3_nb128.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(manifest_path: Path, method: str, device: torch.device):
    manifest = json.loads(manifest_path.read_text())
    record = next(
        item for item in manifest["records"]
        if item["method"] == method and item["seed_index"] == 0
    )
    path = ROOT / record["path"]
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def timing(model, batch, warmup=20, repeats=100):
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model(batch)
            stop.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(stop)))
    return values


def main() -> None:
    outputs = (RESULT_ROOT / "runtime.csv", RESULT_ROOT / "runtime.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("runtime outputs already exist")
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    shard = torch.load(SHARD, map_location="cpu", weights_only=False)
    batch = {
        key: value[:1].to(device)
        for key, value in shard.items()
    }
    models = {
        "radial_normal": load_model(
            RADIAL_MANIFEST, "radial_normal", device,
        ),
        "kappa_boundary_transformer_v2": load_model(
            TRANSFORMER_MANIFEST, "kappa_boundary_transformer_v2", device,
        ),
    }
    rows = []
    raw = {}
    for method, model in models.items():
        values = timing(model, batch)
        raw[method] = values
        rows.append({
            "method": method,
            "device": torch.cuda.get_device_name(device),
            "batch_size": 1,
            "n_boundary": batch["Xb"].shape[1],
            "n_query": batch["Xq"].shape[1],
            "parameters": count_params(model),
            "warmup": 20,
            "repeats": len(values),
            "median_ms_per_sample": statistics.median(values),
            "mean_ms_per_sample": statistics.mean(values),
            "p90_ms_per_sample": sorted(values)[int(0.9 * len(values))],
            "contract": (
                "assembly+dense solve+potential"
                if method == "radial_normal" else "amortized forward"
            ),
        })
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    outputs[1].write_text(json.dumps({
        "rows": rows,
        "raw_ms": raw,
        "input_transfer_excluded": True,
        "same_gpu": True,
        "checkpoint_policy": "seed_index_0, validation-selected and sealed",
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
