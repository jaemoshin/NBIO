#!/usr/bin/env python3
"""Fresh split-QBX confirmation for the endpoint-regular radial model."""

from __future__ import annotations

import json

import torch

import screening_split_qbx_v1 as base

from src.models import build_model


base.RESULT_ROOT = base.ROOT / "results" / "screening_law" / "split_qbx_endpoint_v1"
base.REPORT = base.ROOT / "reports" / "screening_law_split_qbx_endpoint_v1.md"
base.FIGURE = base.ROOT / "figures" / "screening_law_split_qbx_endpoint_v1.png"
base.FAMILY_SEEDS = {"in_dist": 115201, "rough": 115202}
base.PROTOCOL_NAME = "screening_law_split_qbx_endpoint_v1"


def load_endpoint_models(device: torch.device):
    manifest_path = (
        base.ROOT / "results" / "screening_law" / "radial_endpoint_v1"
        / "checkpoint_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    models = []
    for record in manifest["records"]:
        if record["method"] != "radial_endpoint":
            continue
        path = base.ROOT / record["path"]
        if base.sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device, dtype=torch.float64).eval()
        model.qbx_kw = dict(base.QBX_KWARGS)
        models.append((record["seed_index"], model))
    if len(models) != 3:
        raise RuntimeError(f"expected three endpoint checkpoints, found {len(models)}")
    return models, manifest_path


base.load_radial_models = load_endpoint_models


if __name__ == "__main__":
    base.main()
