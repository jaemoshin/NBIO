#!/usr/bin/env python
"""GPU smoke for spline differentiation and exact screened-BIE evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reviewer3_exact_screened import ExactScreenedBIE  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.trainer import rel_l2  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke")
    device = torch.device("cuda")
    path = (
        ROOT / "data_cache" / "radial_factorization_ablation_v1" / "holdout"
        / "point_source_in_dist_k2p4_nb128.pt"
    )
    shard = torch.load(path, map_location="cpu", weights_only=False)
    batch = {
        key: value[:2].to(device) for key, value in shard.items()
        if key != "bem_uq"
    }
    exact = ExactScreenedBIE().to(device)
    exact_prediction = exact(batch)
    if not torch.isfinite(exact_prediction).all():
        raise RuntimeError("exact screened BIE produced nonfinite output")
    spline = build_model({"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_spline", "n_spline_coefficients": 2241,
        "spline_s_max": 16.0,
    }}}).to(device)
    prediction = spline(batch)
    loss = rel_l2(prediction, batch["uq"])
    loss.backward()
    gradient = spline.spline_coefficients.grad
    if not torch.isfinite(loss) or gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("spline forward/backward smoke failed")
    print(
        f"reviewer-3 GPU smoke passed on {torch.cuda.get_device_name(device)}; "
        f"zero-spline loss={float(loss):.4e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
