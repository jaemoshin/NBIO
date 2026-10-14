#!/usr/bin/env python3
"""Train the single preregistered s^2 log(s)-aware radial checkpoint."""

from __future__ import annotations

import train_radial_factorization_ablation as base


base.METHODS = ("radial_asymptotic",)
base.MODEL_SEEDS = (119101, 119111, 119121)
base.BATCH_SEEDS = (119102, 119112, 119122)
base.RESULT_ROOT = (
    base.ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "asymptotic_smoke"
)
base.PROTOCOL_PATH = base.ROOT / "protocols" / "radial_transfer_core_v1.json"
base.EXPECTED_PARAMETERS = {"radial_asymptotic": 2274}


def model_config(method: str) -> dict:
    if method != "radial_asymptotic":
        raise ValueError(method)
    return {"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_asymptotic",
        "hidden": 32,
        "n_layers": 3,
        "region": "interior",
        "solve": "direct",
        "kappa_scale": 4.0,
    }}}


base.model_config = model_config


if __name__ == "__main__":
    base.main()
