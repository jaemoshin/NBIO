#!/usr/bin/env python3
"""Train the direct quadratic endpoint radial law under the matched protocol."""

from __future__ import annotations

import train_radial_factorization_ablation as base


base.METHODS = ("radial_quadratic",)
base.RESULT_ROOT = base.ROOT / "results" / "screening_law" / "radial_quadratic_v1"
base.PROTOCOL_PATH = base.ROOT / "protocols" / "radial_quadratic_v1.json"
base.EXPECTED_PARAMETERS = {"radial_quadratic": 2241}


def model_config(method: str) -> dict:
    return {"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_quadratic",
        "hidden": 32,
        "n_layers": 3,
        "region": "interior",
        "solve": "direct",
        "kappa_scale": 4.0,
    }}}


base.model_config = model_config


if __name__ == "__main__":
    base.main()
