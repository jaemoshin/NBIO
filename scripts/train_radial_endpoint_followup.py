#!/usr/bin/env python3
"""Train endpoint-regular radial and matched prior-free controls."""

from __future__ import annotations

import train_radial_factorization_ablation as base


base.METHODS = ("generic_no_prior_matched", "radial_endpoint")
base.RESULT_ROOT = base.ROOT / "results" / "screening_law" / "radial_endpoint_v1"
base.PROTOCOL_PATH = base.ROOT / "protocols" / "radial_endpoint_v1.json"
base.EXPECTED_PARAMETERS = {
    "generic_no_prior_matched": 2221,
    "radial_endpoint": 2241,
}


def model_config(method: str) -> dict:
    if method == "generic_no_prior_matched":
        return {"model": {"name": "nbio_kappa", "kwargs": {
            "hidden": 30,
            "n_layers": 3,
            "feat_scale": 1.0,
            "region": "interior",
            "solve": "direct",
            "kappa_scale": 4.0,
            "prior_mode": "none",
            "residual_order": 2.0,
            "residual_scale": "normalized",
            "feature_mode": "smooth",
        }}}
    return {"model": {"name": "nbio_screening", "kwargs": {
        "variant": "radial_endpoint",
        "hidden": 32,
        "n_layers": 3,
        "region": "interior",
        "solve": "direct",
        "kappa_scale": 4.0,
    }}}


base.model_config = model_config


if __name__ == "__main__":
    base.main()
