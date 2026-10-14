#!/usr/bin/env python3
"""Train the preregistered same-protocol kappa-conditioned transformer."""

from __future__ import annotations

import train_radial_factorization_ablation as base


base.METHODS = ("kappa_boundary_transformer",)
base.MODEL_SEEDS = (119401, 119402, 119403)
base.BATCH_SEEDS = (119501, 119502, 119503)
base.RESULT_ROOT = (
    base.ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "transformer"
)
base.PROTOCOL_PATH = base.ROOT / "protocols" / "radial_transfer_core_v1.json"


def model_config(method: str) -> dict:
    if method != "kappa_boundary_transformer":
        raise ValueError(method)
    return {"model": {"name": "kappa_btransformer", "kwargs": {
        "width": 128,
        "n_layers": 4,
        "heads": 4,
        "token_layers": 2,
        "kappa_scale": 4.0,
        "region": "interior",
    }}}


_config = model_config("kappa_boundary_transformer")
_model = base.build_model(_config)
base.EXPECTED_PARAMETERS = {
    "kappa_boundary_transformer": base.count_params(_model),
}
del _model
base.model_config = model_config


if __name__ == "__main__":
    base.main()
