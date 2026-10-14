#!/usr/bin/env python3
"""Train the final load-normalized kappa-conditioned transformer baseline."""

from __future__ import annotations

import train_radial_factorization_ablation as base


base.METHODS = ("kappa_boundary_transformer_v2",)
base.MODEL_SEEDS = (119701, 119702, 119703)
base.BATCH_SEEDS = (119801, 119802, 119803)
base.RESULT_ROOT = (
    base.ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "transformer_v2"
)
base.PROTOCOL_PATH = (
    base.ROOT / "protocols" / "radial_transformer_normalized_v2.json"
)


def model_config(method: str) -> dict:
    if method != "kappa_boundary_transformer_v2":
        raise ValueError(method)
    return {"model": {"name": "kappa_btransformer", "kwargs": {
        "width": 128,
        "n_layers": 4,
        "heads": 4,
        "token_layers": 2,
        "kappa_scale": 4.0,
        "normalize_load": True,
        "include_weights": True,
        "region": "interior",
    }}}


_config = model_config("kappa_boundary_transformer_v2")
_model = base.build_model(_config)
base.EXPECTED_PARAMETERS = {
    "kappa_boundary_transformer_v2": base.count_params(_model),
}
del _model
base.model_config = model_config


if __name__ == "__main__":
    base.main()
