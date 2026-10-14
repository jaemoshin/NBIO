#!/usr/bin/env python3
"""Generate a fresh holdout for the endpoint-regular radial confirmation."""

from __future__ import annotations

import radial_factorization_ablation_data as base


base.DATA_ROOT = base.ROOT / "data_cache" / "radial_endpoint_v1"
base.FAMILY_SEEDS = (114201, 114202, 114203, 114204)
base.KAPPAS = (1.3, 1.9, 2.65, 3.35, 3.95)


if __name__ == "__main__":
    base.main()
