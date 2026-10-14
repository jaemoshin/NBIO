#!/usr/bin/env python3
"""Generate the fresh quadratic-endpoint holdout."""

from __future__ import annotations

import radial_factorization_ablation_data as base


base.DATA_ROOT = base.ROOT / "data_cache" / "radial_quadratic_v1"
base.FAMILY_SEEDS = (116201, 116202, 116203, 116204)
base.KAPPAS = (1.15, 1.85, 2.55, 3.25, 3.85)


if __name__ == "__main__":
    base.main()
