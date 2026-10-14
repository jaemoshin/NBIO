#!/usr/bin/env python3
"""Evaluate all frozen controls and the quadratic endpoint model."""

from __future__ import annotations

import eval_radial_factorization_ablation as base


base.DATA_ROOT = base.ROOT / "data_cache" / "radial_quadratic_v1" / "holdout"
base.RESULT_ROOT = base.ROOT / "results" / "screening_law" / "radial_quadratic_v1"
base.DATA_MANIFEST_PATH = base.ROOT / "protocols" / "radial_quadratic_data_manifest.json"
base.KAPPAS = (1.15, 1.85, 2.55, 3.25, 3.85)


if __name__ == "__main__":
    base.main()
