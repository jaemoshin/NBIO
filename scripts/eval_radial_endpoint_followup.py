#!/usr/bin/env python3
"""Evaluate old and new matched controls on the fresh endpoint holdout."""

from __future__ import annotations

import eval_radial_factorization_ablation as base


base.DATA_ROOT = base.ROOT / "data_cache" / "radial_endpoint_v1" / "holdout"
base.RESULT_ROOT = base.ROOT / "results" / "screening_law" / "radial_endpoint_v1"
base.DATA_MANIFEST_PATH = base.ROOT / "protocols" / "radial_endpoint_data_manifest.json"
base.KAPPAS = (1.3, 1.9, 2.65, 3.35, 3.95)


if __name__ == "__main__":
    base.main()
