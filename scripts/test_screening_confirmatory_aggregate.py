#!/usr/bin/env python
"""CPU checks that the locked aggregator pools and clusters the declared population."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_screening_confirmatory import (  # noqa: E402
    cluster_values,
    hidden_failures,
    mean_error,
    ratio_bootstrap,
)


def row(method, family, sample_id, error, seed=0):
    return {
        "method": method,
        "family": family,
        "load_family": "point_source",
        "kappa": 1.35,
        "query_margin": 0.12,
        "n_boundary": 128,
        "sample_id": sample_id,
        "seed_index": seed,
        "rel_l2": error,
        "finite": True,
    }


def main():
    rows = []
    for family, scale in (("in_dist", 1.0), ("rough", 2.0)):
        for sample_id in (0, 1):
            for seed in (0, 1, 2):
                rows.extend((
                    row("radial_mlp", family, sample_id, scale, seed),
                    row("screening_mix8", family, sample_id, scale, seed),
                    row("full_prior", family, sample_id, 2.0 * scale, seed),
                    row("no_prior", family, sample_id, 3.0 * scale, seed),
                ))
    assert mean_error(rows, "radial_mlp") == 1.5
    clusters = cluster_values(rows, "radial_mlp")
    assert len(clusters) == 4
    assert set(clusters) == {
        ("in_dist", 0), ("in_dist", 1), ("rough", 0), ("rough", 1),
    }
    interval = ratio_bootstrap(
        clusters, cluster_values(rows, "full_prior"), draws=500,
    )
    assert interval[1] < 1.0
    assert hidden_failures(rows, "screening_mix8", include_out_of_range=False) == []
    print("confirmatory aggregator: population and clustering checks passed")


if __name__ == "__main__":
    main()
