#!/usr/bin/env python3
"""Fresh split-QBX comparison for original and asymptotic radial laws."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import screening_split_qbx_v1 as qbx_base  # noqa: E402
from src.data.bie import bem_solve, boundary, eval_u  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from tmlr_revision_validation import FAMILIES  # noqa: E402

RESULT_ROOT = (
    ROOT / "results" / "screening_law" / "radial_transfer_core_v1"
    / "asymptotic_smoke"
)
FROZEN_MANIFEST = (
    ROOT / "results" / "screening_law" / "radial_quadratic_v1"
    / "checkpoint_manifest.json"
)
ASYMPTOTIC_MANIFEST = RESULT_ROOT / "checkpoint_manifest.json"
FAMILY_SEEDS = {"in_dist": 119301, "rough": 119302}
KAPPAS = (1.3, 2.5, 3.7)
DISTANCES = (0.01, 0.005, 0.001, 0.0005)
RESOLUTIONS = (128, 256, 512)
LOADS = ("point_source", "fourier_bessel")
N_GEOMETRIES = 12
N_QUERY = 128
QBX_KWARGS = {"order": 24, "r_fac": 2.0, "near_fac": 2.0, "oversample": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(device: torch.device):
    models = []
    frozen = json.loads(FROZEN_MANIFEST.read_text())
    for record in frozen["records"]:
        if record["method"] != "radial_normal":
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device, dtype=torch.float64).eval()
        model.qbx_kw = dict(QBX_KWARGS)
        models.append(("radial_normal_split_qbx", record["seed_index"], model))
    asymptotic = json.loads(ASYMPTOTIC_MANIFEST.read_text())
    for record in asymptotic["records"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device, dtype=torch.float64).eval()
        model.qbx_kw = dict(QBX_KWARGS)
        models.append(("radial_asymptotic_split_qbx", record["seed_index"], model))
    if len(models) != 4:
        raise RuntimeError(f"expected four radial models, found {len(models)}")
    return models


def make_geometries(family: str) -> list[dict]:
    original_seeds = qbx_base.FAMILY_SEEDS
    original_count = qbx_base.N_GEOMETRIES
    original_query_count = qbx_base.N_QUERY
    try:
        qbx_base.FAMILY_SEEDS = dict(FAMILY_SEEDS)
        qbx_base.N_GEOMETRIES = N_GEOMETRIES
        qbx_base.N_QUERY = N_QUERY
        return qbx_base.make_geometries(family)
    finally:
        qbx_base.FAMILY_SEEDS = original_seeds
        qbx_base.N_GEOMETRIES = original_count
        qbx_base.N_QUERY = original_query_count


def write_csv(rows: list[dict], path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_rows(
    rows, values, *, method, seed, family, kappa, load, distance, n_boundary,
):
    for sample_id, value in enumerate(values):
        rows.append({
            "method": method,
            "seed_index": seed,
            "family": family,
            "kappa": kappa,
            "load_family": load,
            "distance": distance,
            "n_boundary": n_boundary,
            "sample_id": sample_id,
            "rel_l2": float(value),
            "finite": bool(np.isfinite(value)),
        })


@torch.no_grad()
def run_family(family: str) -> None:
    output = RESULT_ROOT / "qbx_raw" / f"{family}.csv"
    if output.exists():
        raise FileExistsError(output)
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    geometries = make_geometries(family)
    models = load_models(device)
    rows = []
    for kappa in KAPPAS:
        for n_boundary in RESOLUTIONS:
            curves = [
                boundary(geometry["coefficients"], n_boundary)
                for geometry in geometries
            ]
            for load in LOADS:
                traces = np.asarray([
                    eval_u(curve["X"], geometry["charges"], kappa)
                    if load == "point_source" else
                    qbx_base.fourier_solution(
                        curve["X"], kappa, geometry["fourier"],
                    )
                    for curve, geometry in zip(curves, geometries)
                ])
                targets = {
                    distance: np.asarray([
                        eval_u(
                            geometry["queries"][distance],
                            geometry["charges"],
                            kappa,
                        )
                        if load == "point_source" else
                        qbx_base.fourier_solution(
                            geometry["queries"][distance],
                            kappa,
                            geometry["fourier"],
                        )
                        for geometry in geometries
                    ])
                    for distance in DISTANCES
                }
                exact_densities = []
                for curve, trace in zip(curves, traces):
                    density, _ = bem_solve(
                        curve, trace, "interior", kappa=kappa,
                    )
                    exact_densities.append(density)
                exact_density = torch.tensor(
                    np.asarray(exact_densities),
                    dtype=torch.float64,
                    device=device,
                )
                base = qbx_base.tensor_batch(
                    curves,
                    traces,
                    [geometry["queries"][DISTANCES[0]] for geometry in geometries],
                    kappa,
                    device,
                )
                learned_densities = []
                for method, seed, model in models:
                    matrix = model.assemble(base)
                    density = model.solver(matrix, base["g"], model.jump)
                    learned_densities.append((method, seed, model, density))
                for distance in DISTANCES:
                    queries = [
                        geometry["queries"][distance] for geometry in geometries
                    ]
                    batch = qbx_base.tensor_batch(
                        curves, traces, queries, kappa, device,
                    )
                    target = targets[distance]
                    original_qbx = qbx_base.QBX_KWARGS
                    try:
                        qbx_base.QBX_KWARGS = QBX_KWARGS
                        exact_prediction = qbx_base.exact_split_qbx(
                            batch, exact_density,
                        ).cpu().numpy()
                    finally:
                        qbx_base.QBX_KWARGS = original_qbx
                    append_rows(
                        rows,
                        qbx_base.rel_errors(exact_prediction, target),
                        method="exact_yukawa_split_qbx",
                        seed=-1,
                        family=family,
                        kappa=kappa,
                        load=load,
                        distance=distance,
                        n_boundary=n_boundary,
                    )
                    for method, seed, model, density in learned_densities:
                        model.qbx = True
                        prediction = model.evaluate_density(
                            batch, density,
                        ).cpu().numpy()
                        append_rows(
                            rows,
                            qbx_base.rel_errors(prediction, target),
                            method=method,
                            seed=seed,
                            family=family,
                            kappa=kappa,
                            load=load,
                            distance=distance,
                            n_boundary=n_boundary,
                        )
                print(
                    f"[asymptotic-qbx/{family}] k={kappa:g} "
                    f"load={load} nb={n_boundary}",
                    flush=True,
                )
    expected = (
        len(KAPPAS) * len(LOADS) * len(RESOLUTIONS) * len(DISTANCES)
        * N_GEOMETRIES * 5
    )
    if len(rows) != expected or not all(row["finite"] for row in rows):
        raise RuntimeError(f"coverage/finite failure: {len(rows)} != {expected}")
    write_csv(rows, output)
    (RESULT_ROOT / "qbx_raw" / f"{family}_manifest.json").write_text(
        json.dumps({
            "family": family,
            "seed": FAMILY_SEEDS[family],
            "row_count": len(rows),
            "qbx": QBX_KWARGS,
            "frozen_manifest_sha256": sha256(FROZEN_MANIFEST),
            "asymptotic_manifest_sha256": sha256(ASYMPTOTIC_MANIFEST),
        }, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILY_SEEDS), required=True)
    args = parser.parse_args()
    run_family(args.family)


if __name__ == "__main__":
    main()
