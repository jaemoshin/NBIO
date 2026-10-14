#!/usr/bin/env python
"""OOD transfer audit for the validated continuously conditioned NBIO checkpoints.

Each invocation evaluates one geometry family. The protocol crosses geometry,
boundary-load family, kappa, boundary resolution, and query distance while retaining
correct-Yukawa and wrong-Laplace BIE controls on every instance.
"""
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

from src.data.bie import (  # noqa: E402
    bem_solve,
    boundary,
    eval_u,
    radius_at,
    sample_charges,
    sample_queries,
)
from src.utils.common import configure_torch, get_device  # noqa: E402
from eval_kappa_conditioned import checkpoint_model, instance_rel  # noqa: E402
from tmlr_revision_validation import FAMILIES, ood_coeffs  # noqa: E402


KAPPAS = (0.2, 1.1, 2.2, 3.8, 5.25)
IN_RANGE = frozenset((0.2, 1.1, 2.2, 3.8))
MARGINS = (0.12, 0.02)
RESOLUTIONS = (64, 128, 256)
LOAD_FAMILIES = ("point_source", "fourier")
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows, path: Path) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fourier_coefficients(rng: np.random.Generator, n_modes: int = 6) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / np.arange(1, n_modes + 1, dtype=np.float64) ** 1.5
    return rng.normal(size=n_modes) * scale, rng.normal(size=n_modes) * scale


def fourier_trace(theta: np.ndarray, coefficients: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a, b = coefficients
    modes = np.arange(1, len(a) + 1, dtype=np.float64)
    values = np.cos(theta[:, None] * modes) @ a + np.sin(theta[:, None] * modes) @ b
    rms = np.sqrt(np.mean(np.square(values)))
    return values / max(rms, 1e-12)


def make_bases(family: str, n_samples: int, n_q: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    settings = FAMILIES[family]
    fine_theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    rows = []
    while len(rows) < n_samples:
        coeffs = ood_coeffs(rng, settings)
        fine_radius = radius_at(coeffs, fine_theta)
        if fine_radius.min() <= 0.05:
            continue
        rows.append({
            "coeffs": coeffs,
            "charges": sample_charges(rng, "interior", coeffs, n_charges=6),
            "fourier": fourier_coefficients(rng),
            "queries": {
                margin: sample_queries(rng, coeffs, "interior", n_q=n_q, margin=margin)
                for margin in MARGINS
            },
            "min_radius": float(fine_radius.min()),
            "max_radius": float(fine_radius.max()),
        })
    return rows


def load_models(
    device: torch.device, checkpoint_root: Path
) -> list[tuple[str, int, torch.nn.Module, Path]]:
    models = []
    for method, tag in (("full_prior", "full"), ("no_prior", "none")):
        for data_seed in (0, 1, 2):
            checkpoint = checkpoint_root / f"data_s{data_seed}_{tag}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model, _ = checkpoint_model(checkpoint, device)
            models.append((method, data_seed, model, checkpoint))
    return models


def tensor_batch(cases: list[dict], kappa: float, device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("Xb", "Nb", "wb", "curv", "g", "Xq")
    batch = {
        key: torch.from_numpy(np.stack([case[key] for case in cases])).float().to(device)
        for key in keys
    }
    batch["kappa"] = torch.full((len(cases),), kappa, dtype=torch.float32, device=device)
    return batch


def reference_targets(
    bases: list[dict], kappa: float, reference_n_b: int
) -> dict[tuple[str, float, int], np.ndarray]:
    """Build analytic point-source and refined-BIE Fourier targets."""
    targets: dict[tuple[str, float, int], np.ndarray] = {}
    theta = 2.0 * np.pi * np.arange(reference_n_b, dtype=np.float64) / reference_n_b
    for sample_id, base in enumerate(bases):
        for margin in MARGINS:
            targets[("point_source", margin, sample_id)] = eval_u(
                base["queries"][margin], base["charges"], kappa
            )
        bd = boundary(base["coeffs"], reference_n_b)
        g_ref = fourier_trace(theta, base["fourier"])
        _, evaluator = bem_solve(bd, g_ref, "interior", kappa=kappa)
        for margin in MARGINS:
            targets[("fourier", margin, sample_id)] = evaluator(base["queries"][margin])
    return targets


def coarse_cases(
    bases: list[dict], targets: dict[tuple[str, float, int], np.ndarray],
    load_family: str, kappa: float, margin: float, n_b: int,
) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    theta = 2.0 * np.pi * np.arange(n_b, dtype=np.float64) / n_b
    cases = []
    controls = {"correct_yukawa_bie": [], "fixed_laplace_bie": []}
    metadata = []
    for sample_id, base in enumerate(bases):
        bd = boundary(base["coeffs"], n_b)
        if load_family == "point_source":
            g = eval_u(bd["X"], base["charges"], kappa)
        else:
            g = fourier_trace(theta, base["fourier"])
        target = targets[(load_family, margin, sample_id)]
        _, correct_eval = bem_solve(bd, g, "interior", kappa=kappa)
        _, wrong_eval = bem_solve(bd, g, "interior", kappa=0.0)
        controls["correct_yukawa_bie"].append(correct_eval(base["queries"][margin]))
        controls["fixed_laplace_bie"].append(wrong_eval(base["queries"][margin]))
        cases.append({
            "Xb": np.asarray(bd["X"], dtype=np.float32),
            "Nb": np.asarray(bd["normal"], dtype=np.float32),
            "wb": np.asarray(bd["w"], dtype=np.float32),
            "curv": np.asarray(bd["curv"], dtype=np.float32),
            "g": np.asarray(g, dtype=np.float32),
            "Xq": np.asarray(base["queries"][margin], dtype=np.float32),
            "target": np.asarray(target, dtype=np.float32),
        })
        metadata.append({
            "sample_id": sample_id,
            "min_radius": base["min_radius"],
            "max_radius": base["max_radius"],
            "max_abs_curvature": float(np.max(np.abs(bd["curv"]))),
        })
    return cases, {key: np.stack(value) for key, value in controls.items()}, metadata


def result_row(
    method: str, train_data_seed: int, family: str, load_family: str,
    kappa: float, margin: float, n_b: int, meta: dict, error: float,
    reference_n_b: int,
) -> dict:
    return {
        "method": method,
        "train_data_seed": train_data_seed,
        "geometry_family": family,
        "load_family": load_family,
        "kappa": kappa,
        "kappa_regime": "in_range" if kappa in IN_RANGE else "out_of_range",
        "query_margin": margin,
        "n_b": n_b,
        "reference_n_b": 0 if load_family == "point_source" else reference_n_b,
        "sample_id": meta["sample_id"],
        "rel_l2": float(error),
        "min_radius": meta["min_radius"],
        "max_radius": meta["max_radius"],
        "max_abs_curvature": meta["max_abs_curvature"],
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[list[dict], dict]:
    configure_torch()
    device = get_device(True)
    checkpoint_root = ROOT / args.checkpoint_root
    models = load_models(device, checkpoint_root)
    bases = make_bases(args.family, args.n_samples, args.n_q, args.seed)
    rows = []
    for kappa in KAPPAS:
        targets = reference_targets(bases, kappa, args.reference_n_b)
        for load_family in LOAD_FAMILIES:
            for margin in MARGINS:
                for n_b in RESOLUTIONS:
                    cases, controls, metadata = coarse_cases(
                        bases, targets, load_family, kappa, margin, n_b
                    )
                    target = torch.from_numpy(np.stack([case["target"] for case in cases])).float()
                    for method, predictions in controls.items():
                        errors = instance_rel(torch.from_numpy(predictions).float(), target).numpy()
                        for meta, error in zip(metadata, errors):
                            rows.append(result_row(
                                method, -1, args.family, load_family, kappa, margin,
                                n_b, meta, error, args.reference_n_b,
                            ))
                    batch = tensor_batch(cases, kappa, device)
                    target_device = target.to(device)
                    for method, data_seed, model, _ in models:
                        errors = instance_rel(model(batch), target_device).cpu().numpy()
                        for meta, error in zip(metadata, errors):
                            rows.append(result_row(
                                method, data_seed, args.family, load_family, kappa,
                                margin, n_b, meta, error, args.reference_n_b,
                            ))
                    print(
                        f"[{args.family}] kappa={kappa:g} load={load_family} "
                        f"margin={margin:g} nb={n_b}",
                        flush=True,
                    )
    manifest = {
        "protocol": "conditioned_learned_correction_ood_transfer",
        "geometry_family": args.family,
        "geometry_settings": FAMILIES[args.family],
        "geometry_seed": args.seed,
        "n_samples": args.n_samples,
        "n_query": args.n_q,
        "kappas": list(KAPPAS),
        "in_range_kappas": sorted(IN_RANGE),
        "margins": list(MARGINS),
        "resolutions": list(RESOLUTIONS),
        "load_families": list(LOAD_FAMILIES),
        "fourier_reference_n_b": args.reference_n_b,
        "checkpoint_root": str(checkpoint_root.relative_to(ROOT)),
        "checkpoints": [
            {
                "method": method,
                "train_data_seed": data_seed,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for method, data_seed, _, path in models
        ],
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=tuple(FAMILIES))
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--n-q", type=int, default=256)
    parser.add_argument("--reference-n-b", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--checkpoint-root", default="results/kappa_seed_replication"
    )
    parser.add_argument("--out-dir", default="results/tmlr_revision2")
    args = parser.parse_args()
    rows, manifest = evaluate(args)
    out = ROOT / args.out_dir
    write_csv(rows, out / f"conditioned_transfer_{args.family}.csv")
    (out / f"conditioned_transfer_{args.family}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
