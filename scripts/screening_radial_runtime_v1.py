#!/usr/bin/env python
"""Controlled same-GPU runtime audit for exact, radial, and generic screened BIEs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402
from src.models.nbio_yukawa import yukawa_double_layer_kernel  # noqa: E402
from src.utils.common import configure_torch, count_params, get_device  # noqa: E402

SOURCE = Path(os.environ.get("NBIO_SCREENING_SOURCE", ROOT))
RESULT_ROOT = ROOT / "results" / "screening_law" / "runtime_v1"
REPORT = ROOT / "reports" / "screening_law_runtime_v1.md"
KAPPA = 2.55
RESOLUTIONS = (64, 128, 256)
N_RHS = (1, 8, 32)
N_QUERY = (64, 256, 1024)
METHOD_MAP = {
    "full_prior": "generic_full_prior",
    "no_prior": "generic_no_prior",
    "radial_mlp": "radial_normal_form",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(function, device, repeats, warmup):
    for _ in range(warmup):
        function()
    sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        sync(device)
        samples.append(1000.0 * (time.perf_counter() - start))
    return {
        "median": float(np.median(samples)),
        "q25": float(np.quantile(samples, 0.25)),
        "q75": float(np.quantile(samples, 0.75)),
    }


def load_models(device):
    manifest_path = SOURCE / "results" / "screening_law" / "confirmatory_v1" / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    models = {}
    records = []
    for record in manifest["records"]:
        if record["seed_index"] != 0 or record["method"] not in METHOD_MAP:
            continue
        path = SOURCE / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval()
        name = METHOD_MAP[record["method"]]
        models[name] = model
        records.append({
            "method": name,
            "path": str(path),
            "sha256": record["sha256"],
            "parameters": count_params(model),
        })
    if len(models) != 3:
        raise RuntimeError(f"expected three learned models, got {sorted(models)}")
    return models, records


def query_slice(xq, count):
    if count <= xq.shape[1]:
        return xq[:, :count]
    copies = (count + xq.shape[1] - 1) // xq.shape[1]
    return xq.repeat(1, copies, 1)[:, :count]


def rhs_stack(g, count):
    phase = torch.linspace(0.0, 2.0 * torch.pi, count, device=g.device, dtype=g.dtype)
    scale = 0.75 + 0.25 * torch.cos(phase)
    return g.unsqueeze(-1) * scale.reshape(1, 1, -1)


def assembly_and_potential(method, model, batch):
    n_boundary = batch["Xb"].shape[1]
    identity = torch.eye(n_boundary, device=batch["Xb"].device, dtype=batch["Xb"].dtype).unsqueeze(0)
    if method == "exact_yukawa_structure":
        def assembly():
            kernel = yukawa_double_layer_kernel(
                batch["Xb"], batch["Xb"], batch["Nb"], KAPPA,
                curv_s=batch["curv"], self_kernel=True,
            )
            return -0.5 * identity + kernel * batch["wb"][:, None]

        def potential(query):
            return (
                yukawa_double_layer_kernel(query, batch["Xb"], batch["Nb"], KAPPA)
                * batch["wb"][:, None]
            )
        return assembly, potential
    if method == "radial_normal_form":
        return (
            lambda: model.assemble(batch),
            lambda query: model.potential({**batch, "Xq": query}),
        )
    return (
        lambda: model.bio.assemble(
            batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
            batch["kappa"], model.jump,
        ),
        lambda query: model.bio.potential(
            query, batch["Xb"], batch["Nb"], batch["wb"], batch["kappa"],
        ),
    )


@torch.no_grad()
def run(args):
    configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = get_device(True)
    models, checkpoint_records = load_models(device)
    models = {"exact_yukawa_structure": None, **models}
    rows = []
    for n_boundary in RESOLUTIONS:
        path = SOURCE / "data_cache" / "screening_confirmatory_v1" / "confirmation" / (
            f"point_source_in_dist_k2p55_m120_nb{n_boundary}.pt"
        )
        data = torch.load(path, map_location="cpu", weights_only=False)
        batch = {
            key: value[:args.batch_size].to(device)
            for key, value in data.items() if key != "bem_uq"
        }
        for method, model in models.items():
            assembly_fn, potential_builder = assembly_and_potential(method, model, batch)
            assembly = timed_ms(assembly_fn, device, args.repeats, args.warmup)
            matrix = assembly_fn()
            factorization = timed_ms(
                lambda: torch.linalg.lu_factor(matrix), device, args.repeats, args.warmup,
            )
            lu, pivots = torch.linalg.lu_factor(matrix)
            solve = {}
            densities = {}
            for n_rhs in N_RHS:
                rhs = rhs_stack(batch["g"], n_rhs)
                function = lambda rhs=rhs: torch.linalg.lu_solve(lu, pivots, rhs)
                solve[n_rhs] = timed_ms(function, device, args.repeats, args.warmup)
                densities[n_rhs] = function()
            potentials = {}
            potential_time = {}
            for n_query in N_QUERY:
                query = query_slice(batch["Xq"], n_query)
                function = lambda query=query: potential_builder(query)
                potential_time[n_query] = timed_ms(
                    function, device, args.repeats, args.warmup,
                )
                potentials[n_query] = function()
            for n_rhs in N_RHS:
                for n_query in N_QUERY:
                    potential = potentials[n_query]
                    density = densities[n_rhs]
                    apply_fn = lambda: torch.bmm(potential, density)
                    apply = timed_ms(apply_fn, device, args.repeats, args.warmup)
                    new_geometry = (
                        assembly["median"] + factorization["median"]
                        + solve[n_rhs]["median"] + potential_time[n_query]["median"]
                        + apply["median"]
                    )
                    cached = (
                        solve[n_rhs]["median"] + potential_time[n_query]["median"]
                        + apply["median"]
                    )
                    rows.append({
                        "method": method,
                        "parameters": 0 if model is None else count_params(model),
                        "device": str(device),
                        "dtype": "float32",
                        "batch_size": args.batch_size,
                        "n_boundary": n_boundary,
                        "n_rhs": n_rhs,
                        "n_query": n_query,
                        "assembly_ms_median": assembly["median"],
                        "assembly_ms_q25": assembly["q25"],
                        "assembly_ms_q75": assembly["q75"],
                        "factorization_ms_median": factorization["median"],
                        "solve_ms_median": solve[n_rhs]["median"],
                        "potential_ms_median": potential_time[n_query]["median"],
                        "apply_ms_median": apply["median"],
                        "new_geometry_total_ms_median": new_geometry,
                        "cached_geometry_total_ms_median": cached,
                        "cached_factorized_solve_ms_per_rhs_sample": (
                            solve[n_rhs]["median"] / (args.batch_size * n_rhs)
                        ),
                        "finite": bool(torch.isfinite(apply_fn()).all()),
                    })
            print(f"[runtime] {method} n={n_boundary}", flush=True)
    return rows, {
        "protocol": "screening_radial_runtime_v1",
        "device": str(device),
        "dtype": "float32",
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "kappa": KAPPA,
        "resolutions": list(RESOLUTIONS),
        "n_rhs": list(N_RHS),
        "n_query": list(N_QUERY),
        "checkpoints": checkpoint_records,
        "contract": "All methods assemble and factor a geometry-specific dense second-kind system on the same GPU. Cached rows reuse the LU factorization.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    rows, manifest = run(args)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (RESULT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    primary = [row for row in rows if row["n_boundary"] == 128
               and row["n_rhs"] == 1 and row["n_query"] == 256]
    summary = {row["method"]: {
        "parameters": row["parameters"],
        "assembly_ms": row["assembly_ms_median"],
        "solve_ms": row["solve_ms_median"],
        "potential_ms": row["potential_ms_median"],
        "new_geometry_total_ms": row["new_geometry_total_ms_median"],
    } for row in primary}
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    radial = summary["radial_normal_form"]
    generic = summary["generic_full_prior"]
    exact = summary["exact_yukawa_structure"]
    REPORT.write_text(f"""# Radial screening-law runtime audit

All rows use the same GPU, float32 tensors, batch size `{args.batch_size}`, and identical dense
geometry-specific solve contract. Times are medians over `{args.repeats}` synchronized repeats after
`{args.warmup}` warmups.

| method | params | assembly ms | solve ms | potential ms | total ms |
|---|---:|---:|---:|---:|---:|
| exact Yukawa structure | 0 | `{exact['assembly_ms']:.3f}` | `{exact['solve_ms']:.3f}` | `{exact['potential_ms']:.3f}` | `{exact['new_geometry_total_ms']:.3f}` |
| radial normal form | 2,241 | `{radial['assembly_ms']:.3f}` | `{radial['solve_ms']:.3f}` | `{radial['potential_ms']:.3f}` | `{radial['new_geometry_total_ms']:.3f}` |
| generic full prior | 9,089 | `{generic['assembly_ms']:.3f}` | `{generic['solve_ms']:.3f}` | `{generic['potential_ms']:.3f}` | `{generic['new_geometry_total_ms']:.3f}` |

The timing establishes component cost under a matched structured-solver contract. It is not a comparison
to an amortized surrogate or an optimized production BEM library.
""")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
