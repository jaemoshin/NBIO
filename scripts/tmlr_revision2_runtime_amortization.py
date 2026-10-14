#!/usr/bin/env python
"""Same-GPU NBIO timing with explicit geometry/RHS/query amortization contracts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset, to_device  # noqa: E402
from src.models.layers import double_layer_kernel  # noqa: E402
from src.utils.common import configure_torch, get_device  # noqa: E402
from eval_kappa_conditioned import checkpoint_model  # noqa: E402


RESOLUTIONS = (64, 128, 256)
N_RHS = (1, 8, 32)
N_QUERY = (64, 256, 1024)
KAPPA = 1.1


def shard_path(n_b: int) -> Path:
    return ROOT / "data_cache" / "kappa_confirmation" / f"test_k1p1_m120_nb{n_b}.pt"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(fn, device: torch.device, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return 1000.0 * (time.perf_counter() - start) / repeats


def query_slice(xq: torch.Tensor, n_q: int) -> torch.Tensor:
    if n_q <= xq.shape[1]:
        return xq[:, :n_q]
    repeats = (n_q + xq.shape[1] - 1) // xq.shape[1]
    return xq.repeat(1, repeats, 1)[:, :n_q]


def rhs_stack(g: torch.Tensor, n_rhs: int) -> torch.Tensor:
    if n_rhs == 1:
        return g.unsqueeze(-1)
    phase = torch.linspace(0.0, 2.0 * torch.pi, n_rhs, device=g.device, dtype=g.dtype)
    scale = 0.75 + 0.25 * torch.cos(phase)
    return g.unsqueeze(-1) * scale.reshape(1, 1, -1)


@torch.no_grad()
def run(args: argparse.Namespace) -> tuple[list[dict], dict]:
    configure_torch()
    device = get_device(True)
    checkpoint_root = ROOT / args.checkpoint_root
    full_path = checkpoint_root / "data_s0_full" / "best.pt"
    none_path = checkpoint_root / "data_s0_none" / "best.pt"
    full, _ = checkpoint_model(full_path, device)
    none, _ = checkpoint_model(none_path, device)
    models = {
        "conditioned_full_prior": full,
        "conditioned_no_prior": none,
    }
    rows = []
    for n_b in RESOLUTIONS:
        dataset = BIEDataset(shard_path(n_b))
        batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
        batch = to_device(batch, device)
        batch["kappa"] = torch.full(
            (batch["Xb"].shape[0],), KAPPA, device=device, dtype=batch["Xb"].dtype
        )
        jump = -0.5

        def structure_assembly():
            kernel = double_layer_kernel(
                batch["Xb"], batch["Xb"], batch["Nb"],
                curv_s=batch["curv"], self_kernel=True,
            )
            identity = torch.eye(n_b, device=device, dtype=batch["Xb"].dtype).unsqueeze(0)
            return jump * identity + kernel * batch["wb"][:, None, :]

        assembly_fns = {"structure_only_laplace": structure_assembly}
        assembly_fns.update({
            name: (lambda model=model: model.bio.assemble(
                batch["Xb"], batch["Nb"], batch["wb"], batch["curv"],
                batch["kappa"], jump,
            ))
            for name, model in models.items()
        })

        potential_fns = {
            "structure_only_laplace": lambda xq: (
                double_layer_kernel(xq, batch["Xb"], batch["Nb"])
                * batch["wb"][:, None, :]
            ),
        }
        potential_fns.update({
            name: (lambda xq, model=model: model.bio.potential(
                xq, batch["Xb"], batch["Nb"], batch["wb"], batch["kappa"]
            ))
            for name, model in models.items()
        })

        for method in assembly_fns:
            assembly_fn = assembly_fns[method]
            assembly_ms = timed_ms(assembly_fn, device, args.repeats, args.warmup)
            matrix = assembly_fn()
            factor_ms = timed_ms(
                lambda: torch.linalg.lu_factor(matrix), device, args.repeats, args.warmup
            )
            lu, pivots = torch.linalg.lu_factor(matrix)
            solve_times = {}
            densities = {}
            for n_rhs in N_RHS:
                rhs = rhs_stack(batch["g"], n_rhs)
                solve_fn = lambda rhs=rhs: torch.linalg.lu_solve(lu, pivots, rhs)
                solve_times[n_rhs] = timed_ms(
                    solve_fn, device, args.repeats, args.warmup
                )
                densities[n_rhs] = solve_fn()

            potential_times = {}
            potentials = {}
            for n_q in N_QUERY:
                xq = query_slice(batch["Xq"], n_q)
                potential_fn = lambda xq=xq: potential_fns[method](xq)
                potential_times[n_q] = timed_ms(
                    potential_fn, device, args.repeats, args.warmup
                )
                potentials[n_q] = potential_fn()

            for n_rhs in N_RHS:
                for n_q in N_QUERY:
                    potential = potentials[n_q]
                    density = densities[n_rhs]
                    apply_fn = lambda potential=potential, density=density: torch.bmm(
                        potential, density
                    )
                    apply_ms = timed_ms(apply_fn, device, args.repeats, args.warmup)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    output = apply_fn()
                    sync(device)
                    peak_mb = (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else float("nan")
                    )
                    cached_geometry_ms = solve_times[n_rhs] + potential_times[n_q] + apply_ms
                    cached_geometry_query_ms = potential_times[n_q] + apply_ms
                    cached_geometry_rhs_query_ms = solve_times[n_rhs] + apply_ms
                    new_geometry_ms = assembly_ms + factor_ms + cached_geometry_ms
                    rows.append({
                        "method": method,
                        "device": str(device),
                        "dtype": str(batch["Xb"].dtype).replace("torch.", ""),
                        "batch_size": args.batch_size,
                        "n_b": n_b,
                        "n_rhs": n_rhs,
                        "n_query": n_q,
                        "assembly_ms_batch": assembly_ms,
                        "factorization_ms_batch": factor_ms,
                        "solve_ms_batch": solve_times[n_rhs],
                        "potential_assembly_ms_batch": potential_times[n_q],
                        "potential_apply_ms_batch": apply_ms,
                        "new_geometry_total_ms_batch": new_geometry_ms,
                        "cached_geometry_new_rhs_and_query_ms_batch": cached_geometry_ms,
                        "cached_geometry_new_query_ms_batch": cached_geometry_query_ms,
                        "cached_geometry_cached_query_ms_batch": cached_geometry_rhs_query_ms,
                        "cached_geometry_cached_query_ms_per_rhs_sample": (
                            cached_geometry_rhs_query_ms / (args.batch_size * n_rhs)
                        ),
                        "peak_allocated_mb_after_apply": peak_mb,
                        "output_finite": bool(torch.isfinite(output).all()),
                    })
            print(f"[runtime] method={method} nb={n_b}", flush=True)
    manifest = {
        "protocol": "same_gpu_component_and_reuse_timing",
        "kappa": KAPPA,
        "resolutions": list(RESOLUTIONS),
        "n_rhs": list(N_RHS),
        "n_query": list(N_QUERY),
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "query_note": "n_query=1024 tiles the frozen 256 query coordinates; timing depends on tensor shape, not uniqueness",
        "contract_note": (
            "new_geometry includes assembly and LU factorization; cached-geometry rows reuse the operator factorization. "
            "No speed superiority over optimized BIE software or amortized neural surrogates is claimed."
        ),
        "checkpoint_root": str(checkpoint_root.relative_to(ROOT)),
        "checkpoints": [str(full_path.relative_to(ROOT)), str(none_path.relative_to(ROOT))],
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--checkpoint-root", default="results/kappa_seed_replication"
    )
    parser.add_argument("--out-dir", default="results/tmlr_revision2")
    args = parser.parse_args()
    rows, manifest = run(args)
    out = ROOT / args.out_dir
    write_csv(rows, out / "runtime_amortization.csv")
    (out / "runtime_amortization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(rows)} timing rows to {out}")


if __name__ == "__main__":
    main()
