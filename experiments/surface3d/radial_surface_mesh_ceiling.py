#!/usr/bin/env python
"""Evaluate the exact radial Yukawa law on selected frozen mesh cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

import p2_mesh as p2
import radial_surface_law as radial
import radial_surface_mesh as mesh_eval
import surface_varkernel as vk


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ExactRadialLaw(torch.nn.Module):
    def forward(self, radius, kappa):
        return radial.exact_q(float(kappa) * radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-root", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = mesh_eval.resolve_mesh(args.surface_root, args.mesh)
    vertices, faces = p2.load_obj(path)
    watertight, genus = p2.check_mesh(vertices, faces)
    if not watertight:
        raise RuntimeError(f"frozen valid mesh is no longer watertight: {args.mesh}")

    print(
        f"device={device} mesh={args.mesh} V={len(vertices)} "
        f"seeds={args.eval_seeds}",
        flush=True,
    )
    base = p2.mesh_case(
        vertices, faces, args.kappa, 900 + args.eval_seeds[0], device,
    )
    if base["Xq"].shape[0] < 20:
        raise RuntimeError(
            f"frozen valid mesh has only {base['Xq'].shape[0]} probes: {args.mesh}"
        )

    model = ExactRadialLaw().to(device)
    rows = []
    for eval_seed in args.eval_seeds:
        case = mesh_eval.loads_for_seed(base, 900 + eval_seed, args.kappa)
        identity = torch.eye(
            len(vertices), dtype=case["Sbase"].dtype, device=device,
        )
        with torch.no_grad():
            density0 = torch.linalg.solve(
                case["Sbase"] + 1e-12 * identity, case["g0"],
            )
            known = float(vk.rel(case["Pbase"] @ density0, case["uq0"]))
            floor = float(vk.rel(p2.solve_eval_mesh(case, None), case["uq"]))
            prediction, residual = mesh_eval.radial_solve(
                case, model, args.kappa,
            )
            error = float(vk.rel(prediction, case["uq"]))
        rows.append({
            "eval_seed": eval_seed,
            "mesh": args.mesh,
            "vertices": len(vertices),
            "genus": int(genus),
            "n_probe": int(case["Xq"].shape[0]),
            "known_kernel": known,
            "floor": floor,
            "exact_radial_split": error,
            "gap_closed_pct": 100 * (1 - error / floor),
            "solve_residual": residual,
        })
        print(
            f"seed={eval_seed} known={known:.3e} floor={floor:.3e} "
            f"exact={error:.3e} closure={rows[-1]['gap_closed_pct']:.2f}%",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({
            "mesh": args.mesh,
            "mesh_path": path,
            "kappa": args.kappa,
            "rows": rows,
            "source_hashes": {
                "radial_surface_mesh_ceiling.py": _sha256(__file__),
                "radial_surface_mesh.py": _sha256(mesh_eval.__file__),
                "radial_surface_law.py": _sha256(radial.__file__),
                "p2_mesh.py": _sha256(p2.__file__),
            },
        }, handle, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
