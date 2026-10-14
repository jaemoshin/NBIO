#!/usr/bin/env python
"""Evaluate family-trained radial mixtures on the frozen valid mesh subset."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os

import numpy as np
import torch

import p2_mesh as p2
import panel_qbx as pq
import radial_surface_law as radial
import surface_varkernel as vk

PI = math.pi


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_mesh(root, name):
    for subdir in ("decimated", "thingi10k", "engineering", "data"):
        path = os.path.join(root, "meshes", subdir, f"{name}.obj")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(name)


def load_mixture(path, device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint: {path}")
    if payload.get("method") != "radial_mixture":
        raise ValueError(f"not a radial_mixture checkpoint: {path}")
    model = radial.RadialMixture().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def loads_for_seed(case, seed, kappa):
    rng = np.random.default_rng(seed)
    centers = pq._fib(8) * rng.uniform(2.5, 3.5, (8, 1))
    charges = rng.normal(size=8)
    charges -= charges.mean()
    vertices = case["Xc"].detach().cpu().numpy()
    queries = case["Xq"].detach().cpu().numpy()
    updated = dict(case)
    updated["g"] = torch.tensor(
        vk.yukawa_gt(vertices, centers, charges, kappa),
        device=case["Xc"].device,
    )
    updated["uq"] = torch.tensor(
        vk.yukawa_gt(queries, centers, charges, kappa),
        device=case["Xc"].device,
    )
    updated["g0"] = torch.tensor(
        pq.u_harm(vertices, centers, charges),
        device=case["Xc"].device,
    )
    updated["uq0"] = torch.tensor(
        pq.u_harm(queries, centers, charges),
        device=case["Xc"].device,
    )
    return updated


def radial_solve(case, model, kappa, chunk=256):
    matrix = case["Sbase"].clone()
    potential = case["Pbase"].clone()
    sources = case["Xc"]
    weights = case["wc"]
    for i0 in range(0, sources.shape[0], chunk):
        radius = torch.cdist(sources[i0:i0 + chunk], sources)
        q = model(radius, kappa)
        matrix[i0:i0 + chunk] += (
            float(kappa) * q / (4 * PI) * weights[None, :]
        )
    queries = case["Xq"]
    for i0 in range(0, queries.shape[0], chunk):
        radius = torch.cdist(queries[i0:i0 + chunk], sources)
        q = model(radius, kappa)
        potential[i0:i0 + chunk] += (
            float(kappa) * q / (4 * PI) * weights[None, :]
        )
    identity = torch.eye(
        matrix.shape[0], dtype=matrix.dtype, device=matrix.device,
    )
    matrix = matrix + 1e-12 * identity
    density = torch.linalg.solve(matrix, case["g"])
    prediction = potential @ density
    residual = torch.linalg.vector_norm(matrix @ density - case["g"])
    residual = residual / torch.linalg.vector_norm(case["g"]).clamp_min(1e-30)
    return prediction, float(residual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-root", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_mesh(args.surface_root, args.mesh)
    vertices, faces = p2.load_obj(path)
    watertight, genus = p2.check_mesh(vertices, faces)
    if not watertight:
        raise RuntimeError(f"frozen valid mesh is no longer watertight: {args.mesh}")

    print(
        f"device={device} mesh={args.mesh} V={len(vertices)} seeds={args.eval_seeds}",
        flush=True,
    )
    base = p2.mesh_case(
        vertices, faces, args.kappa, 900 + args.eval_seeds[0], device,
    )
    if base["Xq"].shape[0] < 20:
        raise RuntimeError(
            f"frozen valid mesh has only {base['Xq'].shape[0]} probes: {args.mesh}"
        )

    rows = []
    for eval_seed in args.eval_seeds:
        training_seed = 711 + eval_seed
        checkpoint = os.path.join(
            args.checkpoint_dir,
            f"family_radial_mixture_seed{training_seed}.pt",
        )
        model, payload = load_mixture(checkpoint, device)
        case = loads_for_seed(base, 900 + eval_seed, args.kappa)
        identity = torch.eye(
            len(vertices), dtype=case["Sbase"].dtype, device=device,
        )
        with torch.no_grad():
            density0 = torch.linalg.solve(
                case["Sbase"] + 1e-12 * identity, case["g0"],
            )
            known = float(vk.rel(case["Pbase"] @ density0, case["uq0"]))
            floor = float(vk.rel(p2.solve_eval_mesh(case, None), case["uq"]))
            prediction, residual = radial_solve(
                case, model, args.kappa,
            )
            error = float(vk.rel(prediction, case["uq"]))
        rows.append({
            "eval_seed": eval_seed,
            "training_seed": training_seed,
            "mesh": args.mesh,
            "vertices": len(vertices),
            "genus": int(genus),
            "n_probe": int(case["Xq"].shape[0]),
            "tau_h": case["tau_h"],
            "known_kernel": known,
            "floor": floor,
            "radial_mixture": error,
            "gap_closed_pct": 100 * (1 - error / floor),
            "solve_residual": residual,
            "checkpoint": os.path.abspath(checkpoint),
            "checkpoint_source_sha256": payload["source_sha256"],
        })
        print(
            f"seed={eval_seed} known={known:.3e} floor={floor:.3e} "
            f"mixture={error:.3e} closure={rows[-1]['gap_closed_pct']:.2f}%",
            flush=True,
        )
        del model, case, identity
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "mesh": args.mesh,
            "mesh_path": path,
            "kappa": args.kappa,
            "rows": rows,
            "source_hashes": {
                "radial_surface_mesh.py": _sha256(__file__),
                "radial_surface_law.py": _sha256(radial.__file__),
                "p2_mesh.py": _sha256(p2.__file__),
                "panel_qbx.py": _sha256(pq.__file__),
            },
        }, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
