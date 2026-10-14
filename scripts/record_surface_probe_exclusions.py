#!/usr/bin/env python3
"""Record deterministic no-probe exclusions without running dense surface solves."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface_root", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--meshes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--kappa", type=float, default=3.0)
    return parser.parse_args()


def find_mesh(root: Path, name: str) -> Path:
    for subdir in ("decimated", "thingi10k", "engineering", "data"):
        path = root / "meshes" / subdir / f"{name}.obj"
        if path.exists():
            return path
    raise FileNotFoundError(name)


def main() -> None:
    args = parse_args()
    surface_root = args.surface_root.resolve()
    sys.path.insert(0, str(surface_root))

    import p2_mesh as p2
    import panel_qbx as pq

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    exclusions = {}
    for mesh in args.meshes:
        vertices, faces = p2.load_obj(find_mesh(surface_root, mesh))
        watertight, genus = p2.check_mesh(vertices, faces)
        if not watertight:
            raise RuntimeError(f"{mesh} is not watertight")
        face_area, _, vertex_normals = pq.mesh_geom(vertices, faces)
        panel_h = math.sqrt(float(face_area.mean()))
        thickness = pq.local_thickness(vertices, vertex_normals, panel_h) / panel_h
        probes = p2.interior_probes(
            vertices, faces, vertex_normals, panel_h
        )
        if len(probes) != 0:
            raise RuntimeError(
                f"{mesh} has {len(probes)} accepted probes; a dense evaluation is required"
            )
        mesh_row = {
            "V": len(vertices),
            "genus": int(genus),
            "n_probe": 0,
            "tau_h": {
                "min": float(np.min(thickness)),
                "p5": float(np.percentile(thickness, 5)),
                "med": float(np.median(thickness)),
            },
            "not_evaluated_reason": "zero_accepted_interior_probes",
        }
        exclusions[mesh] = mesh_row
        for seed in args.seeds:
            out = args.raw_dir / f"seed_{seed}_{mesh}.json"
            if out.exists():
                continue
            checkpoint = (
                surface_root
                / "results"
                / "tmlr_surface_verify_20260725"
                / "checkpoints"
                / f"corr_seed{seed}.pt"
            )
            payload = {
                "seed": seed,
                "kappa": args.kappa,
                "correction_checkpoint": str(checkpoint.resolve()),
                "source_hashes": p2.source_hashes(),
                mesh: mesh_row,
            }
            out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(exclusions, indent=2))


if __name__ == "__main__":
    main()
