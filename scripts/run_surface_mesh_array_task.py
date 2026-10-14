#!/usr/bin/env python3
"""Run one frozen seed/mesh cell from a Slurm array."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from aggregate_surface_mesh_verification import MESHES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            parser.error("--task-id or SLURM_ARRAY_TASK_ID is required")
        task_id = int(value)
    if not 0 <= task_id < len(MESHES):
        raise ValueError(f"task id {task_id} outside [0, {len(MESHES) - 1}]")

    root = args.surface_root.resolve()
    checkpoint = args.checkpoint.resolve()
    mesh = MESHES[task_id]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    output = args.raw_dir / f"seed_{args.seed}_{mesh}.json"
    command_record = args.raw_dir / f"seed_{args.seed}_{mesh}.command.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    command = [
        sys.executable,
        "p2_mesh.py",
        "--meshes",
        mesh,
        "--seed",
        str(args.seed),
        "--kappa",
        "3.0",
        "--load-correction",
        str(checkpoint),
        "--out",
        str(output.resolve()),
    ]
    record = {
        "seed": args.seed,
        "task_id": task_id,
        "mesh": mesh,
        "checkpoint": str(checkpoint),
        "surface_root": str(root),
        "command": command,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    command_record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
