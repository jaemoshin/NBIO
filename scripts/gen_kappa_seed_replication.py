#!/usr/bin/env python
"""Generate three fresh continuous-kappa train/validation datasets for seed robustness."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_kappa_continuous import generated_split  # noqa: E402


def main():
    root = ROOT / "data_cache" / "kappa_seed_replication"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"train_n": 2400, "val_n": 400, "datasets": []}
    for data_seed in range(3):
        out = root / f"data_s{data_seed}"
        out.mkdir(parents=True, exist_ok=True)
        train_seed = 12_001 + data_seed
        val_seed = 13_001 + data_seed
        torch.save(generated_split(2400, train_seed), out / "yukawa_train.pt")
        torch.save(generated_split(400, val_seed), out / "yukawa_val.pt")
        shutil.copy2(
            ROOT / "data_cache/kappa_continuous/yukawa_test.pt",
            out / "yukawa_test.pt",
        )
        manifest["datasets"].append({
            "data_seed": data_seed, "train_seed": train_seed, "val_seed": val_seed,
            "model_seed": 100 + data_seed, "path": str(out.relative_to(ROOT)),
        })
        print(f"generated data seed {data_seed}", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
