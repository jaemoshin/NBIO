#!/usr/bin/env python
"""CPU smoke and invariance checks for the localized-singularity NBIO path."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import BIEDataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.layers import double_layer_kernel  # noqa: E402
from src.models.nbio_local_singular import local_prior_weight  # noqa: E402


def main():
    torch.manual_seed(7)
    dataset = BIEDataset(ROOT / "data_cache/yukawa_val.pt")
    batch = {
        key: torch.stack([dataset[i][key] for i in range(2)]).float()
        for key in dataset.keys
    }
    cfg = {
        "model": {
            "name": "nbio_local",
            "kwargs": {
                "hidden": 16,
                "n_layers": 2,
                "prior_cutoff": 0.2,
                "prior_power": 4.0,
                "prior_scope": "potential",
            },
        }
    }
    model = build_model(cfg)
    prediction = model(batch)
    assert prediction.shape == batch["uq"].shape
    assert torch.isfinite(prediction).all()
    loss = torch.mean((prediction - batch["uq"]) ** 2)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    # Potential-only mode must leave assembly exactly equal to the learned kernel system.
    matrix = model.bio.assemble(
        batch["Xb"], batch["Nb"], batch["wb"], batch["curv"], model.jump
    )
    learned = model.bio.smooth(batch["Xb"], batch["Xb"], batch["Nb"], batch["Nb"])
    n = batch["Xb"].shape[1]
    expected = model.jump * torch.eye(n).unsqueeze(0) + learned * batch["wb"][:, None, :]
    torch.testing.assert_close(matrix, expected)

    # The potential increment is exactly the localized analytic layer.
    potential = model.bio.potential(batch["Xq"], batch["Xb"], batch["Nb"], batch["wb"])
    Nq = torch.zeros_like(batch["Xq"])
    learned_potential = model.bio.smooth(batch["Xq"], batch["Xb"], Nq, batch["Nb"])
    known = double_layer_kernel(batch["Xq"], batch["Xb"], batch["Nb"])
    weight = local_prior_weight(batch["Xq"], batch["Xb"], 0.2, 4.0)
    expected_increment = weight * known * batch["wb"][:, None, :]
    torch.testing.assert_close(potential - learned_potential * batch["wb"][:, None, :], expected_increment)

    close = torch.tensor([[[0.0, 0.0]]])
    source = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    weights = local_prior_weight(close, source, 0.2, 4.0)
    assert weights[0, 0, 0] == 1.0
    assert weights[0, 0, 1] < 1e-10
    model.qbx = True
    with torch.no_grad():
        qbx_prediction = model(batch)
    assert qbx_prediction.shape == batch["uq"].shape
    assert torch.isfinite(qbx_prediction).all()
    print(f"localized singular CPU smoke passed; loss={loss.item():.3e}")


if __name__ == "__main__":
    main()
