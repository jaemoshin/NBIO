#!/usr/bin/env python
"""Same-dataset / same-geometry checks against nearby boundary/interface papers.

This script is intentionally conservative:

1. Exact shared dataset: Learning Only on Boundaries 2D parametric Laplace geometry.
   NBIO can solve this directly because it is a Dirichlet Laplace BVP on a changing boundary.

2. Same-geometry subset: XI-DeepONet circular high-contrast interface geometry.
   The published XI/IONet examples include source/jump terms that current NBIO-tm does not ingest.
   We therefore test the same circular interface radii and contrast as a homogeneous transmission subset,
   and label it as non-identical to the published dataset.

3. Feasibility matrix: which external protocols are exact, subset-only, or excluded.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.bie import boundary, bem_solve, sample_queries
from src.data.transmission import analytic_disk, build_curves, transmission_solve
from src.models import build_model
from src.models.nbio_transmission import NBIOTransmission
from src.numerics.qbx_torch import double_layer_qbx


ROOT = Path(".")
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
REPORT_DIR = ROOT / "reports"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def rel(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-12))


def torch_rel(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (torch.linalg.vector_norm(pred - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12))
        .mean()
        .item()
    )


def load_checkpoint_model(run_dir: str):
    ckpt = Path(run_dir) / "best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def to_tensor_batch(d: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v[None], dtype=torch.float32) for k, v in d.items()}


def lob_coeffs(t: float) -> dict:
    """Learning-Only-on-Boundaries 2D Laplace geometry:
    r(alpha;t)=1+0.2(sin(3a)+t sin(4a)+sin(6a)+cos(2a)+cos(5a)).
    """
    a = np.zeros(6, dtype=np.float64)
    b = np.zeros(6, dtype=np.float64)
    a[1] = 0.2
    a[4] = 0.2
    b[2] = 0.2
    b[3] = 0.2 * t
    b[5] = 0.2
    return {"r0": 1.0, "a": a, "b": b}


def harmonic_lob_u(x: np.ndarray) -> np.ndarray:
    """u0(x,y)=exp(x) sin(y), the harmonic manufactured solution used in the LOB Laplace example."""
    return np.exp(x[:, 0]) * np.sin(x[:, 1])


def make_lob_sample(t: float, n_b: int, n_q: int = 2048, seed: int = 123) -> tuple[dict, np.ndarray]:
    coeffs = lob_coeffs(t)
    bd = boundary(coeffs, n_b)
    rng = np.random.default_rng(seed + int(round(1000 * t)) + n_b)
    xq = sample_queries(rng, coeffs, "interior", n_q, margin=0.04)
    g = harmonic_lob_u(bd["X"])
    uq = harmonic_lob_u(xq)
    batch = {
        "Xb": bd["X"].astype(np.float32),
        "Nb": bd["normal"].astype(np.float32),
        "wb": bd["w"].astype(np.float32),
        "curv": bd["curv"].astype(np.float32),
        "g": g.astype(np.float32),
        "Xq": xq.astype(np.float32),
        "uq": uq.astype(np.float32),
    }
    return batch, uq


def qbx_bem_eval(sample: dict[str, np.ndarray], mu: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            double_layer_qbx(
                torch.tensor(sample["Xq"][None], dtype=torch.float32),
                torch.tensor(sample["Xb"][None], dtype=torch.float32),
                torch.tensor(mu[None], dtype=torch.float32),
                Nb=torch.tensor(sample["Nb"][None], dtype=torch.float32),
                wb=torch.tensor(sample["wb"][None], dtype=torch.float32),
                order=12,
                r_fac=4.0,
                near_fac=2.0,
            )
            .cpu()
            .numpy()[0]
        )


def evaluate_lob_laplace() -> list[dict]:
    published = {1.15: 0.0285, 1.35: 0.0287, 1.45: 0.0300}
    model_specs = [
        ("NBIO_trained", "results/nbio_s0"),
        ("BoundaryTransformer", "results/btransformer_s0"),
        ("DeepONet", "results/deeponet_s0"),
        ("DeepSet", "results/deepset_s0"),
    ]
    models = {}
    for name, path in model_specs:
        try:
            models[name] = load_checkpoint_model(path)
        except FileNotFoundError:
            pass

    rows: list[dict] = []
    for n_b in (128, 256, 512, 1024):
        for t in (1.15, 1.35, 1.45):
            sample, uq = make_lob_sample(t, n_b=n_b)
            bd = {k: sample[k] for k in ("Xb", "Nb", "wb", "curv")}
            bd_np = {"X": bd["Xb"], "normal": bd["Nb"], "w": bd["wb"], "curv": bd["curv"]}
            mu, bem_eval = bem_solve(bd_np, sample["g"], region="interior")
            bem_pred = bem_eval(sample["Xq"])
            bem_qbx_pred = qbx_bem_eval(sample, mu)
            rows.append(
                {
                    "protocol": "LOB_2d_parametric_laplace_exact",
                    "method": "Published_LOB_PINO_BIE",
                    "t": t,
                    "n_boundary": n_b,
                    "n_query": len(uq),
                    "rel_l2": published[t],
                    "published_rel_l2": published[t],
                    "comparability": "exact_same_geometry_pde",
                }
            )
            rows.append(
                {
                    "protocol": "LOB_2d_parametric_laplace_exact",
                    "method": "Classical_BEM_NBIO_structure",
                    "t": t,
                    "n_boundary": n_b,
                    "n_query": len(uq),
                    "rel_l2": rel(bem_pred, uq),
                    "published_rel_l2": published[t],
                    "comparability": "exact_same_geometry_pde",
                }
            )
            rows.append(
                {
                    "protocol": "LOB_2d_parametric_laplace_exact",
                    "method": "Classical_BEM_NBIO_structure_QBX",
                    "t": t,
                    "n_boundary": n_b,
                    "n_query": len(uq),
                    "rel_l2": rel(bem_qbx_pred, uq),
                    "published_rel_l2": published[t],
                    "comparability": "exact_same_geometry_pde",
                }
            )
            tb = to_tensor_batch(sample)
            target = tb["uq"]
            with torch.no_grad():
                for name, model in models.items():
                    pred = model(tb)
                    rows.append(
                        {
                            "protocol": "LOB_2d_parametric_laplace_exact",
                            "method": name,
                            "t": t,
                            "n_boundary": n_b,
                            "n_query": len(uq),
                            "rel_l2": torch_rel(pred, target),
                            "published_rel_l2": published[t],
                            "comparability": "exact_same_geometry_pde",
                        }
                    )
                    if name == "NBIO_trained" and hasattr(model, "qbx"):
                        old_qbx = model.qbx
                        old_kw = dict(model.qbx_kw)
                        model.qbx = True
                        model.qbx_kw = {"order": 12, "r_fac": 4.0, "near_fac": 2.0}
                        pred_qbx = model(tb)
                        model.qbx = old_qbx
                        model.qbx_kw = old_kw
                        rows.append(
                            {
                                "protocol": "LOB_2d_parametric_laplace_exact",
                                "method": "NBIO_trained_QBX",
                                "t": t,
                                "n_boundary": n_b,
                                "n_query": len(uq),
                                "rel_l2": torch_rel(pred_qbx, target),
                                "published_rel_l2": published[t],
                                "comparability": "exact_same_geometry_pde",
                            }
                        )
    return rows


def circle_coeff(radius: float) -> dict:
    return {"r0": float(radius), "a": np.zeros(1), "b": np.zeros(1)}


def disk_queries(rng: np.random.Generator, r_interface: float, n_q: int, margin: float = 0.035) -> np.ndarray:
    pts = []
    while len(pts) < n_q:
        ang = rng.uniform(0, 2 * np.pi, 4 * n_q)
        rad = np.sqrt(rng.uniform(0, 1, 4 * n_q)) * 0.93
        cand = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        rr = np.linalg.norm(cand, axis=1)
        cand = cand[np.abs(rr - r_interface) > margin]
        pts.extend(cand.tolist())
    return np.array(pts[:n_q], dtype=np.float64)


def make_xi_circle_subset(r0: float, contrast: float, n_b_out: int = 192, n_b_in: int = 96, seed: int = 77):
    """Same circular radii and high contrast as XI-DeepONet Example 2, but homogeneous transmission.

    Published XI uses source/jump data not represented by current NBIO-tm inputs. This subset uses the
    same geometry/contrast with outer Dirichlet g=x and zero interface jumps, for which analytic_disk is exact.
    """
    outer = circle_coeff(1.0)
    incls = [(np.zeros(2), r0, circle_coeff(r0))]
    curves = build_curves(outer, incls, n_b_out, n_b_in)
    g = curves[0]["X"][:, 0]
    xq = disk_queries(np.random.default_rng(seed + int(1000 * r0)), r0, 2048)
    uq = analytic_disk(1.0, r0, 1.0, contrast, xq)
    _, bem_eval = transmission_solve(curves, [1.0, contrast], g)
    sample = {
        "Xb_out": curves[0]["X"].astype(np.float32),
        "Nb_out": curves[0]["normal"].astype(np.float32),
        "wb_out": curves[0]["w"].astype(np.float32),
        "curv_out": curves[0]["curv"].astype(np.float32),
        "g": g.astype(np.float32),
        "Xb_in": np.stack([curves[1]["X"]]).astype(np.float32),
        "Nb_in": np.stack([curves[1]["normal"]]).astype(np.float32),
        "wb_in": np.stack([curves[1]["w"]]).astype(np.float32),
        "curv_in": np.stack([curves[1]["curv"]]).astype(np.float32),
        "a_in": np.array([contrast], dtype=np.float32),
        "Xq": xq.astype(np.float32),
        "uq": uq.astype(np.float32),
    }
    return sample, uq, bem_eval(xq)


def evaluate_xi_circle_subset() -> list[dict]:
    published_xi = {0.5: 5.08e-3, 0.6: 1.48e-3, 0.7: 1.30e-3}
    models = {
        "NBIO_tm_structure": NBIOTransmission(split=False).eval(),
    }
    for name, path in [
        ("NBIO_tm_trained", "results/tm_nbio_s0"),
        ("TransmissionTransformer", "results/tm_transformer_s0"),
        ("DeepSet_tm", "results/tm_deepset_s0"),
    ]:
        try:
            models[name] = load_checkpoint_model(path)
        except FileNotFoundError:
            pass

    rows = []
    for r0 in (0.5, 0.6, 0.7):
        sample, uq, bem_pred = make_xi_circle_subset(r0, contrast=1000.0)
        rows.append(
            {
                "protocol": "XI_circle_high_contrast_homogeneous_subset",
                "method": "Published_XI_DeepONet_not_same_forcing",
                "interface_radius": r0,
                "contrast": 1000.0,
                "n_boundary_outer": 192,
                "n_boundary_inner": 96,
                "n_query": len(uq),
                "rel_l2": published_xi[r0],
                "published_rel_l2": published_xi[r0],
                "comparability": "same_geometry_contrast_not_same_source_or_jump",
            }
        )
        rows.append(
            {
                "protocol": "XI_circle_high_contrast_homogeneous_subset",
                "method": "Classical_BEM_NBIO_structure",
                "interface_radius": r0,
                "contrast": 1000.0,
                "n_boundary_outer": 192,
                "n_boundary_inner": 96,
                "n_query": len(uq),
                "rel_l2": rel(bem_pred, uq),
                "published_rel_l2": published_xi[r0],
                "comparability": "same_geometry_contrast_not_same_source_or_jump",
            }
        )
        tb = to_tensor_batch(sample)
        target = tb["uq"]
        with torch.no_grad():
            for name, model in models.items():
                try:
                    pred = model(tb)
                    err = torch_rel(pred, target)
                    note = "same_geometry_contrast_not_same_source_or_jump"
                except Exception:
                    err = float("nan")
                    note = "model_failed_on_single_inclusion_or_extreme_contrast"
                rows.append(
                    {
                        "protocol": "XI_circle_high_contrast_homogeneous_subset",
                        "method": name,
                        "interface_radius": r0,
                        "contrast": 1000.0,
                        "n_boundary_outer": 192,
                        "n_boundary_inner": 96,
                        "n_query": len(uq),
                        "rel_l2": err,
                        "published_rel_l2": published_xi[r0],
                        "comparability": note,
                    }
                )
    return rows


def feasibility_rows() -> list[dict]:
    return [
        {
            "paper": "Learning Only on Boundaries",
            "protocol": "2D parametric Laplace",
            "status": "tested_exact",
            "reason": "Dirichlet Laplace on changing boundary matches NBIO inputs.",
            "output_table": "external_same_dataset_lob_laplace.csv",
        },
        {
            "paper": "Learning Only on Boundaries",
            "protocol": "2D biharmonic, 3D Helmholtz scattering",
            "status": "excluded",
            "reason": "Current NBIO kernels are Laplace/Yukawa, not biharmonic or Helmholtz scattering.",
            "output_table": "",
        },
        {
            "paper": "IONet",
            "protocol": "parametric elliptic interface with source/boundary/interface inputs",
            "status": "excluded_exact",
            "reason": "Published inputs include volume source and interface jump conditions; current NBIO-tm ingests boundary Dirichlet plus conductivity only.",
            "output_table": "",
        },
        {
            "paper": "XI-DeepONet",
            "protocol": "circular high-contrast interface",
            "status": "tested_subset",
            "reason": "Same circle radii and contrast, but homogeneous transmission subset because published source/jump data are outside current NBIO-tm input contract.",
            "output_table": "external_same_dataset_xi_circle_subset.csv",
        },
        {
            "paper": "BENO",
            "protocol": "Poisson/Darcy corner domains with volume source and inhomogeneous boundary values",
            "status": "excluded_exact",
            "reason": "Requires volume-source Poisson/Darcy operator on corner domains; current NBIO is boundary-only homogeneous Laplace/transmission.",
            "output_table": "",
        },
        {
            "paper": "Learned function extensions",
            "protocol": "Poisson, elasticity, hyperelasticity with complex BCs",
            "status": "excluded",
            "reason": "Mostly boundary-condition encoding for volume neural operators and elasticity; not a current NBIO target.",
            "output_table": "",
        },
    ]


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_lob(rows: list[dict]) -> None:
    methods = [
        "Published_LOB_PINO_BIE",
        "Classical_BEM_NBIO_structure",
        "Classical_BEM_NBIO_structure_QBX",
        "NBIO_trained",
        "NBIO_trained_QBX",
        "BoundaryTransformer",
        "DeepONet",
        "DeepSet",
    ]
    nbs = sorted({int(r["n_boundary"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in methods:
        xs, ys = [], []
        for n_b in nbs:
            vals = [float(r["rel_l2"]) for r in rows if r["method"] == method and int(r["n_boundary"]) == n_b]
            if vals:
                xs.append(n_b)
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("boundary nodes")
    ax.set_ylabel("mean relative L2 over t")
    ax.set_title("Exact LOB 2D parametric Laplace geometry")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "external_same_dataset_lob_laplace.png", dpi=180)
    plt.close(fig)


def write_report(lob_rows: list[dict], xi_rows: list[dict], feas: list[dict]) -> None:
    def best(rows, protocol, method):
        vals = [float(r["rel_l2"]) for r in rows if r["method"] == method and math.isfinite(float(r["rel_l2"]))]
        return min(vals) if vals else float("nan")

    lob_nbio = best(lob_rows, "LOB_2d_parametric_laplace_exact", "NBIO_trained")
    lob_nbio_qbx = best(lob_rows, "LOB_2d_parametric_laplace_exact", "NBIO_trained_QBX")
    lob_bem = best(lob_rows, "LOB_2d_parametric_laplace_exact", "Classical_BEM_NBIO_structure")
    lob_bem_qbx = best(lob_rows, "LOB_2d_parametric_laplace_exact", "Classical_BEM_NBIO_structure_QBX")
    lob_pub = best(lob_rows, "LOB_2d_parametric_laplace_exact", "Published_LOB_PINO_BIE")
    xi_nbio = best(xi_rows, "XI_circle_high_contrast_homogeneous_subset", "NBIO_tm_structure")
    xi_bem = best(xi_rows, "XI_circle_high_contrast_homogeneous_subset", "Classical_BEM_NBIO_structure")

    lines = [
        "# NBIO same-dataset checks against related papers",
        "",
        "This is the first exact/shared-protocol pass. It separates exact shared datasets from cases where the",
        "published paper uses inputs outside the current NBIO contract.",
        "",
        "## Headline",
        "",
        f"- Exact Learning-Only-on-Boundaries 2D Laplace geometry: published error is about `{lob_pub:.3e}`; "
        f"direct NBIO trained reaches best `{lob_nbio:.3e}`, while QBX close evaluation reaches `{lob_nbio_qbx:.3e}`. "
        f"The classical BIE/NBIO-structure ceiling is `{lob_bem:.3e}` directly and `{lob_bem_qbx:.3e}` with QBX.",
        f"- XI-DeepONet circular high-contrast geometry: only a homogeneous-transmission subset is tested. "
        f"NBIO-tm structure reaches best `{xi_nbio:.3e}` against analytic/BEM reference; BEM ceiling `{xi_bem:.3e}`. "
        "Do not compare this row directly to XI's published source/jump problem.",
        "",
        "## Exact LOB 2D Laplace results",
        "",
        "| method | n_b | t=1.15 | t=1.35 | t=1.45 |",
        "|---|---:|---:|---:|---:|",
    ]
    methods = [
        "Published_LOB_PINO_BIE",
        "Classical_BEM_NBIO_structure",
        "Classical_BEM_NBIO_structure_QBX",
        "NBIO_trained",
        "NBIO_trained_QBX",
        "BoundaryTransformer",
        "DeepONet",
        "DeepSet",
    ]
    for method in methods:
        for n_b in (128, 256, 512, 1024):
            vals = []
            for t in (1.15, 1.35, 1.45):
                row = next((r for r in lob_rows if r["method"] == method and int(r["n_boundary"]) == n_b and abs(float(r["t"]) - t) < 1e-9), None)
                vals.append("nan" if row is None else f"{float(row['rel_l2']):.3e}")
            lines.append(f"| {method} | {n_b} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += [
        "",
        "## XI circular high-contrast subset",
        "",
        "This uses XI-DeepONet's interface radii and contrast `a+=1000, a-=1`, but not the exact published",
        "source/jump dataset. Current NBIO-tm does not ingest volume source or nonzero interface jumps.",
        "",
        "| method | r=0.5 | r=0.6 | r=0.7 | comparability |",
        "|---|---:|---:|---:|---|",
    ]
    methods_xi = ["Published_XI_DeepONet_not_same_forcing", "Classical_BEM_NBIO_structure", "NBIO_tm_structure", "NBIO_tm_trained", "TransmissionTransformer", "DeepSet_tm"]
    for method in methods_xi:
        vals = []
        comp = ""
        for r0 in (0.5, 0.6, 0.7):
            row = next((r for r in xi_rows if r["method"] == method and abs(float(r["interface_radius"]) - r0) < 1e-9), None)
            vals.append("nan" if row is None else f"{float(row['rel_l2']):.3e}")
            if row is not None:
                comp = row["comparability"]
        lines.append(f"| {method} | {vals[0]} | {vals[1]} | {vals[2]} | {comp} |")
    lines += [
        "",
        "## Feasibility matrix",
        "",
        "| paper | protocol | status | reason |",
        "|---|---|---|---|",
    ]
    for row in feas:
        lines.append(f"| {row['paper']} | {row['protocol']} | {row['status']} | {row['reason']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The exact Learning-Only-on-Boundaries check is the cleanest external same-dataset evidence. The direct",
        "Nyström evaluation underresolves the high-curvature external geometry at low boundary resolution, which",
        "is why direct BEM/NBIO is poor at `n_b=128`. With the current QBX close-evaluation path, NBIO recovers",
        "solver-grade accuracy by `n_b=256` on the same PDE and geometry, without retraining on the LOB dataset.",
        "",
        "IONet, XI-DeepONet, and BENO should not be presented as beaten on their full datasets until NBIO is extended",
        "to ingest volume sources, nonzero interface jump data, and more general boundary-condition types. The subset",
        "tests still support the scoped claim: when the problem reduces to known-kernel boundary/interface structure,",
        "NBIO behaves like the classical solver.",
    ]
    (REPORT_DIR / "external_same_dataset_comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    lob_rows = evaluate_lob_laplace()
    xi_rows = evaluate_xi_circle_subset()
    feas = feasibility_rows()
    save_csv(TABLE_DIR / "external_same_dataset_lob_laplace.csv", lob_rows)
    save_csv(TABLE_DIR / "external_same_dataset_xi_circle_subset.csv", xi_rows)
    save_csv(TABLE_DIR / "external_same_dataset_feasibility.csv", feas)
    plot_lob(lob_rows)
    write_report(lob_rows, xi_rows, feas)
    print(TABLE_DIR / "external_same_dataset_lob_laplace.csv")
    print(TABLE_DIR / "external_same_dataset_xi_circle_subset.csv")
    print(REPORT_DIR / "external_same_dataset_comparison.md")


if __name__ == "__main__":
    main()
