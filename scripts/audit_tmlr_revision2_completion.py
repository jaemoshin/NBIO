#!/usr/bin/env python
"""Fail-closed audit for the second TMLR reviewer-revision tranche."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "tmlr" / "nbio.tex"
PDF = ROOT / "paper" / "tmlr" / "nbio.pdf"
BIB = ROOT / "paper" / "tmlr" / "references.bib"
SUPPLEMENT = ROOT / "paper" / "tmlr" / "nbio_anonymous_supplement.tar.gz"
MAIN_RESPONSE = ROOT / "paper" / "tmlr" / "reviewer_response.md"
REVIEWER1_RESPONSE = ROOT / "paper" / "tmlr" / "reviewer_1PjB_response.md"
REVIEWER3_RESPONSE = ROOT / "paper" / "tmlr" / "reviewer_yiXB_response.md"
TRANSFER = ROOT / "results" / "tmlr_revision2_smooth_ood"
USE = ROOT / "results" / "tmlr_revision2_smooth"
SMOOTH = ROOT / "results" / "kappa_smooth_features"
OUT = ROOT / "results" / "tmlr_revision2" / "completion_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    tex = TEX.read_text()
    bib = BIB.read_text()
    main_response = MAIN_RESPONSE.read_text()
    reviewer1_response = REVIEWER1_RESPONSE.read_text()
    reviewer3_response = REVIEWER3_RESPONSE.read_text()
    response_package = "\n".join(
        (main_response, reviewer1_response, reviewer3_response)
    )
    checks: dict[str, bool] = {}

    markers = {
        "structure_learning_distinction": "Solver-grade accuracy on exact-kernel rows is a property of the",
        "conditioned_family": "For the conditioned family we use the regularity-controlled feature map",
        "smooth_features": "regularity-controlled feature map",
        "fixed_kappa_boundary": "fixed-mismatch baseline",
        "endpoint_fairness": "including its self-interaction, without a $\\kappa$ envelope",
        "out_of_range_boundary": "does not transfer reliably from the",
        "inverse_use": "shape inverse problem",
        "cached_runtime": "share nearly identical cached repeated-right-hand-side cost",
        "theory_regularized_features": "raw displacement, squared distance",
        "no_universal_prior_claim": "NBIO is not justified as a more accurate forward solver",
    }
    for name, marker in markers.items():
        checks[name] = marker in tex

    cited: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]*)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    checks["all_citations_resolve"] = cited == bib_keys and len(cited) > 0
    checks["response_files_present"] = all(
        path.is_file()
        for path in (MAIN_RESPONSE, REVIEWER1_RESPONSE, REVIEWER3_RESPONSE)
    )
    checks["response_current_runtime"] = (
        "`2.049`, `2.816`, and `4.159` ms/sample" in response_package
    )
    checks["response_current_efficiency"] = all(
        token in response_package
        for token in ("`1.54e-4/8.92e-3`", "`1.19e-3/1.43e-2`")
    )
    checks["response_family_stability"] = (
        "`0.471/0.758` for deformed shapes" in response_package
    )
    checks["response_stale_values_absent"] = all(
        token not in response_package
        for token in (
            "`[0.0313,0.1022]`",
            "`3.30` and `2.35`",
            "`1.23e-4/1.02e-2`",
            "`1.20e-3/1.65e-2`",
            "`6.54/10.80/3.88`",
        )
    )

    checks["transfer_raw_cardinality"] = csv_rows(TRANSFER / "conditioned_transfer_raw.csv") == 46_080
    checks["transfer_summary_cardinality"] = csv_rows(TRANSFER / "conditioned_transfer_summary.csv") == 960
    transfer_audit = json.loads((TRANSFER / "conditioned_transfer_audit.json").read_text())
    checks["transfer_integrity"] = (
        transfer_audit["integrity"]["actual_raw_rows"] == 46_080
        and transfer_audit["integrity"]["unique_keys"] == 46_080
        and transfer_audit["integrity"]["nonfinite_rel_l2"] == 0
        and transfer_audit["integrity"]["pass"]
    )

    inverse = json.loads((USE / "inverse_demo_summary.json").read_text())
    checks["inverse_all_starts_pass"] = all(
        row["n_starts"] == 8
        and row["fraction_coefficient_rmse_below_3e-2"] == 1.0
        and row["fraction_forward_below_1e-2"] == 1.0
        for row in inverse["summary"].values()
    )
    checks["runtime_contract_rows"] = csv_rows(USE / "runtime_amortization.csv") == 81
    smooth = json.loads((SMOOTH / "audit.json").read_text())
    checks["smooth_features_retained_by_frozen_rule"] = bool(smooth["promote_smooth_features"])

    reader = PdfReader(str(PDF))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # Split sentinel strings so this audit can itself be included in the anonymous archive.
    forbidden = (
        "/" + "gallery",
        "jaemo" + ".shin",
        "snu" + ".vision",
        "@" + "gmail",
        "@" + "snu",
    )
    checks["pdf_anonymous"] = (
        "Anonymous authors" in pdf_text
        and not any(token in pdf_text for token in forbidden)
    )
    checks["pdf_has_conditioned_evidence"] = (
        "Operational algorithm" in pdf_text
        and "Setting γ= 0" in pdf_text
        and "classical Nystr" in pdf_text
        and "shape inverse" in pdf_text
    )

    required_members = {
        "nbio_anonymous_supplement/src/models/nbio_kappa_conditioned.py",
        "nbio_anonymous_supplement/results/tmlr_revision2_smooth_ood/conditioned_transfer_raw.csv",
        "nbio_anonymous_supplement/results/tmlr_revision2_smooth/inverse_demo_raw.csv",
        "nbio_anonymous_supplement/results/tmlr_revision2_smooth/runtime_amortization.csv",
        "nbio_anonymous_supplement/results/kappa_smooth_features/audit.json",
    }
    with tarfile.open(SUPPLEMENT, "r:gz") as archive:
        owners = {(member.uid, member.gid, member.uname, member.gname) for member in archive.getmembers()}
        names = set(archive.getnames())
        leaked = False
        for member in archive.getmembers():
            if not member.isfile():
                continue
            data = archive.extractfile(member).read()
            sentinels = (b"/" + b"gallery", b"jaemo" + b".shin", b"snu" + b".vision")
            if any(token in data for token in sentinels):
                leaked = True
                break
    checks["supplement_metadata_anonymous"] = owners <= {
        (0, 0, "", ""), (0, 0, "root", "root")
    }
    checks["supplement_content_anonymous"] = not leaked
    checks["supplement_has_revision2_evidence"] = required_members <= names

    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed": failed,
        "pdf_pages": len(reader.pages),
        "cited_entries": len(cited),
        "pdf_sha256": sha256(PDF),
        "supplement_sha256": sha256(SUPPLEMENT),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
