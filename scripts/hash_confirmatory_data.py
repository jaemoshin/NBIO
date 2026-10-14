#!/usr/bin/env python
"""OBSOLETE pre-lock draft; retained only as provenance.

Use seal_screening_confirmatory_data.py from the locked protocol.
"""
raise SystemExit(
    "obsolete pre-lock draft; use scripts/seal_screening_confirmatory_data.py"
)
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cache/screening_confirmatory_v1"
h = {}
for p in sorted(DATA.rglob("*")):
    if p.is_file():
        h[str(p.relative_to(DATA))] = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / "results/screening_law/confirmatory_v1/data_hashes.json").write_text(json.dumps(h, indent=2))
print(f"hashed {len(h)} files")
