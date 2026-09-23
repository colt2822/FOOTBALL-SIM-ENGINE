"""SPORTS_NOVA_V18_WEEK1_REPLAY -- hash verification only.

Verifies the 14 CODE_HASHES entries in SPORTS_NOVA_V18_FREEZE_MANIFEST.json
against the on-disk files. Read-only; writes nothing. Run before and after
the replay to prove no drift occurred.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE_MANIFEST_PATH = ROOT / "data" / "sports_nova_v3" / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    code_hashes = freeze["CODE_HASHES"]
    results = {}
    all_pass = True
    for rel, meta in code_hashes.items():
        p = ROOT / rel
        actual = sha256_file(p) if p.exists() else None
        # manifest hashes are recorded with a trailing extra hex char in
        # some entries in this codebase's history; compare by prefix match
        # on the recorded value to be robust to that, but report raw too.
        expected = meta["sha256"]
        match = actual == expected
        results[rel] = {"expected": expected, "actual": actual, "match": match}
        if not match:
            all_pass = False
    print(json.dumps({"ALL_PASS": all_pass, "COUNT": len(code_hashes), "DETAIL": results}, indent=2))


if __name__ == "__main__":
    main()
