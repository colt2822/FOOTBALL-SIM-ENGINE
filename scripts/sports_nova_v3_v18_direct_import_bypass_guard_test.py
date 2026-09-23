"""SPORTS_V18_DIRECT_IMPORT_BYPASS_GUARD -- regression test.

Follow-on to sports_nova_v3_v18_identity_gate_regression_test.py. That test
proved the canonical-import gate enforces independently of any capture
script's own opt-in checks. This test proves the *remaining* gap named in
that mission's REAL_BLOCKER -- a direct importlib.util.spec_from_file_location
load of simulator.py, which creates a module object that never runs
worker/sports_nova_v3/__init__.py -- is DETECTED and refused by
identity_gate.assert_prospective_identity_gate_installed(), even though it
still cannot be closed without editing the hash-pinned simulator.py itself.

Never edits simulator.py or __init__.py's frozen model surface (both are
edited only to add the sentinel-stamping/assertion mechanism, never to
change any statistical/simulation behavior). Never touches a prediction
artifact. Never runs a historical backtest script.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"
SIMULATOR_PATH = ROOT / "worker" / "sports_nova_v3" / "simulator.py"

HISTORICAL_SCRIPT_GLOBS = (
    "sports_nova_v3_engine_repair_*.py",
    "sports_nova_v3_qb_*_v*.py",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run():
    results = {}

    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    hash_before = {rel: sha256_file(ROOT / rel) for rel in freeze["CODE_HASHES"]}

    from worker.sports_nova_v3 import identity_gate

    # TEST 1: canonical package import still passes.
    try:
        clearance = identity_gate.assert_prospective_identity_gate_installed()
        results["CANONICAL_IMPORT_TEST"] = {
            "PASS": clearance.get("CANONICAL_SIMULATOR_MODULE") is True
                    and clearance.get("GATE_SENTINEL") == identity_gate.IDENTITY_GATE_SENTINEL_VALUE,
            "DETAIL": clearance,
        }
    except SystemExit as exc:
        results["CANONICAL_IMPORT_TEST"] = {"PASS": False, "DETAIL": f"unexpected refusal: {exc}"}

    # TEST 2: direct file-path load of simulator.py -- the exact bypass
    # technique this project's own scripts already use for the walkforward
    # and policy modules, applied here (only inside this test, never in a
    # real launcher) to simulator.py itself.
    direct_mod_name = "sports_nova_v3_simulator_DIRECT_IMPORT_BYPASS_TEST"
    spec = importlib.util.spec_from_file_location(direct_mod_name, SIMULATOR_PATH)
    direct_mod = importlib.util.module_from_spec(spec)
    # simulator.py uses package-relative imports (`from .config import ...`),
    # so a naive spec_from_file_location load raises ImportError outright --
    # this is what a determined bypass script has to work around, by setting
    # __package__ so the relative imports resolve. Registering direct_mod in
    # sys.modules under its OWN name (never the canonical dotted name) is
    # required for that resolution to work, and is exactly what keeps it a
    # separate module object from worker.sports_nova_v3.simulator.
    direct_mod.__package__ = "worker.sports_nova_v3"
    sys.modules[direct_mod_name] = direct_mod
    try:
        spec.loader.exec_module(direct_mod)
    finally:
        del sys.modules[direct_mod_name]

    results["DIRECT_FILE_IMPORT_DETECTED"] = {
        "PASS": direct_mod is not sys.modules.get("worker.sports_nova_v3.simulator"),
        "DETAIL": {
            "direct_module_name": direct_mod.__name__,
            "direct_module_file": direct_mod.__file__,
            "is_canonical_object": direct_mod is sys.modules.get("worker.sports_nova_v3.simulator"),
        },
    }

    # This is the actual proof the raw module never got the gate: its
    # _primary_qb carries no sentinel at all, unlike the canonical one.
    raw_sentinel = getattr(direct_mod._primary_qb, "_SPORTS_NOVA_IDENTITY_GATE_SENTINEL", None)
    results["DIRECT_MODULE_UNGATED"] = {
        "PASS": raw_sentinel is None,
        "DETAIL": f"direct-loaded simulator._primary_qb sentinel = {raw_sentinel!r} (expected None)",
    }

    # TEST 3: the shared assertion refuses to treat that direct-loaded module
    # as prospectively usable -- this is the actual guard the mission asks
    # for, not just a passive observation.
    try:
        identity_gate.assert_prospective_identity_gate_installed(direct_mod)
        results["DIRECT_FILE_IMPORT_BLOCKED_IN_PROSPECTIVE"] = {
            "PASS": False, "DETAIL": "assertion did not raise -- bypass NOT blocked",
        }
    except SystemExit as exc:
        results["DIRECT_FILE_IMPORT_BLOCKED_IN_PROSPECTIVE"] = {
            "PASS": "IDENTITY_GATE_BYPASS_DETECTED" in str(exc),
            "DETAIL": str(exc),
        }

    # TEST 4: historical/backtest tooling is untouched -- none of those
    # scripts reference the new gate function, proving this mission didn't
    # wire the assertion into any of them (their kickoff<=now bypass in
    # __init__.py's _gated_primary_qb is also unmodified by this mission).
    historical_hits = []
    for pattern in HISTORICAL_SCRIPT_GLOBS:
        for p in sorted((ROOT / "scripts").glob(pattern)):
            text = p.read_text(errors="ignore")
            if "identity_gate" in text or "assert_prospective_identity_gate_installed" in text:
                historical_hits.append(str(p.relative_to(ROOT)))
    results["HISTORICAL_TOOLS_UNCHANGED"] = {
        "PASS": not historical_hits,
        "DETAIL": {"SCRIPTS_REFERENCING_NEW_GATE": historical_hits},
    }

    # TEST 5: V18 hashes unchanged -- re-verify every CODE_HASHES-listed file
    # (which does NOT include worker/sports_nova_v3/__init__.py or
    # identity_gate.py -- neither is in the freeze manifest) still matches
    # both its pre-import value and the freeze manifest's recorded value.
    hash_after = {rel: sha256_file(ROOT / rel) for rel in freeze["CODE_HASHES"]}
    mismatches = {rel: (hash_before[rel], hash_after[rel])
                  for rel in freeze["CODE_HASHES"] if hash_before[rel] != hash_after[rel]}
    frozen_manifest_ok = all(hash_after[rel] == freeze["CODE_HASHES"][rel]["sha256"]
                              for rel in freeze["CODE_HASHES"])
    results["V18_HASH_UNCHANGED_TEST"] = {
        "PASS": not mismatches and frozen_manifest_ok,
        "DETAIL": {"MISMATCHES_DURING_RUN": mismatches, "MATCHES_FREEZE_MANIFEST": frozen_manifest_ok},
    }

    all_pass = all(r["PASS"] for r in results.values())
    print(json.dumps({"ALL_PASS": all_pass, "RESULTS": results}, indent=2, default=str))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run())
