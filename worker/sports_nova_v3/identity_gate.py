"""SPORTS_V18_IDENTITY_POLICY_SINGLE_ENFORCEMENT_POINT -- canonical gate.

Not part of the frozen V18 model surface (not in SPORTS_NOVA_V18_FREEZE_MANIFEST
.json's CODE_HASHES) and touches no statistical/simulation code -- this module
only decides whether a (game, team) may receive a reported QB identity at
prediction time, per the policy frozen in
scripts/sports_nova_v3_v18_qb_uncertainty_policy.py. It is loaded via direct
file path (see _load_policy_module below), the same pattern the existing
capture scripts already use for the frozen walkforward module, so it needs no
sys.path mutation and no dependency on `scripts/` being an importable package.

Why this lives at worker/sports_nova_v3/, not in scripts/: worker/sports_nova_v3/
__init__.py wraps simulator._primary_qb here (see that file) so that ANY code
which does `from worker.sports_nova_v3.simulator import _primary_qb` --
including a future script that has never heard of the policy module -- gets
the gated version, because Python always runs a package's __init__.py before
a dotted submodule import completes, and only ever runs it once (the
monkeypatch on the shared module object persists for the whole process after
that). This is the closest thing to a non-bypassable enforcement point that
does not require editing simulator.py itself (forbidden: it is hash-pinned,
and editing it would trip BLOCKED_FREEZE_HASH_MISMATCH in every entrypoint
that checks V18_HASH_UNCHANGED).

Known, honest limit (see REAL_BLOCKER in the mission report, not hidden
here): a script that loads simulator.py by direct file path via
importlib.util.spec_from_file_location -- exactly the technique this
project's own scripts use for the walkforward module -- creates a SEPARATE
module object that never passes through worker/sports_nova_v3/__init__.py,
and would get the raw, unwrapped _primary_qb. That is a real residual bypass
vector, not something this module can close without modifying simulator.py.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOLUTION_PATH = (
    ROOT / "data" / "sports_nova_v3" / "prospective" / "identity"
    / "SPORTS_NOVA_V18_QB_IDENTITY_RESOLUTION_2026.json"
)

DECISION_PRIMARY_ALLOWED = "PRIMARY_ALLOWED"
DECISION_EXCLUDED_UNCERTAIN = "EXCLUDED_IDENTITY_UNCERTAIN"
DECISION_BLOCKED_STALE = "BLOCKED_STALE_IDENTITY"
DECISION_BLOCKED_MISSING = "BLOCKED_MISSING_IDENTITY_ARTIFACT"

# SPORTS_V18_DIRECT_IMPORT_BYPASS_GUARD: stamped onto the gated _primary_qb in
# __init__.py so any caller can prove the wrapper actually installed, without
# re-deriving policy logic. Bump the suffix only if the gate's semantics
# change in a way a caller should be able to tell apart from the prior gate.
IDENTITY_GATE_SENTINEL_VALUE = "SPORTS_NOVA_V18_IDENTITY_GATE_V1"
_SENTINEL_ATTR = "_SPORTS_NOVA_IDENTITY_GATE_SENTINEL"


def assert_prospective_identity_gate_installed(simulator_module=None) -> dict:
    """Prospective-start assertion. Every prospective launcher (a script that
    may call _primary_qb for a game whose kickoff is still in the future)
    must call this once, before its first prediction, with no arguments.

    Fails closed (raises SystemExit) unless:
      1. the simulator module in play is the EXACT object registered under
         the canonical dotted name "worker.sports_nova_v3.simulator" in
         sys.modules -- proof it was reached via a normal package import and
         therefore passed through worker/sports_nova_v3/__init__.py; and
      2. that module's _primary_qb carries the gate sentinel __init__.py
         stamps onto the wrapper it installs.

    A script that instead loads simulator.py directly (e.g. via
    importlib.util.spec_from_file_location, the same technique this
    project's own scripts already use for the walkforward/policy modules)
    gets a distinct module object that never runs __init__.py and so fails
    check 1 even if, by some future refactor, its _primary_qb happened to
    carry a stray attribute of the same name -- the two checks are
    independent, not redundant.

    simulator_module: defaults to `from worker.sports_nova_v3 import
    simulator` (the real case for every known launcher). A caller passes it
    explicitly only to assert about a *different* module object it already
    holds -- this is how the bypass-detection test below exercises this
    function against a deliberately direct-file-loaded copy without ever
    installing that copy as the process's real simulator.
    """
    import sys

    from . import simulator as _canonical_simulator

    mod = simulator_module if simulator_module is not None else _canonical_simulator

    canonical_registered = sys.modules.get("worker.sports_nova_v3.simulator")
    if mod is not canonical_registered:
        raise SystemExit(
            "IDENTITY_GATE_BYPASS_DETECTED: simulator module object is not the "
            "canonical worker.sports_nova_v3.simulator instance in sys.modules "
            f"(module name={getattr(mod, '__name__', None)!r}, "
            f"file={getattr(mod, '__file__', None)!r}). This looks like a direct "
            "file-path (importlib.util.spec_from_file_location) load of "
            "simulator.py, which never runs worker/sports_nova_v3/__init__.py and "
            "therefore never receives the QB identity gate. Refusing to run "
            "prospectively.")

    sentinel = getattr(getattr(mod, "_primary_qb", None), _SENTINEL_ATTR, None)
    if sentinel != IDENTITY_GATE_SENTINEL_VALUE:
        raise SystemExit(
            "IDENTITY_GATE_BYPASS_DETECTED: worker.sports_nova_v3.simulator."
            f"_primary_qb has no valid identity-gate sentinel (found {sentinel!r}, "
            f"expected {IDENTITY_GATE_SENTINEL_VALUE!r}) -- the canonical package "
            "__init__.py monkeypatch did not install. Refusing to run "
            "prospectively.")

    return {"GATE_SENTINEL": sentinel, "CANONICAL_SIMULATOR_MODULE": True}


def _load_policy_module():
    """Loads scripts/sports_nova_v3_v18_qb_uncertainty_policy.py by direct
    file path -- same technique load_frozen_module_and_verify_freeze() in
    the existing capture scripts already uses for the walkforward module, so
    this works regardless of CWD or whether scripts/ is on sys.path."""
    path = ROOT / "scripts" / "sports_nova_v3_v18_qb_uncertainty_policy.py"
    spec = importlib.util.spec_from_file_location(
        "sports_nova_v3_v18_qb_uncertainty_policy_GATE", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_policy = _load_policy_module()
FRESHNESS_MAX_AGE_HOURS = _policy.FRESHNESS_MAX_AGE_HOURS


def _sha256_file(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def canonical_exclusion_hash(game_id: str, team_id: str, record: dict, identity_artifact_sha256: str) -> str:
    """Deterministic given identical inputs: hashed over exactly the
    provenance fields the mission requires (team, qb_status, candidates,
    source, as_of, identity_artifact_hash) plus game_id -- deliberately
    EXCLUDES any wall-clock value (no CREATED_AT/run timestamp), so calling
    this twice against the same identity artifact for the same team/game
    always produces the same hash."""
    payload = {
        "GAME_ID": game_id,
        "TEAM": team_id,
        "QB_STATUS": record.get("STATUS"),
        "CANDIDATES": record.get("CANDIDATES", []),
        "SOURCE": record.get("SOURCE", []),
        "AS_OF": record.get("AS_OF"),
        "IDENTITY_ARTIFACT_HASH": identity_artifact_sha256,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def require_identity_clearance(game_id: str, team_id: str, prediction_time: datetime,
                                resolution_path: Optional[Path] = None) -> dict:
    """The canonical gate. Returns a dict with at least:
      DECISION: one of DECISION_PRIMARY_ALLOWED / DECISION_EXCLUDED_UNCERTAIN /
                DECISION_BLOCKED_STALE / DECISION_BLOCKED_MISSING
    and, for every decision except PRIMARY_ALLOWED, provenance fields:
      TEAM, QB_STATUS, CANDIDATES, SOURCE, AS_OF, IDENTITY_ARTIFACT_HASH,
      EXCLUSION_HASH (deterministic; see canonical_exclusion_hash).

    Fail-closed: a missing or unreadable identity artifact returns
    DECISION_BLOCKED_MISSING, never PRIMARY_ALLOWED.

    resolution_path is an override for testing only; production callers
    (the __init__.py monkeypatch) never pass it, always using the one real
    identity artifact on disk.
    """
    path = resolution_path or DEFAULT_RESOLUTION_PATH
    identity_artifact_sha256 = _sha256_file(path)

    if not path.exists():
        return {"DECISION": DECISION_BLOCKED_MISSING, "TEAM": team_id, "GAME_ID": game_id,
                "REASON": f"no identity artifact at {path}"}

    try:
        resolution = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"DECISION": DECISION_BLOCKED_MISSING, "TEAM": team_id, "GAME_ID": game_id,
                "REASON": f"identity artifact unreadable: {exc}"}

    as_of_raw = resolution.get("AS_OF_UTC")
    if not as_of_raw:
        return {"DECISION": DECISION_BLOCKED_STALE, "TEAM": team_id, "GAME_ID": game_id,
                "REASON": "identity artifact has no AS_OF_UTC"}
    as_of = datetime.fromisoformat(str(as_of_raw))
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.replace(tzinfo=timezone.utc)
    age_hours = (prediction_time - as_of).total_seconds() / 3600.0
    if age_hours > FRESHNESS_MAX_AGE_HOURS:
        return {"DECISION": DECISION_BLOCKED_STALE, "TEAM": team_id, "GAME_ID": game_id,
                "AGE_HOURS": round(age_hours, 2),
                "REASON": f"identity artifact is {age_hours:.1f}h old vs "
                          f"FRESHNESS_MAX_AGE_HOURS={FRESHNESS_MAX_AGE_HOURS}"}

    record = resolution.get("TEAMS", {}).get(team_id)
    if record is None:
        return {"DECISION": DECISION_EXCLUDED_UNCERTAIN, "TEAM": team_id, "GAME_ID": game_id,
                "QB_STATUS": "NO_RESOLUTION_RECORD", "CANDIDATES": [], "SOURCE": [],
                "AS_OF": None, "IDENTITY_ARTIFACT_HASH": identity_artifact_sha256,
                "EXCLUSION_HASH": canonical_exclusion_hash(
                    game_id, team_id, {"STATUS": "NO_RESOLUTION_RECORD"}, identity_artifact_sha256)}

    if record.get("STATUS") == "CONFIRMED":
        return {"DECISION": DECISION_PRIMARY_ALLOWED, "TEAM": team_id, "GAME_ID": game_id,
                "QB_ID": record.get("QB_ID"), "IDENTITY_ARTIFACT_HASH": identity_artifact_sha256}

    return {"DECISION": DECISION_EXCLUDED_UNCERTAIN, "TEAM": team_id, "GAME_ID": game_id,
            "QB_STATUS": record.get("STATUS"), "CANDIDATES": record.get("CANDIDATES", []),
            "SOURCE": record.get("SOURCE", []), "AS_OF": record.get("AS_OF"),
            "IDENTITY_ARTIFACT_HASH": identity_artifact_sha256,
            "EXCLUSION_HASH": canonical_exclusion_hash(game_id, team_id, record, identity_artifact_sha256)}
