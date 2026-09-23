"""Frozen V3 contracts. Implementation and empirical certification belong to Luna.

SPORTS_V18_IDENTITY_POLICY_SINGLE_ENFORCEMENT_POINT: wraps simulator._primary_qb
in-place (module attribute swap on the already-imported module object, not an
edit to simulator.py's source -- that file stays hash-pinned and unchanged)
so every normal `from worker.sports_nova_v3.simulator import _primary_qb` --
including one written by a future script that has never heard of
worker/sports_nova_v3/identity_gate.py -- reaches the canonical identity gate.
Package __init__.py always runs, exactly once, before a dotted submodule
import completes; this is the closest available enforcement point that does
not require touching frozen, hash-pinned model code. See identity_gate.py's
own docstring for the one bypass this cannot close (a direct
importlib.util.spec_from_file_location load of simulator.py, which never
executes this file) -- SPORTS_V18_DIRECT_IMPORT_BYPASS_GUARD stamps a
sentinel onto the gated function below precisely so that bypass can be
DETECTED and refused at prospective-launch time via
identity_gate.assert_prospective_identity_gate_installed(), even though it
still cannot be closed without editing simulator.py itself.
"""
from datetime import datetime, timezone

from . import simulator as _simulator
from . import identity_gate as _identity_gate

_ORIGINAL_PRIMARY_QB = _simulator._primary_qb


def _gated_primary_qb(pregame, team_id):
    now = datetime.now(timezone.utc)
    kickoff = pregame.kickoff if pregame.kickoff.tzinfo else pregame.kickoff.replace(tzinfo=timezone.utc)
    if kickoff <= now:
        # Already-played game: this can only be a historical diagnostic/
        # backtest call (~15 scripts under scripts/sports_nova_v3_engine_repair_*
        # and sports_nova_v3_qb_*_v*.py replay 2020-2025 games this way).
        # The identity-uncertainty policy exists for PROSPECTIVE games only --
        # a completed game's identity is already fixed by its own box score,
        # nothing to gate. Bypassing here is what keeps "all current
        # entrypoints preserve behavior" true for those unrelated scripts:
        # without this check, a legacy backtest re-run today would return
        # None for ATL/MIN/SEA team-games purely because those three
        # abbreviations happen to be UNCERTAIN in TODAY's identity artifact,
        # for reasons having nothing to do with the historical game being
        # replayed.
        return _ORIGINAL_PRIMARY_QB(pregame, team_id)

    # Freshness is judged against real wall-clock now, never pregame.as_of:
    # that field is the frozen model's own causal cutoff, set to ~1 second
    # before the target game's kickoff (see make_state() in the walkforward
    # script) -- for a future game that is itself days ahead of the real
    # present, using it here would make every prospective game "stale"
    # relative to an identity artifact refreshed today, or "fresh" for the
    # wrong reason.
    clearance = _identity_gate.require_identity_clearance(
        pregame.game_id, team_id, now)
    decision = clearance["DECISION"]
    if decision == _identity_gate.DECISION_PRIMARY_ALLOWED:
        return _ORIGINAL_PRIMARY_QB(pregame, team_id)
    if decision == _identity_gate.DECISION_EXCLUDED_UNCERTAIN:
        # Never silently guess: no player is returned. Existing callers
        # already treat a None return as "nothing to report for this team"
        # (see capture_new_game's `if qb_state is None: continue`), so a
        # caller that has never heard of this policy still fails safe
        # instead of reporting an unvetted identity.
        return None
    # BLOCKED_STALE_IDENTITY / BLOCKED_MISSING_IDENTITY_ARTIFACT: these are
    # infrastructure failures, not ordinary per-team uncertainty, so this
    # halts the whole run rather than silently skipping one team.
    raise SystemExit(f"{decision}: {clearance.get('REASON', clearance)}")


_gated_primary_qb._SPORTS_NOVA_IDENTITY_GATE_SENTINEL = _identity_gate.IDENTITY_GATE_SENTINEL_VALUE

_simulator._primary_qb = _gated_primary_qb
