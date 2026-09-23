"""V25 simulator boundary.  The frozen V23 simulator remains the execution engine, byte-for-byte; V25 repairs only the opportunity shares.

A RosterSnapshot is REQUIRED: there is no silent opt-out.  NoEligibleRecipientsError propagates (never a silent invalid-role allocation).
"""
from __future__ import annotations

from worker.sports_nova_v23 import simulator as _v23
from worker.sports_nova_v23.simulator import (  # re-exported, unchanged
    STAT_NAMES, TEAM_STAT_NAMES, SimulationBatch, set_identity_resolutions,
    _model_ref, _state_hash, _feature, _inc, _qb_shares, _primary_qb,
)
from .config import (
    DOUBTFUL_POLICY_HASH, ELIGIBILITY_POLICY_HASH, MODEL_VERSION, REDISTRIBUTION_POLICY_HASH, RNG_ALGORITHM, ROLE_POLICY_HASH)
from .eligibility import RosterSnapshot
from .redistribution import assert_repair_invariants, repair_pregame_state


def prepare_state(pregame_state, roster: RosterSnapshot):
    """Repair + verify; returns (repaired_state, audit).  Raises if any repair invariant fails."""
    if not isinstance(roster, RosterSnapshot):
        raise TypeError("V25 requires a RosterSnapshot (pregame roster/injury/position evidence)")
    repaired, audit = repair_pregame_state(pregame_state, roster)
    assert_repair_invariants(pregame_state, repaired, audit)
    return repaired, audit


def simulate_game(pregame_state, n_sims: int, seed: int, model_version: str, *, roster: RosterSnapshot) -> SimulationBatch:
    if model_version != MODEL_VERSION:
        raise ValueError("unfrozen simulation version")
    repaired, audit = prepare_state(pregame_state, roster)
    batch = _v23.simulate_game(repaired, n_sims, seed, _v23.MODEL_VERSION)
    runtime = dict(batch.runtime)
    runtime.update({"m1_engine_base": _v23.MODEL_VERSION, "rng_algorithm": RNG_ALGORITHM,
                    "allocation_repair": "role_aware_share_repair_then_unchanged_v23_draw",
                    "raw_state_hash": audit.raw_state_hash, "repaired_state_hash": audit.repaired_state_hash,
                    "eligibility_policy_hash": ELIGIBILITY_POLICY_HASH, "role_policy_hash": ROLE_POLICY_HASH,
                    "redistribution_policy_hash": REDISTRIBUTION_POLICY_HASH, "doubtful_policy_hash": DOUBTFUL_POLICY_HASH,
                    "roster_week": roster.roster_week})
    return SimulationBatch(batch.game_id, batch.n_sims, batch.seed, MODEL_VERSION, batch.model, runtime,
                           batch.player_ids, batch.player_team, batch.team_ids, batch.player_stats,
                           batch.team_stats, batch.winner, batch.status, audit.raw_state_hash)
