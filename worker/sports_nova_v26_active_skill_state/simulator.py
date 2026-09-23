"""V26 simulator boundary: complete the state (V26), repair opportunity shares (V25, byte-identical), draw with the frozen V23 engine.

The batch reports the RAW M1 state hash as `state_hash` (the input the panel's smoke hash and every earlier version's replay checks refer to); the completed
and V25-repaired hashes are recorded in the runtime block.
"""
from __future__ import annotations

from worker.sports_nova_v23 import simulator as _v23
from worker.sports_nova_v23.simulator import SimulationBatch, _state_hash
from worker.sports_nova_v25_role_aware import simulator as _v25
from worker.sports_nova_v25_role_aware.config import MODEL_VERSION as V25_MODEL_VERSION
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from .completion import CompletionResult, complete_state
from .config import COMPLETION_POLICY_HASH, MODEL_VERSION


def prepare_state(state, roster: RosterSnapshot, prior, *, known_out=frozenset()):
    """(completion, repaired_state, v25_audit).  Completion first, then V25's unchanged repair + invariants on the completed state."""
    completion = complete_state(state, roster, prior, known_out=known_out)
    repaired, audit = _v25.prepare_state(completion.state, roster)
    return completion, repaired, audit


def simulate_game(raw_state, n_sims: int, seed: int, model_version: str, *, roster: RosterSnapshot, completion: CompletionResult) -> SimulationBatch:
    if model_version != MODEL_VERSION:
        raise ValueError("unfrozen simulation version")
    if completion.raw_state_hash != _state_hash(raw_state):
        raise ValueError("completion was built from a different raw state")
    batch = _v25.simulate_game(completion.state, n_sims, seed, V25_MODEL_VERSION, roster=roster)
    runtime = dict(batch.runtime)
    runtime.update({"m1_engine_base": _v23.MODEL_VERSION, "v25_layer": V25_MODEL_VERSION, "state_completion": "active_skill_state_completion_then_unchanged_v25_then_unchanged_v23_draw",
                    "completion_policy_hash": COMPLETION_POLICY_HASH, "raw_input_state_hash": completion.raw_state_hash,
                    "completed_state_hash": completion.completed_state_hash, "completion_changed": completion.changed})
    return SimulationBatch(batch.game_id, batch.n_sims, batch.seed, MODEL_VERSION, batch.model, runtime, batch.player_ids, batch.player_team, batch.team_ids,
                           batch.player_stats, batch.team_stats, batch.winner, batch.status, completion.raw_state_hash)
