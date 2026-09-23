"""V27 simulator boundary: V26 completion -> V25 repair (both unchanged) -> V23 scoring execution with the causal FG rate substituted.

V25/V26 delegate scoring to `worker.sports_nova_v23.simulator.simulate_game`, which fits its own parameters internally and offers no override hook.  V27 therefore
performs the same outer steps that function performs (validation, `fit_distributions([], as_of)`, the `_run_one` loop, array construction) and changes exactly one
thing between fitting and running: `fg_rate`.  `_v23._run_one` -- the entire play/score model -- is called unchanged, with the same per-simulation RNG streams.

Entry points
  simulate_game(raw_state, n, seed, mv, *, roster, completion)   full V26->V25->V23 pipeline, causal FG rate (production path)
  simulate_scoring(pregame_state, n, seed, mv)                    the scoring boundary alone (no roster layer); used for the historical reproduction, where no
                                                                  RosterSnapshot exists.  Same code the pipeline calls.
  *_forced_fg_rate(...)                                           REGRESSION HARNESS ONLY (test T1): explicit float, no default, source recorded as FORCED.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from worker.sports_nova_v23 import simulator as _v23
from worker.sports_nova_v23.simulator import SimulationBatch, STAT_NAMES, TEAM_STAT_NAMES, _model_ref, _state_hash
from worker.sports_nova_v25_role_aware import simulator as _v25
from worker.sports_nova_v25_role_aware.config import (
    DOUBTFUL_POLICY_HASH, ELIGIBILITY_POLICY_HASH, MODEL_VERSION as V25_MODEL_VERSION, REDISTRIBUTION_POLICY_HASH, ROLE_POLICY_HASH)
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from worker.sports_nova_v26_active_skill_state import simulator as _v26
from worker.sports_nova_v26_active_skill_state.completion import CompletionResult
from worker.sports_nova_v26_active_skill_state.config import COMPLETION_POLICY_HASH
from worker.sports_nova_v3.distributions import fit_distributions
from .causal_fg import CausalFGError, causal_fg_rate_for_game
from .config import CAUSAL_FG_POLICY_HASH, MODEL_VERSION, RNG_ALGORITHM, V23_MODEL_VERSION, V26_MODEL_VERSION

prepare_state = _v26.prepare_state          # completion first, then V25's unchanged repair + invariants (re-exported, unchanged)


def _score(pregame_state, n_sims: int, seed: int, fg_rate: float, fg_evidence: dict) -> SimulationBatch:
    """Body of worker.sports_nova_v23.simulator.simulate_game with `fg_rate` substituted after fitting; `_run_one` unchanged."""
    if type(n_sims) is not int or n_sims <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer n_sims and nonnegative integer seed required")
    if not isinstance(fg_rate, float) or not math.isfinite(fg_rate):
        raise CausalFGError("fg_rate must be an explicit finite float")
    model = _model_ref(pregame_state)
    params = fit_distributions([], pregame_state.as_of)
    params = dataclasses.replace(params, fg_rate=fg_rate)
    player_ids = tuple(sorted(p.player_id for p in pregame_state.players))
    team_ids = (pregame_state.home.team_id, pregame_state.away.team_id)
    player_team = {p.player_id: p.team_id for p in pregame_state.players}
    player_arrays = {stat: np.zeros((n_sims, len(player_ids)), dtype=np.int64) for stat in STAT_NAMES}
    team_arrays = {stat: np.zeros((n_sims, len(team_ids)), dtype=np.int64) for stat in TEAM_STAT_NAMES}
    winners = np.empty(n_sims, dtype="U4")
    for i in range(n_sims):
        stats, team, win = _v23._run_one(pregame_state, i, seed, params)
        winners[i] = win
        for j, pid in enumerate(player_ids):
            for stat in STAT_NAMES:
                player_arrays[stat][i, j] = stats[pid][stat]
        for j, tid in enumerate(team_ids):
            for stat in TEAM_STAT_NAMES:
                team_arrays[stat][i, j] = team[tid][stat]
    for values in list(player_arrays.values()) + list(team_arrays.values()) + [winners]:
        values.setflags(write=False)
    runtime = {"rng_algorithm": RNG_ALGORITHM, "numpy_version": np.__version__, "dtype": "int64", "path_order": "simulation_index_ascending", "status": params.status}
    runtime.update(fg_evidence)
    return SimulationBatch(pregame_state.game_id, n_sims, seed, V23_MODEL_VERSION, model, runtime, player_ids, player_team, team_ids,
                           player_arrays, team_arrays, winners, "RESEARCH_GRADE_DEFAULT_PARAMETERS", _state_hash(pregame_state))


def _causal(pregame_state) -> tuple[float, dict]:
    r = causal_fg_rate_for_game(pregame_state.game_id)
    ev = r.evidence()
    ev.update({"fg_rate_source": "CAUSAL_ESTIMATOR_A", "causal_fg_policy_hash": CAUSAL_FG_POLICY_HASH})
    return r.rate, ev


def _forced(fg_rate: float) -> tuple[float, dict]:
    return float(fg_rate), {"fg_rate": float(fg_rate), "fg_rate_source": "FORCED_REGRESSION_HARNESS", "causal_fg_policy_hash": CAUSAL_FG_POLICY_HASH}


def _check_version(model_version: str) -> None:
    if model_version != MODEL_VERSION:
        raise ValueError("unfrozen simulation version")


def simulate_scoring(pregame_state, n_sims: int, seed: int, model_version: str) -> SimulationBatch:
    """Scoring boundary only: V23 execution with the causal FG rate (no roster layer)."""
    _check_version(model_version)
    rate, ev = _causal(pregame_state)
    b = _score(pregame_state, n_sims, seed, rate, ev)
    return dataclasses.replace(b, model_version=MODEL_VERSION)


def simulate_scoring_forced_fg_rate(pregame_state, n_sims: int, seed: int, model_version: str, *, fg_rate: float) -> SimulationBatch:
    _check_version(model_version)
    rate, ev = _forced(fg_rate)
    b = _score(pregame_state, n_sims, seed, rate, ev)
    return dataclasses.replace(b, model_version=MODEL_VERSION)


def _pipeline(raw_state, n_sims, seed, model_version, roster: RosterSnapshot, completion: CompletionResult, provider) -> SimulationBatch:
    _check_version(model_version)
    if completion.raw_state_hash != _state_hash(raw_state):
        raise ValueError("completion was built from a different raw state")
    repaired, audit = _v25.prepare_state(completion.state, roster)          # V25's unchanged repair + invariants on the V26-completed state
    rate, ev = provider(repaired)
    batch = _score(repaired, n_sims, seed, rate, ev)
    runtime = dict(batch.runtime)
    runtime.update({"m1_engine_base": V23_MODEL_VERSION, "v25_layer": V25_MODEL_VERSION, "v26_layer": V26_MODEL_VERSION,
                    "rng_algorithm": RNG_ALGORITHM,
                    "allocation_repair": "role_aware_share_repair_then_unchanged_v23_draw",
                    "raw_state_hash": audit.raw_state_hash, "repaired_state_hash": audit.repaired_state_hash,
                    "eligibility_policy_hash": ELIGIBILITY_POLICY_HASH, "role_policy_hash": ROLE_POLICY_HASH,
                    "redistribution_policy_hash": REDISTRIBUTION_POLICY_HASH, "doubtful_policy_hash": DOUBTFUL_POLICY_HASH, "roster_week": roster.roster_week,
                    "state_completion": "active_skill_state_completion_then_unchanged_v25_then_v23_draw_with_causal_fg_rate",
                    "completion_policy_hash": COMPLETION_POLICY_HASH, "raw_input_state_hash": completion.raw_state_hash,
                    "completed_state_hash": completion.completed_state_hash, "completion_changed": completion.changed})
    return SimulationBatch(batch.game_id, batch.n_sims, batch.seed, MODEL_VERSION, batch.model, runtime, batch.player_ids, batch.player_team, batch.team_ids,
                           batch.player_stats, batch.team_stats, batch.winner, batch.status, completion.raw_state_hash)


def simulate_game(raw_state, n_sims: int, seed: int, model_version: str, *, roster: RosterSnapshot, completion: CompletionResult) -> SimulationBatch:
    """Production path.  A RosterSnapshot and a CompletionResult are REQUIRED (as in V26); the FG rate is ALWAYS the causal estimate."""
    return _pipeline(raw_state, n_sims, seed, model_version, roster, completion, _causal)


def simulate_game_forced_fg_rate(raw_state, n_sims: int, seed: int, model_version: str, *, roster: RosterSnapshot, completion: CompletionResult,
                                 fg_rate: float) -> SimulationBatch:
    """REGRESSION HARNESS ONLY: the full pipeline with an explicit fg_rate (no default).  Used by T1 to show V27 == V26 at fg_rate=0.08."""
    return _pipeline(raw_state, n_sims, seed, model_version, roster, completion, lambda _s: _forced(fg_rate))
