"""V23 challenger: identical to worker/sports_nova_v21/simulator.py except
`allocate_opportunities` is imported from worker.sports_nova_v23.allocation
(the residual-allocation fix) instead of worker.sports_nova_v3.allocation.
See worker/sports_nova_v21/allocation.py fork rationale in
worker/sports_nova_v23/allocation.py and the ablation evidence in
worker/sports_nova_v23/__init__.py.

Everything else -- PAT crediting, pace/block-volume, efficiency, TD rates,
catch rate, QB identity/roster construction -- is imported UNCHANGED from
worker.sports_nova_v21 (and, through it, v19/v3), not copied, so this is a
single, attributable, diffable change: only the allocation import differs
from worker/sports_nova_v21/simulator.py.
"""
from __future__ import annotations

import numpy as np

from .config import MODEL_VERSION, RNG_ALGORITHM
from worker.sports_nova_v21.config import PAT_MAKE_RATE
from worker.sports_nova_v3.schemas import PregameState
from worker.sports_nova_v3.distributions import fit_distributions, sample_compound_signed
from worker.sports_nova_v3.environment import draw_environment
from worker.sports_nova_v3.game_script import script_pass_rate
from worker.sports_nova_v3.game_state import GameState
from worker.sports_nova_v3.play_volume import draw_block_volume
from worker.sports_nova_v3.play_selection import select_plays
from worker.sports_nova_v23.allocation import allocate_opportunities
from worker.sports_nova_v3.scoring import winner
from worker.sports_nova_v19.simulator import (
    STAT_NAMES, TEAM_STAT_NAMES, SimulationBatch,
    _model_ref, _state_hash, _feature, _inc, _qb_shares, _primary_qb,
    set_identity_resolutions,
)


def _run_one(pregame: PregameState, sim_id: int, seed: int, params):
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), int(sim_id)])))
    player_ids = sorted(p.player_id for p in pregame.players)
    stats = {pid: {name: 0 for name in STAT_NAMES} for pid in player_ids}
    team_ids = (pregame.home.team_id, pregame.away.team_id)
    team = {tid: {name: 0 for name in TEAM_STAT_NAMES} for tid in team_ids}
    state = GameState(game_id=pregame.game_id, simulation_id=sim_id, block_index=0,
        seconds_remaining=3600, home_score=0, away_score=0,
        possession=pregame.home.team_id, field_position=25)
    env = draw_environment(pregame, rng, epistemic_id=sim_id)
    eff_mult = {
        pregame.home.team_id: float(np.clip(1.0 + env.home_efficiency, 0.3, 2.5)),
        pregame.away.team_id: float(np.clip(1.0 + env.away_efficiency, 0.3, 2.5)),
    }
    player_by_id = {p.player_id: p for p in pregame.players}
    qb_shares = {tid: _qb_shares(pregame, tid) for tid in team_ids}
    qb_draw_probs = {}
    for tid, (ids, shares) in qb_shares.items():
        qb_draw_probs[tid] = (ids, rng.dirichlet(np.maximum(shares * 80.0, 1e-6)) if len(ids) else shares)
    for _ in range(60):
        if state.seconds_remaining <= 0:
            break
        offense = state.possession
        offense_team = pregame.home if offense == pregame.home.team_id else pregame.away
        team[offense]["blocks"] += 1
        plays = draw_block_volume(state, env, rng)
        selection = select_plays(plays, script_pass_rate(pregame, state, env), rng,
            sack_rate=_feature(offense_team, "sack_rate", .065),
            scramble_rate=_feature(offense_team, "scramble_rate", .07))
        alloc = allocate_opportunities(pregame, offense, selection.pass_attempts,
                                       selection.designed_rushes, rng)
        ids, probs = qb_draw_probs[offense]
        qb = player_by_id[ids[int(rng.choice(len(ids), p=probs))]] if len(ids) else None
        if qb is not None:
            _inc(stats, qb.player_id, "pass_attempts", selection.pass_attempts)
            _inc(stats, qb.player_id, "rush_attempts", selection.scrambles)
            scramble_yards = sample_compound_signed(selection.scrambles, 4.0 * eff_mult[offense], 5.0, rng)
            _inc(stats, qb.player_id, "rush_yards", scramble_yards)
            team[offense]["rush_yards"] += scramble_yards
        team[offense]["pass_attempts"] += selection.pass_attempts
        team[offense]["rush_attempts"] += selection.rush_attempts
        block_points = 0
        for pid, targets in alloc.target_counts.items():
            _inc(stats, pid, "targets", targets)
            receiver = next(p for p in pregame.players if p.player_id == pid)
            catch_rate = np.clip(_feature(receiver, "catch_rate", .64), .05, .95)
            rec = int(rng.binomial(targets, catch_rate))
            ypr_mean = _feature(receiver, "yards_per_reception", params.receiving_yards_mean)
            rec_yards = sample_compound_signed(rec, ypr_mean * eff_mult[offense], params.receiving_yards_sd, rng)
            _inc(stats, pid, "receptions", rec)
            _inc(stats, pid, "receiving_yards", rec_yards)
            if qb is not None:
                _inc(stats, qb.player_id, "pass_yards", rec_yards)
            team[offense]["pass_yards"] += rec_yards
            pass_td_rate = _feature(offense_team, "pass_td_rate", params.td_rate)
            td_count = int(rng.binomial(rec, np.clip(pass_td_rate, .001, .2)))
            if td_count:
                _inc(stats, pid, "receiving_tds", td_count)
                _inc(stats, pid, "scored_tds", td_count)
                if qb is not None:
                    _inc(stats, qb.player_id, "pass_tds", td_count)
                made_pats = int(rng.binomial(td_count, PAT_MAKE_RATE))
                block_points += 6 * td_count + made_pats
        for pid, carries in alloc.carry_counts.items():
            _inc(stats, pid, "rush_attempts", carries)
            rush_yards = sample_compound_signed(carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng)
            _inc(stats, pid, "rush_yards", rush_yards)
            team[offense]["rush_yards"] += rush_yards
            rush_td_rate = _feature(offense_team, "rush_td_rate", params.td_rate * .65)
            rush_td = int(rng.binomial(carries, np.clip(rush_td_rate, .0005, .15)))
            if rush_td:
                _inc(stats, pid, "rush_tds", rush_td)
                _inc(stats, pid, "scored_tds", rush_td)
                made_pats = int(rng.binomial(rush_td, PAT_MAKE_RATE))
                block_points += 6 * rush_td + made_pats
        if alloc.residual_carries:
            team[offense]["rush_yards"] += sample_compound_signed(
                alloc.residual_carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng)
        if block_points == 0 and selection.plays >= 3 and rng.random() < params.fg_rate:
            block_points = 3
        elapsed = max(1, min(500, int(round(plays * env.pace_seconds))))
        if offense == team_ids[0]:
            state = state.advance(seconds=elapsed, possession=team_ids[1], field_position=int(rng.integers(15, 86)),
                home_score=state.home_score + block_points)
        else:
            state = state.advance(seconds=elapsed, possession=team_ids[0], field_position=int(rng.integers(15, 86)),
                away_score=state.away_score + block_points)
    for tid in team_ids:
        team[tid]["score"] = state.home_score if tid == team_ids[0] else state.away_score
        team[tid]["total"] = team[tid]["score"]
    return stats, team, winner(state.home_score, state.away_score)


def simulate_game(pregame_state: PregameState, n_sims: int, seed: int, model_version: str) -> SimulationBatch:
    if not isinstance(pregame_state, PregameState):
        raise TypeError("validated PregameState required")
    if type(n_sims) is not int or n_sims <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer n_sims and nonnegative integer seed required")
    if model_version != MODEL_VERSION:
        raise ValueError("unfrozen simulation version")
    model = _model_ref(pregame_state)
    params = fit_distributions([], pregame_state.as_of)
    player_ids = tuple(sorted(p.player_id for p in pregame_state.players))
    team_ids = (pregame_state.home.team_id, pregame_state.away.team_id)
    player_team = {p.player_id: p.team_id for p in pregame_state.players}
    player_arrays = {stat: np.zeros((n_sims, len(player_ids)), dtype=np.int64) for stat in STAT_NAMES}
    team_arrays = {stat: np.zeros((n_sims, len(team_ids)), dtype=np.int64) for stat in TEAM_STAT_NAMES}
    winners = np.empty(n_sims, dtype="U4")
    for i in range(n_sims):
        stats, team, win = _run_one(pregame_state, i, seed, params)
        winners[i] = win
        for j, pid in enumerate(player_ids):
            for stat in STAT_NAMES:
                player_arrays[stat][i, j] = stats[pid][stat]
        for j, tid in enumerate(team_ids):
            for stat in TEAM_STAT_NAMES:
                team_arrays[stat][i, j] = team[tid][stat]
    for values in list(player_arrays.values()) + list(team_arrays.values()) + [winners]:
        values.setflags(write=False)
    runtime = {"rng_algorithm": RNG_ALGORITHM, "numpy_version": np.__version__,
               "dtype": "int64", "path_order": "simulation_index_ascending", "status": params.status}
    return SimulationBatch(pregame_state.game_id, n_sims, seed, model_version, model, runtime,
        player_ids, player_team, team_ids, player_arrays, team_arrays, winners,
        "RESEARCH_GRADE_DEFAULT_PARAMETERS", _state_hash(pregame_state))
