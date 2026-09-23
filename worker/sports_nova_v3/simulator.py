"""Deterministic V3 drive/block simulation engine."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
import numpy as np

from .config import MODEL_VERSION, RNG_ALGORITHM
from .schemas import FrozenModelRef, PregameState
from .distributions import fit_distributions, sample_compound_signed
from .environment import draw_environment
from .game_script import script_pass_rate
from .game_state import GameState
from .play_volume import draw_block_volume
from .play_selection import select_plays
from .allocation import allocate_opportunities
from .scoring import winner

STAT_NAMES = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards",
              "rush_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "scored_tds")
TEAM_STAT_NAMES = ("score", "pass_attempts", "pass_yards", "rush_attempts", "rush_yards", "total", "blocks")

@dataclass(frozen=True)
class SimulationBatch:
    game_id: str
    n_sims: int
    seed: int
    model_version: str
    model: FrozenModelRef
    runtime: dict[str, Any]
    player_ids: tuple[str, ...]
    player_team: dict[str, str]
    team_ids: tuple[str, ...]
    player_stats: dict[str, np.ndarray]
    team_stats: dict[str, np.ndarray]
    winner: np.ndarray
    status: str
    state_hash: str

    def __post_init__(self):
        if self.n_sims <= 0 or len(self.winner) != self.n_sims:
            raise ValueError("invalid simulation shape")
        if set(self.player_team) != set(self.player_ids):
            raise ValueError("player identity axis mismatch")
        for name, values in {**self.player_stats, **self.team_stats}.items():
            if np.asarray(values).shape[0] != self.n_sims:
                raise ValueError(f"stat axis mismatch for {name}")
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError(f"nonfinite simulation stat {name}")
        if set(np.unique(self.winner)) - {"HOME", "AWAY", "TIE"}:
            raise ValueError("invalid winner value")

    def __eq__(self, other):
        if not isinstance(other, SimulationBatch):
            return NotImplemented
        if (self.game_id, self.n_sims, self.seed, self.model_version, self.player_ids,
            self.player_team, self.team_ids, self.status, self.state_hash) != (
            other.game_id, other.n_sims, other.seed, other.model_version, other.player_ids,
            other.player_team, other.team_ids, other.status, other.state_hash):
            return False
        return (self.model == other.model and self.runtime == other.runtime and
                np.array_equal(self.winner, other.winner) and
                all(np.array_equal(self.player_stats[k], other.player_stats[k]) for k in self.player_stats) and
                all(np.array_equal(self.team_stats[k], other.team_stats[k]) for k in self.team_stats))

    def player(self, player_id: str, stat: str) -> np.ndarray:
        if player_id not in self.player_team:
            raise KeyError(player_id)
        if stat not in self.player_stats:
            raise KeyError(stat)
        return np.asarray(self.player_stats[stat])[:, self.player_ids.index(player_id)]

    def team(self, team_id: str, stat: str) -> np.ndarray:
        if team_id not in self.team_ids:
            raise KeyError(team_id)
        if stat not in self.team_stats:
            raise KeyError(stat)
        return np.asarray(self.team_stats[stat])[:, self.team_ids.index(team_id)]

def _state_hash(pregame: PregameState) -> str:
    payload = pregame.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _model_ref(pregame: PregameState) -> FrozenModelRef:
    raw = hashlib.sha256(b"SPORTS_NOVA_V3_ARCH_1:drive_block:default:research").hexdigest()
    return FrozenModelRef(version=MODEL_VERSION, sha256=raw,
        training_available_through=pregame.as_of,
        calibration_available_through=pregame.as_of)

def _feature(obj, name: str, default: float) -> float:
    for f in getattr(obj, "features", ()):
        if f.name == name and f.value is not None:
            return float(f.value)
    return default

def _qb_shares(pregame: PregameState, team_id: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Causal per-QB attempt-share distribution for one team.

    Built only from prior-game usage (pass_rate is populated by the runner
    from prior games only). A single active QB gets share 1.0; multiple
    active QBs split proportional to recent usage instead of one player
    winning the whole game's attempts.
    """
    qbs = [p for p in pregame.players if p.team_id == team_id and p.position == "QB" and p.availability != "OUT"]
    if not qbs:
        return (), np.zeros(0)
    # SN3_V10_QB_RECENCY_GATE_FIX (ATTEMPTED, REVERTED): the hard `<= min_gap`
    # eligibility filter was replaced with a parameter-free continuous decay
    # (1/(1+gap)) multiplied into the volume score, to let a QB who sat out
    # only the single most recent prior game still compete on recent volume.
    # Measured on the V9 60-game/120-team-game cohort: this FAILED empirically
    # -- the per-QB "pass_rate" volume feature (recent attempts/game, not a
    # 0-1 fraction) spans a wide dynamic range (~5 to ~150 in this cohort), so
    # 1/(1+gap) decays far too slowly to suppress a high-volume gap>=1
    # candidate against a genuine low-volume gap=0 current starter. Result:
    # only 1/12 originally gap-filter-caused wrong-starter team-games was
    # actually corrected, while 8 team-games that were CORRECT under the
    # original hard filter flipped to wrong (a high-volume stale QB now
    # outweighs the real, lower-volume current starter) -- net identity
    # accuracy fell from 104/120 to 97/120 team-games, and QB attempt bias
    # worsened from -3.59 to -5.13 (yards MAE 74.2 -> 79.4). Reverted to the
    # original hard min-gap filter below; see
    # SPORTS_NOVA_V10_QB_RECENCY_GATE_FIX_REPORT.json for the full comparison.
    # Recency gates eligibility before volume weighs in: a QB who has not
    # appeared in the team's most recent known prior game is not a live
    # competitor for the start, however much cumulative volume he has from
    # earlier in the lookback window (a since-benched/former starter should
    # not outrank the QB who just took the job). Ties (true committees, or no
    # recency signal at all) fall through to the existing volume-based split.
    gaps = {p.player_id: _feature(p, "games_since_last_team_game", 0.0) for p in qbs}
    min_gap = min(gaps.values())
    qbs = [p for p in qbs if gaps[p.player_id] <= min_gap]
    ids = tuple(sorted(p.player_id for p in qbs))
    by_id = {p.player_id: p for p in qbs}
    # Weight by recent *volume* (rate x games observed), not rate alone: a
    # one-game spot starter and an 8-game full-time starter can share the same
    # attempts-per-game rate but should not receive equal shares.
    volume = np.asarray([max(_feature(by_id[pid], "pass_rate", 0.0), 0.0) *
                         max(by_id[pid].uncertainty.effective_sample_size, 0.0) for pid in ids], dtype=float)
    if volume.sum() <= 0:
        volume = np.ones(len(ids))
    # A team almost always has a single clear starter (single-QB games should
    # have share ~1.0). Linear proportional-to-volume splitting understates
    # this: two candidates with a 4:1 real usage gap still land near 80/20
    # rather than the near-winner-take-all reality. Raise to a power before
    # normalizing to sharpen toward the dominant recent starter.
    volume = volume ** 2.5
    return ids, volume / volume.sum()

def _primary_qb(pregame: PregameState, team_id: str):
    """The single QB the engine would credit as starter, for reporting only."""
    ids, shares = _qb_shares(pregame, team_id)
    if not ids:
        return None
    winner_id = ids[int(np.argmax(shares))]
    return next(p for p in pregame.players if p.player_id == winner_id)

def _inc(mapping: dict[str, dict[str, int]], pid: str, stat: str, value: int) -> None:
    mapping[pid][stat] += int(value)

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
    # draw_environment already draws one home/away "recent_form"-centered
    # efficiency shock per simulation but nothing previously consumed it, so
    # every reception/carry was an independent normal draw. Summing ~20
    # independent draws per game collapses the across-simulation spread of a
    # team's season total by root-n (measured: sim pass-yards SD ~25 vs
    # realized ~79 on matched-QB rows) because real per-game yardage swings
    # together -- a team has a good or bad day, it does not roll independently
    # for every catch. Turn the existing per-sim draw into a single
    # game-level multiplier shared by every yardage sample for that team this
    # simulation, instead of adding a new fitted parameter.
    #
    # SN3_SPORTS_V6 fix: env.home_efficiency/away_efficiency are already drawn
    # as Normal(recent_form, .35) (see environment.draw_environment), so
    # subtracting recent_form back out here (as the original V4 expression
    # did) cancelled it exactly, leaving eff_mult == 1.0 + Normal(0, .35) for
    # every team regardless of recent_form's value -- recent_form had zero
    # effect on simulated yardage no matter how well it separated teams.
    # Consume the draw directly so a team's recent quality actually shifts
    # this multiplier's center; the shared-per-simulation mechanism itself
    # (the reason this is computed once per team here, not once per sample)
    # is unchanged from V4.
    eff_mult = {
        pregame.home.team_id: float(np.clip(1.0 + env.home_efficiency, 0.3, 2.5)),
        pregame.away.team_id: float(np.clip(1.0 + env.away_efficiency, 0.3, 2.5)),
    }
    player_by_id = {p.player_id: p for p in pregame.players}
    qb_shares = {tid: _qb_shares(pregame, tid) for tid in team_ids}
    # One epistemic draw per team per simulation of how the season-long share
    # actually manifests this game; each drive then independently samples the
    # quarterback under that path.  A single active QB keeps this ~deterministic
    # (share 1.0); genuine multi-QB usage varies game-to-game and drive-to-drive
    # instead of one player winning 100% of every block forever.
    qb_draw_probs = {}
    for tid, (ids, shares) in qb_shares.items():
        qb_draw_probs[tid] = (ids, rng.dirichlet(np.maximum(shares * 80.0, 1e-6)) if len(ids) else shares)
    # SN3_SPORTS_V7: the loop bound and the elapsed-time cap below used to be
    # the actual block-count mechanism (24 blocks always ran, since neither
    # this bound nor the 90s cap below could ever let the clock reach zero
    # within 24 iterations -- see SN3_SPORTS_V6_REPORT.json MECHANISM_CHECKS
    # for the proof). 60 is a generous, non-binding safety net (historical
    # combined blocks/game on NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet:
    # mean=23.62, sd=3.62, observed max=39 across 7,273 games, 1999-2025) --
    # the `if state.seconds_remaining <= 0: break` below is now the actual
    # governing mechanism, as the surrounding code already intended.
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
        # Designed rushes are allocated by role; scrambles remain credited to
        # the passer and are added to team rush attempts below.
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
            # Per-catch yards use the receiver's own prior yards_per_reception
            # (already computed causally in state-building, same as catch_rate
            # just above) rather than the single global mean/sd every receiver
            # on every team shared before -- a possession slot receiver and a
            # deep threat do not have the same reception-yardage distribution.
            ypr_mean = _feature(receiver, "yards_per_reception", params.receiving_yards_mean)
            rec_yards = sample_compound_signed(rec, ypr_mean * eff_mult[offense], params.receiving_yards_sd, rng)
            _inc(stats, pid, "receptions", rec)
            _inc(stats, pid, "receiving_yards", rec_yards)
            if qb is not None:
                _inc(stats, qb.player_id, "pass_yards", rec_yards)
            team[offense]["pass_yards"] += rec_yards
            # SN3_SPORTS_V6 Q3: team-specific recent scoring propensity
            # (pass_td_rate) replaces the single global params.td_rate=.045
            # constant every team/game previously shared -- params comes from
            # fit_distributions([], ...) (simulate_game always passes an
            # empty row list), so params.td_rate never varied by team. Falls
            # back to params.td_rate when the feature is absent (e.g. states
            # built by callers other than make_state).
            pass_td_rate = _feature(offense_team, "pass_td_rate", params.td_rate)
            td_count = int(rng.binomial(rec, np.clip(pass_td_rate, .001, .2)))
            if td_count:
                _inc(stats, pid, "receiving_tds", td_count)
                _inc(stats, pid, "scored_tds", td_count)
                if qb is not None:
                    _inc(stats, qb.player_id, "pass_tds", td_count)
                block_points += 6 * td_count
        for pid, carries in alloc.carry_counts.items():
            _inc(stats, pid, "rush_attempts", carries)
            rush_yards = sample_compound_signed(carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng)
            _inc(stats, pid, "rush_yards", rush_yards)
            team[offense]["rush_yards"] += rush_yards
            # Same SN3_SPORTS_V6 Q3 fix as the passing-TD draw above, using
            # the team's own recent rush_td_rate instead of the global
            # params.td_rate * .65 fixed ratio assumption.
            rush_td_rate = _feature(offense_team, "rush_td_rate", params.td_rate * .65)
            rush_td = int(rng.binomial(carries, np.clip(rush_td_rate, .0005, .15)))
            if rush_td:
                _inc(stats, pid, "rush_tds", rush_td)
                _inc(stats, pid, "scored_tds", rush_td)
                block_points += 6 * rush_td
        if alloc.residual_carries:
            team[offense]["rush_yards"] += sample_compound_signed(
                alloc.residual_carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng)
        if block_points == 0 and selection.plays >= 3 and rng.random() < params.fg_rate:
            block_points = 3
        # SN3_SPORTS_V7: 90 was an undocumented, un-evidenced cap that made
        # this the real (and wrong) block-terminating mechanism instead of
        # the game clock -- see the loop-bound comment above. plays is
        # already capped at 12 (draw_block_volume) and env.pace_seconds at
        # 36 (draw_environment), so 500 is a defensive numerical bound (true
        # max is 12*36=432) that never actively constrains a legitimate draw.
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
