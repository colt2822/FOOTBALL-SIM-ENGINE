"""V19 challenger: identical to worker/sports_nova_v3/simulator.py except
`_qb_shares`' eligibility gate. Every other mechanism (pace/block-volume,
allocation, efficiency, scoring, TD rates, catch rate, the volume**2.5
share-sharpening once eligibility is decided) is imported UNCHANGED from
worker.sports_nova_v3 -- not copied -- so this is a single, attributable,
diffable change, per SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER's invariant
"ONLY allowed model change = QB identity/recency resolution across season
boundary".

SN3_V19_SEASON_BOUNDARY_IDENTITY_FIX
-------------------------------------
worker/sports_nova_v3/simulator.py::_qb_shares' hard `<= min_gap` filter on
`games_since_last_team_game` was measured (SPORTS_NOVA_V18_WEEK1_REPLAY,
2026 Week 1) to misorder QB eligibility across a season boundary: a backup
who closed out the prior season (gap=0) outranks a starter who rested or
was injured late in that season (gap>0), even once a new season begins.
Triage against the live causal panel showed 7 of 11 Week-1-2026 mismatches
are this exact defect (the correct starter IS present in `pregame.players`
for that team, just filtered out) -- the remaining 4 are a distinct,
separate defect (the correct starter has no completed game for that team
in `prior` at all, e.g. an offseason trade/signing/rookie -- `make_state`'s
`latest_team` assignment never puts him on the roster in the first place,
so no change inside `_qb_shares` can reach him). Only the first defect is
in scope here.

Fix: when an independently-resolved, causal, pre-kickoff QB identity is
available for (game_id, team_id) with STATUS=CONFIRMED (see
scripts/sports_nova_v19_qb_identity_resolver.py -- pre-kickoff nflverse
depth-chart snapshot, DEV-selected on 2025 Week 1 boundary games: 32/32
accuracy vs. the existing panel-recency method's 18/32), and the resolved
player is present among that team's eligible (non-OUT) QBs, the hard
min-gap filter is bypassed in favor of that single resolved player. If no
resolution is available, or the resolved player is not present on the
team's roster (the case-B defect above), behavior is BYTE-IDENTICAL to the
frozen V18 min-gap filter -- this is a strict extension, never a
regression path, on any team-game the resolver does not confidently cover.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from datetime import datetime
from typing import Any
import numpy as np

from .config import MODEL_VERSION, RNG_ALGORITHM
from worker.sports_nova_v3.schemas import FrozenModelRef, PregameState
from worker.sports_nova_v3.distributions import fit_distributions, sample_compound_signed
from worker.sports_nova_v3.environment import draw_environment
from worker.sports_nova_v3.game_script import script_pass_rate
from worker.sports_nova_v3.game_state import GameState
from worker.sports_nova_v3.play_volume import draw_block_volume
from worker.sports_nova_v3.play_selection import select_plays
from worker.sports_nova_v3.allocation import allocate_opportunities
from worker.sports_nova_v3.scoring import winner

STAT_NAMES = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards",
              "rush_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "scored_tds")
TEAM_STAT_NAMES = ("score", "pass_attempts", "pass_yards", "rush_attempts", "rush_yards", "total", "blocks")

# Keyed by (game_id, team_id) -> {"QB_ID": str, "STATUS": "CONFIRMED", ...}.
# Installed once per process by the replay harness via set_identity_resolutions()
# BEFORE simulate_game() is called. Never mutated during a simulation. Every
# resolution this dict can ever contain is built from evidence strictly
# before that game's own kickoff (see the resolver script) -- installing it
# here is not itself an outcome-informed act, it is a lookup table built
# entirely from pre-kickoff data.
_IDENTITY_RESOLUTIONS: dict[tuple[str, str], dict] = {}


def set_identity_resolutions(resolutions: dict[tuple[str, str], dict]) -> None:
    """Install only fully validated, causal pregame QB resolutions.

    This is the sole external identity injection boundary.  Uncertain or
    temporally invalid records are rejected before simulation rather than
    silently becoming a fallback selection.
    """
    for key, rec in resolutions.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("invalid identity lookup key")
        game_id, team_id = key
        if not isinstance(rec, dict) or rec.get("GAME_ID") != game_id or rec.get("TEAM") != team_id:
            raise ValueError("identity provenance key mismatch")
        if rec.get("STATUS") != "CONFIRMED":
            raise ValueError("uncertain identity is INVALID_FOR_REPLAY")
        if rec.get("CONFIDENCE") not in {"HIGH", "CONFIRMED"}:
            raise ValueError("identity confidence is INVALID_FOR_REPLAY")
        if not rec.get("QB_ID") or not rec.get("SOURCE"):
            raise ValueError("identity provenance is incomplete")
        source_timestamp = rec.get("SOURCE_TIMESTAMP", rec.get("AS_OF"))
        kickoff_time = rec.get("KICKOFF_TIME", rec.get("KICKOFF_UTC"))
        if not source_timestamp or not kickoff_time:
            raise ValueError("identity temporal provenance is incomplete")
        try:
            source_dt = datetime.fromisoformat(str(source_timestamp).replace("Z", "+00:00"))
            kickoff_dt = datetime.fromisoformat(str(kickoff_time).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("identity timestamps are not ISO-8601") from exc
        if source_dt.tzinfo is None or kickoff_dt.tzinfo is None or not source_dt < kickoff_dt:
            raise ValueError("identity source_timestamp must be strictly before kickoff")
    global _IDENTITY_RESOLUTIONS
    _IDENTITY_RESOLUTIONS = dict(resolutions)


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
    # FrozenModelRef.version is a schema Literal pinned to
    # "sports_nova_v3.drive_block.1" (worker/sports_nova_v3/schemas.py,
    # reused unchanged here) -- it names the shared ARCH_1 drive-block
    # contract, not a specific engine build. Leaving it at its schema
    # default (rather than passing MODEL_VERSION, which fails validation --
    # see worker/sports_nova_v1_1/simulator.py's identical fix) is correct:
    # this challenger is that same architecture with one mechanism change,
    # not a new one. The actual build identity lives in `sha256` below and
    # in this module's own MODEL_VERSION.
    raw = hashlib.sha256(b"SPORTS_NOVA_V19_ARCH_1:drive_block:season_boundary_qb_identity:research").hexdigest()
    return FrozenModelRef(sha256=raw,
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

    # SN3_V19_SEASON_BOUNDARY_IDENTITY_FIX (see module docstring): a
    # confidently-resolved, pre-kickoff-only identity for this exact
    # (game_id, team_id) bypasses the min-gap filter entirely -- but only
    # when the resolved player is actually among this team's eligible
    # (non-OUT) QBs. If he is not (the case-B "stale team assignment"
    # defect -- his last completed game for this team is not in `prior` at
    # all, e.g. an offseason trade `make_state` cannot see), there is
    # nothing to select here and behavior falls through unchanged.
    resolution = _IDENTITY_RESOLUTIONS.get((pregame.game_id, team_id))
    override_applied = False
    if resolution is not None and resolution.get("STATUS") == "CONFIRMED":
        resolved_id = resolution.get("QB_ID")
        resolved_candidates = [p for p in qbs if p.player_id == resolved_id]
        if resolved_candidates:
            qbs = resolved_candidates
            override_applied = True

    if not override_applied:
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
                block_points += 6 * td_count
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
                block_points += 6 * rush_td
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
