"""V28 simulator: a fork of `worker.sports_nova_v23.simulator._run_one` with a possession-terminal scoring boundary.

Same main RNG stream (PCG64DXSM, SeedSequence([seed, sim])) and same statement order for everything that produces football state (environment, volume, play
selection, opportunity allocation, QB draw, yardage).  Every scoring draw (TD / scorer / PAT / FG / OT coin) comes from a SEPARATE stream
SeedSequence([seed, sim, SCORE_TAG]) so the football-state stream is not consumed by the score decision.

Entry points
  simulate_scoring(pregame_state, n, seed, model_version, ...)   scoring boundary alone (historical pilot; no roster layer)
  simulate_game(raw_state, n, seed, model_version, *, roster, completion)   full V26 -> V25 -> V28 pipeline (production path)
Both take an optional `perturb` (counterfactual yardage / explosive-play hooks used only by the score-world coherence tests; identity by default) and
`want_blocks` (observer rows; never changes the outputs).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from worker.sports_nova_v23 import simulator as _v23
from worker.sports_nova_v23.simulator import SimulationBatch, STAT_NAMES, TEAM_STAT_NAMES, _model_ref, _state_hash
from worker.sports_nova_v21.config import PAT_MAKE_RATE
from worker.sports_nova_v25_role_aware import simulator as _v25
from worker.sports_nova_v25_role_aware.config import (
    DOUBTFUL_POLICY_HASH, ELIGIBILITY_POLICY_HASH, MODEL_VERSION as V25_MODEL_VERSION, REDISTRIBUTION_POLICY_HASH, ROLE_POLICY_HASH)
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from worker.sports_nova_v26_active_skill_state import simulator as _v26
from worker.sports_nova_v26_active_skill_state.completion import CompletionResult
from worker.sports_nova_v26_active_skill_state.config import COMPLETION_POLICY_HASH
from worker.sports_nova_v27_causal_fg_rate import causal_fg as fg27
from worker.sports_nova_v27_causal_fg_rate.causal_fg import CausalFGError
from worker.sports_nova_v27_causal_fg_rate.config import CAUSAL_FG_POLICY_HASH, MODEL_VERSION as V27_MODEL_VERSION
from worker.sports_nova_v3.distributions import fit_distributions
from . import hazard as hz_mod
from .config import (TEAM_MULT_CLIP, EXTRA_TEAM_STATS, HAZARD_POLICY_HASH, OT_MAX_BLOCKS, OT_SECONDS_POST, OT_SECONDS_REG, OVERTIME_POLICY_HASH, RNG_ALGORITHM, SCORE_TAG,
                     TD_ALLOCATION_POLICY_HASH, V23_MODEL_VERSION, VERSIONS, is_postseason, variant_of)

prepare_state = _v26.prepare_state

BLOCK_FIELDS = ("sim", "block", "off", "sec_before", "pre_h", "pre_a", "plays", "pass_att", "designed", "scrambles", "targets", "rec", "pass_yds", "rush_yds",
                "yds", "p_td", "td", "td_pass", "td_rush", "td_qb_rush", "td_unattr", "fg_elig", "fg", "pat", "pts", "ot")
BI = {k: i for i, k in enumerate(BLOCK_FIELDS)}
ALL_TEAM_STATS = tuple(TEAM_STAT_NAMES) + tuple(EXTRA_TEAM_STATS)


@dataclass(frozen=True)
class Perturb:
    """Counterfactual hooks for the score-world coherence tests.  Identity by default (production never sets it)."""
    yard_scale: float = 1.0
    explosive_cap: float | None = None      # per-opportunity yardage draw is capped (removes explosive plays); same number of RNG draws
    only_team: str | None = None            # apply only while this team_id is on offense (asymmetric counterfactuals for the winner test)


IDENTITY = Perturb()


def _yards(count: int, mean: float, sd: float, rng, pt: Perturb) -> int:
    """worker.sports_nova_v3.distributions.sample_compound_signed, plus the identity-by-default perturbation hooks (identical draws)."""
    if count <= 0:
        return 0
    d = rng.normal(float(mean), max(float(sd), 1e-6), size=int(count))
    if pt.explosive_cap is not None:
        d = np.minimum(d, pt.explosive_cap)
    tot = int(np.rint(d).sum())
    return tot if pt.yard_scale == 1.0 else int(round(tot * pt.yard_scale))


def run_one(pregame, sim_id: int, seed: int, params, hz, fg_rate: float, *, overtime: bool, postseason: bool, perturb: Perturb = IDENTITY,
            script_trace=None, sink=None, eff_kappa: float = 1.0, team_mult: bool = False, clock_scale: float = 1.0, eff_slope: float = 1.0):
    m = _v23
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), int(sim_id)])))
    srng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), int(sim_id), SCORE_TAG])))
    player_ids = sorted(p.player_id for p in pregame.players)
    stats = {pid: {name: 0 for name in STAT_NAMES} for pid in player_ids}
    team_ids = (pregame.home.team_id, pregame.away.team_id)
    team = {tid: {name: 0 for name in ALL_TEAM_STATS} for tid in team_ids}
    state = m.GameState(game_id=pregame.game_id, simulation_id=sim_id, block_index=0, seconds_remaining=3600, home_score=0, away_score=0,
                        possession=pregame.home.team_id, field_position=25)
    env = m.draw_environment(pregame, rng, epistemic_id=sim_id)
    if eff_kappa != 1.0 or eff_slope != 1.0:   # shrink the noise around recent_form (kappa) and/or the recent_form signal itself (slope); same draws consumed
        hf, af = m._feature(pregame.home, "recent_form", 0.), m._feature(pregame.away, "recent_form", 0.)
        env = dataclasses.replace(env, home_efficiency=eff_slope * hf + eff_kappa * (env.home_efficiency - hf), away_efficiency=eff_slope * af + eff_kappa * (env.away_efficiency - af))
    eff_mult = {pregame.home.team_id: float(np.clip(1.0 + env.home_efficiency, 0.3, 2.5)),
                pregame.away.team_id: float(np.clip(1.0 + env.away_efficiency, 0.3, 2.5))}
    player_by_id = {p.player_id: p for p in pregame.players}
    qb_shares = {tid: m._qb_shares(pregame, tid) for tid in team_ids}
    qb_draw_probs = {}
    for tid, (ids, shares) in qb_shares.items():
        qb_draw_probs[tid] = (ids, rng.dirichlet(np.maximum(shares * 80.0, 1e-6)) if len(ids) else shares)

    def block(state, *, walkoff_if_td: bool, script_state, in_ot: bool):
        """One possession.  Returns (points, elapsed_seconds).  Football state first (main stream), then ONE terminal scoring decision (score stream)."""
        offense = state.possession
        offense_team = pregame.home if offense == pregame.home.team_id else pregame.away
        pt = perturb if perturb.only_team in (None, offense) else IDENTITY
        team[offense]["blocks"] += 1
        plays = m.draw_block_volume(state, env, rng)
        selection = m.select_plays(plays, m.script_pass_rate(pregame, script_state, env), rng,
                                   sack_rate=m._feature(offense_team, "sack_rate", .065), scramble_rate=m._feature(offense_team, "scramble_rate", .07))
        alloc = m.allocate_opportunities(pregame, offense, selection.pass_attempts, selection.designed_rushes, rng)
        ids, probs = qb_draw_probs[offense]
        qb = player_by_id[ids[int(rng.choice(len(ids), p=probs))]] if len(ids) else None
        b_pass = b_rush = b_targets = b_rec = 0
        if qb is not None:
            m._inc(stats, qb.player_id, "pass_attempts", selection.pass_attempts)
            m._inc(stats, qb.player_id, "rush_attempts", selection.scrambles)
            scramble_yards = _yards(selection.scrambles, 4.0 * eff_mult[offense], 5.0, rng, pt)
            m._inc(stats, qb.player_id, "rush_yards", scramble_yards)
            team[offense]["rush_yards"] += scramble_yards
            b_rush += scramble_yards
        team[offense]["pass_attempts"] += selection.pass_attempts
        team[offense]["rush_attempts"] += selection.rush_attempts
        rec_by_pid: dict = {}
        for pid, targets in alloc.target_counts.items():
            m._inc(stats, pid, "targets", targets)
            receiver = next(p for p in pregame.players if p.player_id == pid)
            catch_rate = np.clip(m._feature(receiver, "catch_rate", .64), .05, .95)
            rec = int(rng.binomial(targets, catch_rate))
            ypr_mean = m._feature(receiver, "yards_per_reception", params.receiving_yards_mean)
            rec_yards = _yards(rec, ypr_mean * eff_mult[offense], params.receiving_yards_sd, rng, pt)
            m._inc(stats, pid, "receptions", rec)
            m._inc(stats, pid, "receiving_yards", rec_yards)
            if qb is not None:
                m._inc(stats, qb.player_id, "pass_yards", rec_yards)
            team[offense]["pass_yards"] += rec_yards
            b_pass += rec_yards
            b_targets += targets
            b_rec += rec
            if rec > 0:
                rec_by_pid[pid] = rec
        carries_by_pid: dict = {}
        for pid, carries in alloc.carry_counts.items():
            m._inc(stats, pid, "rush_attempts", carries)
            rush_yards = _yards(carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng, pt)
            m._inc(stats, pid, "rush_yards", rush_yards)
            team[offense]["rush_yards"] += rush_yards
            b_rush += rush_yards
            if carries > 0:
                carries_by_pid[pid] = carries
        if alloc.residual_carries:
            resid = _yards(alloc.residual_carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng, pt)
            team[offense]["rush_yards"] += resid
            b_rush += resid
        # ------------------------------------------------------------------ terminal scoring decision (score stream only)
        yards = b_pass + b_rush
        diff_off = (state.home_score - state.away_score) * (1 if offense == pregame.home.team_id else -1)
        hz_sec = 1500 if in_ot else state.seconds_remaining         # OT is not an end-of-half clock situation
        p_td = float(hz.p_td(yards, selection.plays, diff_off, hz_sec))
        pass_rate = float(np.clip(m._feature(offense_team, "pass_td_rate", params.td_rate), .001, .2))
        rush_rate = float(np.clip(m._feature(offense_team, "rush_td_rate", params.td_rate * .65), .0005, .15))
        w_pass = [(pid, rc * pass_rate) for pid, rc in sorted(rec_by_pid.items())]
        rush_w: dict = {pid: c * rush_rate for pid, c in carries_by_pid.items()}
        if qb is not None and selection.scrambles > 0:
            rush_w[qb.player_id] = rush_w.get(qb.player_id, 0.0) + selection.scrambles * rush_rate
        w_rush = sorted(rush_w.items())
        total_w = sum(w for _, w in w_pass) + sum(w for _, w in w_rush)
        if team_mult and total_w > 0.0:      # V23's team-specific finishing propensity (pass_td_rate / rush_td_rate) as an odds multiplier relative to the league reference
            w_ref = (sum(rc for _, rc in sorted(rec_by_pid.items())) * hz.league_pass_td_rate
                     + (sum(carries_by_pid.values()) + (selection.scrambles if qb is not None else 0)) * hz.league_rush_td_rate)
            if w_ref > 0.0:
                ratio = float(np.clip(total_w / w_ref, TEAM_MULT_CLIP[0], TEAM_MULT_CLIP[1]))
                pc = min(max(p_td, 1e-12), 1.0 - 1e-12)
                p_td = float(1.0 / (1.0 + np.exp(-(np.log(pc / (1.0 - pc)) + np.log(ratio)))))
        u_td, u_pick, u_pat, u_fg = srng.random(4)                   # ALWAYS four uniforms per possession -> common random numbers across counterfactual arms
        td = u_td < p_td
        points = td_pass = td_rush = td_qb = unattr = fg = pat = 0
        if td and total_w <= 0.0:
            td, unattr = False, 1                                    # a TD needs an attributable scorer; counted, never credited to a phantom
        if td:
            r = u_pick * total_w
            acc, pick = 0.0, None
            for kind, lst in (("P", w_pass), ("R", w_rush)):
                for pid, w in lst:
                    acc += w
                    pick = (kind, pid)
                    if acc > r:
                        break
                else:
                    continue
                break
            kind, pid = pick
            m._inc(stats, pid, "scored_tds", 1)
            if kind == "P":
                m._inc(stats, pid, "receiving_tds", 1)
                if qb is not None:
                    m._inc(stats, qb.player_id, "pass_tds", 1)
                td_pass = 1
            else:
                m._inc(stats, pid, "rush_tds", 1)
                td_rush = 1
                td_qb = int(qb is not None and pid == qb.player_id)
            pat = 0 if walkoff_if_td else int(u_pat < PAT_MAKE_RATE)
            points = 6 + pat
        fg_elig = int((not td) and selection.plays >= 3)
        if fg_elig and u_fg < fg_rate:
            points, fg = 3, 1
        t = team[offense]
        t["tds"] += td_pass + td_rush
        t["fgs"] += fg
        t["td_pass"] += td_pass
        t["td_rush"] += td_rush
        t["td_qb_rush"] += td_qb
        t["td_unattributable"] += unattr
        if sink is not None:
            sink.append((sim_id, state.block_index, 0 if offense == team_ids[0] else 1, state.seconds_remaining, state.home_score, state.away_score,
                         selection.plays, selection.pass_attempts, selection.designed_rushes, selection.scrambles, b_targets, b_rec, b_pass, b_rush, yards, p_td,
                         td_pass + td_rush, td_pass, td_rush, td_qb, unattr, fg_elig, fg, pat, points, int(in_ot)))
        elapsed = max(1, min(500, int(round(plays * env.pace_seconds * clock_scale))))
        return points, elapsed

    def step(state, points, elapsed):
        offense = state.possession
        nxt = team_ids[1] if offense == team_ids[0] else team_ids[0]
        fp = int(rng.integers(15, 86))                               # V23 draws (and never uses) the next field position; kept so the main stream is comparable
        if offense == team_ids[0]:
            return state.advance(seconds=elapsed, possession=nxt, field_position=fp, home_score=state.home_score + points)
        return state.advance(seconds=elapsed, possession=nxt, field_position=fp, away_score=state.away_score + points)

    for _ in range(60):
        if state.seconds_remaining <= 0:
            break
        script_state = state
        if script_trace is not None:                                 # CTRL replay: feed BASE's pre-block score differential to the pass-rate script only
            ph, pa = script_trace[state.block_index]
            script_state = state.model_copy(update={"home_score": int(ph), "away_score": int(pa)})
        pts, el = block(state, walkoff_if_td=False, script_state=script_state, in_ot=False)
        state = step(state, pts, el)
    reg_h, reg_a = state.home_score, state.away_score
    went_ot = 0
    if overtime and reg_h == reg_a:
        went_ot = 1
        first = team_ids[0] if srng.random() < 0.5 else team_ids[1]
        second = team_ids[1] if first == team_ids[0] else team_ids[0]
        state = state.model_copy(update={"seconds_remaining": OT_SECONDS_POST if postseason else OT_SECONDS_REG, "possession": first, "overtime": True})
        nposs = {team_ids[0]: 0, team_ids[1]: 0}
        for _k in range(OT_MAX_BLOCKS):
            if state.seconds_remaining <= 0:
                break
            offense = state.possession
            own = state.home_score if offense == team_ids[0] else state.away_score
            opp = state.away_score if offense == team_ids[0] else state.home_score
            other = second if offense == first else first
            sudden = nposs[team_ids[0]] >= 1 and nposs[team_ids[1]] >= 1
            second_poss = nposs[offense] == 0 and nposs[other] >= 1
            walkoff = sudden or (second_poss and own + 6 > opp)
            pts, el = block(state, walkoff_if_td=walkoff, script_state=state, in_ot=True)
            state = step(state, pts, el)
            nposs[offense] += 1
            if nposs[team_ids[0]] >= 1 and nposs[team_ids[1]] >= 1 and state.home_score != state.away_score:
                break
        if postseason and state.home_score == state.away_score:      # unreachable in practice (P ~ 1e-11); never leave a postseason tie silently
            raise RuntimeError("postseason overtime undecided after OT_MAX_BLOCKS")
    for j, tid in enumerate(team_ids):
        final = state.home_score if j == 0 else state.away_score
        reg = reg_h if j == 0 else reg_a
        team[tid]["score"] = final
        team[tid]["total"] = final
        team[tid]["reg_score"] = reg
        team[tid]["ot_score"] = final - reg
        team[tid]["went_ot"] = went_ot
    return stats, team, m.winner(state.home_score, state.away_score)


def _score(pregame_state, n_sims: int, seed: int, model_version: str, fg_rate: float, fg_evidence: dict, hz, *, perturb: Perturb = IDENTITY,
           script_traces=None, want_blocks: bool = False, eff_kappa: float | None = None):
    if type(n_sims) is not int or n_sims <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer n_sims and nonnegative integer seed required")
    if not isinstance(fg_rate, float) or not math.isfinite(fg_rate):
        raise CausalFGError("fg_rate must be an explicit finite float")
    var = variant_of(model_version)
    overtime = var.ot
    kappa = var.kappa if eff_kappa is None else float(eff_kappa)
    season, week = fg27.parse_game_key(pregame_state.game_id)
    post = is_postseason(season, week)
    eff_slope = hz_mod.recent_form_slope(season, week) if var.eff_shrink else 1.0
    model = _model_ref(pregame_state)
    params = fit_distributions([], pregame_state.as_of)
    params = dataclasses.replace(params, fg_rate=fg_rate)
    player_ids = tuple(sorted(p.player_id for p in pregame_state.players))
    team_ids = (pregame_state.home.team_id, pregame_state.away.team_id)
    player_team = {p.player_id: p.team_id for p in pregame_state.players}
    player_arrays = {stat: np.zeros((n_sims, len(player_ids)), dtype=np.int64) for stat in STAT_NAMES}
    team_arrays = {stat: np.zeros((n_sims, len(team_ids)), dtype=np.int64) for stat in ALL_TEAM_STATS}
    winners = np.empty(n_sims, dtype="U4")
    sink = [] if want_blocks else None
    for i in range(n_sims):
        stats, team, win = run_one(pregame_state, i, seed, params, hz, fg_rate, overtime=overtime, postseason=post, perturb=perturb,
                                   script_trace=None if script_traces is None else script_traces[i], sink=sink, eff_kappa=kappa, team_mult=var.team_mult, clock_scale=var.clock_scale, eff_slope=eff_slope)
        winners[i] = win
        for j, pid in enumerate(player_ids):
            for stat in STAT_NAMES:
                player_arrays[stat][i, j] = stats[pid][stat]
        for j, tid in enumerate(team_ids):
            for stat in ALL_TEAM_STATS:
                team_arrays[stat][i, j] = team[tid][stat]
    for values in list(player_arrays.values()) + list(team_arrays.values()) + [winners]:
        values.setflags(write=False)
    runtime = {"rng_algorithm": RNG_ALGORITHM, "numpy_version": np.__version__, "dtype": "int64", "path_order": "simulation_index_ascending", "status": params.status,
               "scoring_boundary": "possession_terminal_single_event", "score_rng_tag": SCORE_TAG, "overtime": bool(overtime), "postseason": bool(post),
               "hazard_policy_hash": HAZARD_POLICY_HASH, "td_allocation_policy_hash": TD_ALLOCATION_POLICY_HASH, "overtime_policy_hash": OVERTIME_POLICY_HASH,
               "perturbation": dataclasses.asdict(perturb), "efficiency_noise_kappa": kappa, "team_finishing_multiplier": bool(var.team_mult), "clock_scale": float(var.clock_scale), "recent_form_slope": float(eff_slope)}
    runtime.update(fg_evidence)
    runtime.update(hz.evidence())
    batch = SimulationBatch(pregame_state.game_id, n_sims, seed, model_version, model, runtime, player_ids, player_team, team_ids, player_arrays, team_arrays,
                            winners, "RESEARCH_GRADE_DEFAULT_PARAMETERS", _state_hash(pregame_state))
    if want_blocks:
        return batch, np.asarray(sink, dtype=np.float64).reshape(-1, len(BLOCK_FIELDS))
    return batch


def _causal(pregame_state, model_version: str, ref_quantiles=None):
    align = variant_of(model_version).align
    r = fg27.causal_fg_rate_for_game(pregame_state.game_id)
    ev = r.evidence()
    ev.update({"fg_rate_source": "CAUSAL_ESTIMATOR_A", "causal_fg_policy_hash": CAUSAL_FG_POLICY_HASH})
    hz = hz_mod.hazard_for_game(pregame_state.game_id, align=align, ref_quantiles=ref_quantiles)
    return r.rate, ev, hz


def simulate_scoring(pregame_state, n_sims: int, seed: int, model_version: str, *, ref_quantiles=None, perturb: Perturb = IDENTITY, script_traces=None,
                     want_blocks: bool = False, eff_kappa: float | None = None):
    """Scoring boundary only (no roster layer): the code the full pipeline calls."""
    variant_of(model_version)
    rate, ev, hz = _causal(pregame_state, model_version, ref_quantiles)
    return _score(pregame_state, n_sims, seed, model_version, rate, ev, hz, perturb=perturb, script_traces=script_traces, want_blocks=want_blocks, eff_kappa=eff_kappa)


def simulate_scoring_forced_fg_rate(pregame_state, n_sims: int, seed: int, model_version: str, *, fg_rate: float, ref_quantiles=None,
                                    perturb: Perturb = IDENTITY, want_blocks: bool = False):
    """REGRESSION HARNESS ONLY: explicit fg_rate, no default (mirrors V27's forced entry point)."""
    align = variant_of(model_version).align
    ev = {"fg_rate": float(fg_rate), "fg_rate_source": "FORCED_REGRESSION_HARNESS", "causal_fg_policy_hash": CAUSAL_FG_POLICY_HASH}
    hz = hz_mod.hazard_for_game(pregame_state.game_id, align=align, ref_quantiles=ref_quantiles)
    return _score(pregame_state, n_sims, seed, model_version, float(fg_rate), ev, hz, perturb=perturb, want_blocks=want_blocks)


def _pipeline(raw_state, n_sims, seed, model_version, roster: RosterSnapshot, completion: CompletionResult, ref_quantiles) -> SimulationBatch:
    variant_of(model_version)
    if completion.raw_state_hash != _state_hash(raw_state):
        raise ValueError("completion was built from a different raw state")
    repaired, audit = _v25.prepare_state(completion.state, roster)
    rate, ev, hz = _causal(repaired, model_version, ref_quantiles)
    batch = _score(repaired, n_sims, seed, model_version, rate, ev, hz)
    runtime = dict(batch.runtime)
    runtime.update({"m1_engine_base": V23_MODEL_VERSION, "v25_layer": V25_MODEL_VERSION, "v26_layer": "sports_nova_v26 (completion)", "v27_layer": V27_MODEL_VERSION,
                    "allocation_repair": "role_aware_share_repair_then_v23_opportunity_draw_then_v28_possession_terminal_scoring",
                    "raw_state_hash": audit.raw_state_hash, "repaired_state_hash": audit.repaired_state_hash, "eligibility_policy_hash": ELIGIBILITY_POLICY_HASH,
                    "role_policy_hash": ROLE_POLICY_HASH, "redistribution_policy_hash": REDISTRIBUTION_POLICY_HASH, "doubtful_policy_hash": DOUBTFUL_POLICY_HASH,
                    "roster_week": roster.roster_week, "completion_policy_hash": COMPLETION_POLICY_HASH, "raw_input_state_hash": completion.raw_state_hash,
                    "completed_state_hash": completion.completed_state_hash, "completion_changed": completion.changed})
    return SimulationBatch(batch.game_id, batch.n_sims, batch.seed, model_version, batch.model, runtime, batch.player_ids, batch.player_team, batch.team_ids,
                           batch.player_stats, batch.team_stats, batch.winner, batch.status, completion.raw_state_hash)


def simulate_game(raw_state, n_sims: int, seed: int, model_version: str, *, roster: RosterSnapshot, completion: CompletionResult, ref_quantiles=None) -> SimulationBatch:
    """Production path.  RosterSnapshot and CompletionResult are REQUIRED (as V26/V27); FG rate is ALWAYS the causal estimate; hazard is ALWAYS the causal fit."""
    return _pipeline(raw_state, n_sims, seed, model_version, roster, completion, ref_quantiles)
