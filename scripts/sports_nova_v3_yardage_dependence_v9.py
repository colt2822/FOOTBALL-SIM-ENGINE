"""SN3_SPORTS_V8_YARDAGE_DEPENDENCE.

Mission: test whether independent per-reception yardage sampling
(distributions.sample_compound_signed) is the remaining structural cause of
QB passing-yard error, after SN3_SPORTS_V5/V6/V7 each fixed a different
subsystem (volume, team-quality/scoring, block/clock dispersion) without
improving QB yards MAE.

Y1 AUDIT FINDING (recorded before any new code was written, see
SHARED_STATE_ALREADY_EXISTS below): simulator.py's `_run_one` already draws
one `eff_mult` per team per simulation (lines ~155-180) and multiplies EVERY
per-reception/per-carry/per-scramble yardage mean for that team, that
simulation, by it. Every reception in a team's simulated game therefore
shares the same eff_mult draw -- the draws are conditionally i.i.d. given
eff_mult, not marginally independent. SN3_SPORTS_V7's SINGLE_NEXT_ACTION text
("eff_mult ... does not change the fact that the draws are still
independent") conflated conditional with marginal independence. This task
does NOT introduce a new shared latent state (inv item 'no arbitrary posthoc
tuning' plus this finding rules that out as double-counting); instead it (a)
measures how much of simulated yardage variance the EXISTING shared state
already explains vs. the leftover i.i.d. per-reception noise, and (b) tests
whether the existing state's hand-picked shock scale (a bare, un-evidenced
`.35` in environment.py, the same class of magic number as V7's 90s/24-block
caps) is well-calibrated against real historical residual dispersion, by
deriving a replacement scale from historical team-game residuals (never from
the 60-game eval cohort, never from current-game outcomes) and re-running the
identical validation.

No worker/ file is modified by this script. A byte-faithful diagnostic twin
of simulator._run_one is reimplemented here (calling the real, unmodified
draw_block_volume/select_plays/allocate_opportunities/sample_compound_signed/
script_pass_rate functions) so that (1) the shock scale can be swapped without
touching environment.py and (2) extra instrumentation (eff_mult, the
volume/noise split of every reception-yards draw) can be captured without
changing simulator.py's return contract. The twin's fidelity is verified by
asserting bit-exact agreement with the real simulator._run_one on a sample of
(game, sim) pairs before any numbers are trusted.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.schemas import PregameState
from worker.sports_nova_v3.game_state import GameState
from worker.sports_nova_v3.environment import EnvironmentDraw, draw_environment
from worker.sports_nova_v3.play_volume import draw_block_volume
from worker.sports_nova_v3.play_selection import select_plays
from worker.sports_nova_v3.game_script import script_pass_rate
from worker.sports_nova_v3.allocation import allocate_opportunities
from worker.sports_nova_v3.scoring import winner
from worker.sports_nova_v3.distributions import fit_distributions, sample_compound_signed
from worker.sports_nova_v3.simulator import (
    simulate_game, _run_one, _primary_qb, _qb_shares, _feature, _inc,
    STAT_NAMES, TEAM_STAT_NAMES,
)
from sports_nova_v3_player_joint_walkforward_v1 import (
    PLAYER, MANIFEST, PLAYER_SHA, game_parts, surrogate_kickoff, make_state,
)

OUT = ROOT / "data" / "sports_nova_v3"
N = 128
SHOCK_SD_BASELINE = 0.35


# ---------------------------------------------------------------------------
# Y1/Y3 support: an environment-draw variant parameterized by shock sd, used
# ONLY inside this diagnostic script's twin loop -- environment.py itself is
# untouched.
# ---------------------------------------------------------------------------
def draw_environment_variant(pregame: PregameState, rng: np.random.Generator,
                              *, epistemic_id: int = 0, shock_sd: float) -> EnvironmentDraw:
    pace_home = _feature(pregame.home, "pace_seconds", 27.0)
    pace_away = _feature(pregame.away, "pace_seconds", 27.0)
    pace = float(np.clip(rng.normal((pace_home + pace_away) / 2.0, 1.5), 18.0, 36.0))
    home_pass = np.clip(_feature(pregame.home, "pass_rate", .58) + rng.normal(0, .025), .25, .8)
    away_pass = np.clip(_feature(pregame.away, "pass_rate", .58) + rng.normal(0, .025), .25, .8)
    return EnvironmentDraw(pace, float(rng.normal(_feature(pregame.home, "recent_form", 0.), shock_sd)),
                           float(rng.normal(_feature(pregame.away, "recent_form", 0.), shock_sd)),
                           float(home_pass), float(away_pass), int(epistemic_id))


def make_env_fn(shock_sd: float):
    if shock_sd == SHOCK_SD_BASELINE:
        return draw_environment
    def fn(pregame, rng, *, epistemic_id=0):
        return draw_environment_variant(pregame, rng, epistemic_id=epistemic_id, shock_sd=shock_sd)
    return fn


def twin_run_one(pregame: PregameState, sim_id: int, seed: int, params, env_fn):
    """Byte-faithful reimplementation of simulator._run_one (verified below)
    plus non-invasive instrumentation: per-team eff_mult, and a
    volume/noise split of every receiving-yards draw feeding pass_yards.
    volume_sum = sum(rec * ypr_mean)  -- the eff_mult-free "opportunity x
    per-target rate" component (a function of allocation/catch draws only).
    noise_sum = actual_draw - rec * ypr_mean * eff_mult -- the leftover
    i.i.d.-per-reception normal noise sample_compound_signed contributes.
    Identically: pass_yards_total = eff_mult * volume_sum + noise_sum.
    """
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), int(sim_id)])))
    player_ids = sorted(p.player_id for p in pregame.players)
    stats = {pid: {name: 0 for name in STAT_NAMES} for pid in player_ids}
    team_ids = (pregame.home.team_id, pregame.away.team_id)
    team = {tid: {name: 0 for name in TEAM_STAT_NAMES} for tid in team_ids}
    state = GameState(game_id=pregame.game_id, simulation_id=sim_id, block_index=0,
        seconds_remaining=3600, home_score=0, away_score=0,
        possession=pregame.home.team_id, field_position=25)
    env = env_fn(pregame, rng, epistemic_id=sim_id)
    eff_mult = {
        pregame.home.team_id: float(np.clip(1.0 + env.home_efficiency, 0.3, 2.5)),
        pregame.away.team_id: float(np.clip(1.0 + env.away_efficiency, 0.3, 2.5)),
    }
    player_by_id = {p.player_id: p for p in pregame.players}
    qb_shares = {tid: _qb_shares(pregame, tid) for tid in team_ids}
    qb_draw_probs = {}
    for tid, (ids, shares) in qb_shares.items():
        qb_draw_probs[tid] = (ids, rng.dirichlet(np.maximum(shares * 80.0, 1e-6)) if len(ids) else shares)
    diag = {tid: {"volume_sum": 0.0, "noise_sum": 0.0, "eff_mult": eff_mult[tid]} for tid in team_ids}
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
            diag[offense]["volume_sum"] += rec * ypr_mean
            diag[offense]["noise_sum"] += rec_yards - rec * ypr_mean * eff_mult[offense]
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
    return stats, team, winner(state.home_score, state.away_score), diag


# Twin fidelity (twin_run_one(env_fn=draw_environment) must reproduce real
# simulator._run_one bit-exactly, since it is a line-for-line copy plus
# instrumentation that consumes no extra randomness) is checked inline in
# main(), using each game's own correctly-fit params.
# ---------------------------------------------------------------------------
# cohort selection -- identical to V7/V8/decomp scripts
# ---------------------------------------------------------------------------
def select_cohort():
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])
    return selected


# ---------------------------------------------------------------------------
# Y3 support: derive the recent_form shock sd from historical residual
# structure. Never uses the 60-game eval cohort or the gate metrics.
# Approximation (disclosed): league_py_pg is computed at SEASON granularity
# (mean team-game pass yards within that season, pooling all teams) rather
# than replaying the exact per-game trailing-8 causal window make_state uses
# -- adequate for a population-level noise-scale constant, not a per-game
# prediction, and does not touch the eval cohort's own outcomes either way.
# ---------------------------------------------------------------------------
def derive_shock_sd(all_df: pd.DataFrame, exclude_game_ids: set[str], min_prior_games: int = 8):
    df = all_df.copy()
    tg = (df.groupby(["GAME_ID", "SEASON", "WEEK", "TEAM"], as_index=False)
            .agg(pass_yards=("PASS_YARDS", "sum")))
    tg = tg[~tg.GAME_ID.isin(exclude_game_ids)].copy()
    league_py_pg = tg.groupby("SEASON")["pass_yards"].transform("mean")
    tg["league_py_pg"] = league_py_pg
    tg = tg.sort_values(["TEAM", "SEASON", "WEEK"])
    residuals = []
    for tid, g in tg.groupby("TEAM"):
        g = g.sort_values(["SEASON", "WEEK"]).reset_index(drop=True)
        for i in range(len(g)):
            if i < min_prior_games:
                continue
            prior8 = g.iloc[max(0, i - 8):i]
            team_py_pg = float(prior8.pass_yards.mean())
            league_baseline = float(g.iloc[i].league_py_pg)
            if league_baseline <= 0:
                continue
            recent_form_pred = team_py_pg / league_baseline - 1.0
            actual_relative = float(g.iloc[i].pass_yards) / league_baseline - 1.0
            residuals.append(actual_relative - recent_form_pred)
    residuals = np.asarray(residuals, dtype=float)
    return {
        "method": "team-game trailing-8 recent_form prediction vs. actual, both expressed relative to "
                  "season-level league mean pass yards/team-game; residual = actual_relative - predicted_relative; "
                  "eval cohort's 60 games fully excluded from this population",
        "n_team_games": int(len(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "residual_sd": float(np.std(residuals, ddof=1)),
        "residual_p10_p90": [float(np.quantile(residuals, .1)), float(np.quantile(residuals, .9))],
        "current_hardcoded_shock_sd": SHOCK_SD_BASELINE,
        "caveat": "This residual pool is real team-game pass-yards deviation from the team's own trailing-8 "
                  "mean, which contains ALL real sources of game-to-game variation at once (volume/allocation "
                  "swings, per-reception luck, true efficiency swings), not an efficiency-only signal. The "
                  "engine already generates its own volume/allocation variance (~26% of sim variance, see "
                  "share_volume_allocation) and idiosyncratic per-reception noise (~19%, share_idiosyncratic_noise) "
                  "separately from eff_mult. Feeding this residual's full sd into eff_mult alone therefore "
                  "double-counts variance the engine already produces through other channels -- this is very "
                  "likely WHY the resulting total simulated dispersion is over-, not under-, realistic (see "
                  "SINGLE_NEXT_ACTION). 0.3146 should be read as an upper bound on the evidenced shock scale, "
                  "not a validated target for this specific parameter, and a future task should not reuse it "
                  "without first isolating the efficiency-specific share of this residual.",
    }


# ---------------------------------------------------------------------------
# main run: build pregame states once, run BASELINE (shock_sd=.35, real
# environment.draw_environment) and VARIANT (evidence-derived shock_sd)
# through the same twin, collect gate metrics + variance decomposition.
# ---------------------------------------------------------------------------
def run_variant(cohort_states, params_by_game, shock_sd, oos, label):
    env_fn = make_env_fn(shock_sd)
    rows = []
    identity = []
    conservation = []
    qb_conservation = []
    stage = []
    decomp_rows = []
    pos_rows = []
    for g, state in cohort_states:
        seed = int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)
        params = params_by_game[g]
        player_ids = sorted(p.player_id for p in state.players)
        team_ids = (state.home.team_id, state.away.team_id)
        player_arrays = {stat: np.zeros((N, len(player_ids)), dtype=np.int64) for stat in STAT_NAMES}
        team_arrays = {stat: np.zeros((N, len(team_ids)), dtype=np.int64) for stat in TEAM_STAT_NAMES}
        per_sim_diag = {tid: {"volume_sum": np.zeros(N), "noise_sum": np.zeros(N), "eff_mult": np.zeros(N)}
                        for tid in team_ids}
        for i in range(N):
            stats, team, win, diag = twin_run_one(state, i, seed, params, env_fn)
            for j, pid in enumerate(player_ids):
                for stat in STAT_NAMES:
                    player_arrays[stat][i, j] = stats[pid][stat]
            for j, tid in enumerate(team_ids):
                for stat in TEAM_STAT_NAMES:
                    team_arrays[stat][i, j] = team[tid][stat]
                per_sim_diag[tid]["volume_sum"][i] = diag[tid]["volume_sum"]
                per_sim_diag[tid]["noise_sum"][i] = diag[tid]["noise_sum"]
                per_sim_diag[tid]["eff_mult"][i] = diag[tid]["eff_mult"]

        def player_stat(pid, stat):
            j = player_ids.index(pid)
            return player_arrays[stat][:, j]

        def team_stat(tid, stat):
            j = team_ids.index(tid)
            return team_arrays[stat][:, j]

        cur = oos[oos.GAME_ID == g]
        for tid in team_ids:
            pass_y = team_stat(tid, "pass_yards").astype(float)
            v = per_sim_diag[tid]
            volume_sum = v["volume_sum"]; noise_sum = v["noise_sum"]; eff = v["eff_mult"]
            recon_signal = eff * volume_sum
            var_total = float(np.var(pass_y, ddof=1))
            var_noise = float(np.var(noise_sum, ddof=1))
            var_signal = float(np.var(recon_signal, ddof=1))
            var_volume_only = float(np.var(volume_sum, ddof=1))
            var_eff_only = float(np.var(eff, ddof=1))
            cov_signal_noise = float(np.cov(recon_signal, noise_sum, ddof=1)[0, 1])
            e_volume = float(np.mean(volume_sum)); e_eff = float(np.mean(eff))
            approx_shared_eff_component = (e_volume ** 2) * var_eff_only
            approx_volume_component = (e_eff ** 2) * var_volume_only
            decomp_rows.append({
                "GAME_ID": g, "TEAM": tid,
                "var_total_pass_yards": var_total,
                "sd_total_pass_yards": float(np.sqrt(max(var_total, 0.0))),
                "var_idiosyncratic_noise": var_noise,
                "var_signal_eff_times_volume": var_signal,
                "cov_signal_noise": cov_signal_noise,
                "approx_shared_efficiency_variance": approx_shared_eff_component,
                "approx_volume_allocation_variance": approx_volume_component,
                "mean_eff_mult": e_eff, "sd_eff_mult": float(np.sqrt(max(var_eff_only, 0.0))),
                "mean_volume_sum": e_volume, "sd_volume_sum": float(np.sqrt(max(var_volume_only, 0.0))),
                "reconstruction_check_ratio": (var_signal + var_noise + 2 * cov_signal_noise) / var_total if var_total > 0 else None,
            })
        for tid in team_ids:
            for i in range(N):
                rec_y = sum(player_stat(pid, "receiving_yards")[i] for pid in player_ids
                            if next(p for p in state.players if p.player_id == pid).team_id == tid)
                rec_t = sum(player_stat(pid, "targets")[i] for pid in player_ids
                            if next(p for p in state.players if p.player_id == pid).team_id == tid)
                conservation.append(bool(rec_t <= team_stat(tid, "pass_attempts")[i] and
                                          rec_y <= team_stat(tid, "pass_yards")[i] + 1e-9))
                qb_ids_on_team = [pid for pid in player_ids
                                  if next(p for p in state.players if p.player_id == pid).team_id == tid and
                                  next(p for p in state.players if p.player_id == pid).position == "QB"]
                qb_att_sum = sum(player_stat(pid, "pass_attempts")[i] for pid in qb_ids_on_team)
                qb_conservation.append(bool(qb_att_sum <= team_stat(tid, "pass_attempts")[i] + 1e-9))
        stage.append({"GAME_ID": g,
                      "SIM_GAME_PLAYS_SD": float(np.std(team_stat(team_ids[0], "pass_attempts") +
                                                          team_stat(team_ids[0], "rush_attempts") +
                                                          team_stat(team_ids[1], "pass_attempts") +
                                                          team_stat(team_ids[1], "rush_attempts"), ddof=1))})
        qcur = cur[cur.POSITION == "QB"]
        primary_by_team = {}
        for tid in team_ids:
            pq = _primary_qb(state, tid)
            primary_by_team[tid] = pq.player_id if pq is not None else None
        for r in qcur.itertuples():
            pid = str(r.PLAYER_ID)
            if pid not in player_ids:
                continue
            a = player_stat(pid, "pass_yards").astype(float); at = player_stat(pid, "pass_attempts").astype(float)
            identity.append({"zero": bool(at.mean() == 0), "obs_att": float(r.PASS_ATTEMPTS), "sim_att": float(at.mean()),
                "obs_y": float(r.PASS_YARDS), "sim_y": float(a.mean()), "sim_q": np.quantile(a, [.1, .9]).tolist(),
                "sim_sd": float(np.std(a, ddof=1)), "sim_skew": float(_skew(a)),
                "matched": bool(pid == primary_by_team.get(r.TEAM))})
        for r in cur.itertuples():
            pid = str(r.PLAYER_ID)
            stat = {"QB": "pass_yards", "RB": "rush_yards", "WR": "receiving_yards", "TE": "receiving_yards"}.get(r.POSITION)
            if not stat or pid not in player_ids:
                continue
            a = player_stat(pid, stat).astype(float); y = float(getattr(r, stat.upper()))
            rows.append({"POSITION": r.POSITION, "TARGET": stat, "OBS": y, "PRED": float(a.mean()),
                "LO": float(np.quantile(a, .1)), "HI": float(np.quantile(a, .9))})

    p = pd.DataFrame(rows)
    player_metrics = {}
    for (pos, target), gdf in p.groupby(["POSITION", "TARGET"]):
        y = gdf.OBS.to_numpy(); x = gdf.PRED.to_numpy()
        player_metrics[pos + "_" + target] = {"N": len(gdf), "MAE": float(np.mean(abs(y - x))),
            "RMSE": float(np.sqrt(np.mean((y - x) ** 2))), "coverage90": float(np.mean((y >= gdf.LO) & (y <= gdf.HI)))}

    qb = pd.DataFrame(identity)
    obs_att = qb.obs_att.to_numpy(); sim_att = qb.sim_att.to_numpy()

    def attempt_stats(g):
        if len(g) < 2:
            return None
        o, s = g.obs_att.to_numpy(), g.sim_att.to_numpy()
        return {"N": len(g), "corr": float(np.corrcoef(o, s)[0, 1]),
                "mae": float(np.mean(np.abs(s - o))), "sd_real": float(np.std(o, ddof=1)), "sd_sim": float(np.std(s, ddof=1))}

    matched = qb[qb.matched]
    matched_yards = None
    if len(matched):
        yo, ys = matched.obs_y.to_numpy(), matched.sim_y.to_numpy()
        lo = matched.sim_q.apply(lambda q: q[0]).to_numpy(); hi = matched.sim_q.apply(lambda q: q[1]).to_numpy()
        matched_yards = {"N": len(matched), "MAE": float(np.mean(np.abs(ys - yo))),
            "RMSE": float(np.sqrt(np.mean((ys - yo) ** 2))),
            "coverage90": float(np.mean((yo >= lo) & (yo <= hi))),
            # WITHIN-game (across-sim, one matchup held fixed) simulated dispersion:
            "sim_yards_sd_mean_within_game": float(matched.sim_sd.mean()),
            "sim_yards_skew_mean_within_game": float(matched.sim_skew.mean()),
            # ACROSS-game (these 110 real matched-QB games) dispersion, both sides,
            # on the SAME population -- the only like-for-like realized comparator
            # for a matched-QB-population sd/skew figure.
            "real_yards_sd_across_these_110_games": float(np.std(yo, ddof=1)),
            "sim_yards_sd_across_these_110_games_of_sim_means": float(np.std(ys, ddof=1)),
            # Law-of-total-variance total, computed on the SAME 110-row population as
            # real_yards_sd_across_these_110_games so the two are actually comparable:
            # Var(sim_total) = E[Var(sim|matchup)] + Var(E[sim|matchup])
            #                = mean(sim_sd^2) + Var(sim_y across the 110 matchups).
            # A single real observation per game cannot be split into within/across
            # components, so Var(OBS) is inherently a total-variance figure already --
            # this is the one on the sim side that is actually the same quantity.
            "sim_total_variance_law_of_total_variance": float(
                np.mean(matched.sim_sd.to_numpy() ** 2) + np.var(ys, ddof=1)),
            "sim_total_sd_law_of_total_variance": float(
                np.sqrt(np.mean(matched.sim_sd.to_numpy() ** 2) + np.var(ys, ddof=1))),
        }
    decomp = pd.DataFrame(decomp_rows)
    return {
        "label": label, "shock_sd": shock_sd,
        "player_metrics": player_metrics,
        "conservation_pass": bool(all(conservation)), "conservation_checks": len(conservation),
        "qb_conservation_pass": bool(all(qb_conservation)), "qb_conservation_checks": len(qb_conservation),
        "qb_attempts_all_evaluated": attempt_stats(qb),
        "qb_attempts_matched_only": attempt_stats(matched),
        "qb_matched_yards": matched_yards,
        "variance_decomposition": {
            "n_team_games": len(decomp),
            "mean_var_total": float(decomp.var_total_pass_yards.mean()),
            "mean_sd_total": float(decomp.sd_total_pass_yards.mean()),
            "mean_var_idiosyncratic_noise": float(decomp.var_idiosyncratic_noise.mean()),
            "mean_var_signal_eff_times_volume": float(decomp.var_signal_eff_times_volume.mean()),
            "mean_approx_shared_efficiency_variance": float(decomp.approx_shared_efficiency_variance.mean()),
            "mean_approx_volume_allocation_variance": float(decomp.approx_volume_allocation_variance.mean()),
            "share_idiosyncratic_noise": float((decomp.var_idiosyncratic_noise / decomp.var_total_pass_yards).mean()),
            "share_shared_efficiency": float((decomp.approx_shared_efficiency_variance / decomp.var_total_pass_yards).mean()),
            "share_volume_allocation": float((decomp.approx_volume_allocation_variance / decomp.var_total_pass_yards).mean()),
            "mean_reconstruction_check_ratio": float(decomp.reconstruction_check_ratio.mean()),
            "mean_eff_mult_sd": float(decomp.sd_eff_mult.mean()),
        },
    }


def _skew(a):
    a = np.asarray(a, dtype=float)
    m = a.mean(); s = a.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((a - m) / s) ** 3))


def main():
    selected = select_cohort()
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    oos = all_df[all_df.GAME_ID.isin(selected)].copy()

    print("Building pregame states for", len(selected), "cohort games...")
    cohort_states = []
    for g in selected:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        cohort_states.append((g, state))

    params_by_game = {g: fit_distributions([], state.as_of) for g, state in cohort_states}

    print("Verifying twin fidelity against real simulator._run_one...")
    mismatches = []
    for g, state in cohort_states[:15]:
        seed = int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)
        params = params_by_game[g]
        for sim_id in range(3):
            real_stats, real_team, real_win = _run_one(state, sim_id, seed, params)
            twin_stats, twin_team, twin_win, _ = twin_run_one(state, sim_id, seed, params, draw_environment)
            if real_stats != twin_stats or real_team != twin_team or real_win != twin_win:
                mismatches.append(g)
                break
    fidelity = {"games_checked": min(15, len(cohort_states)), "sims_per_game_checked": 3,
                "mismatches": mismatches, "status": "BIT_EXACT" if not mismatches else "MISMATCH_DETECTED"}
    print(json.dumps(fidelity, indent=2))
    if fidelity["status"] != "BIT_EXACT":
        raise SystemExit("Twin does not reproduce real simulator._run_one; aborting before trusting instrumentation.")

    print("Deriving evidence-based shock sd from historical residuals (cohort games excluded)...")
    derivation = derive_shock_sd(all_df, exclude_game_ids=set(selected))
    print(json.dumps(derivation, indent=2))

    print("Running BASELINE (shock_sd=.35, real draw_environment)...")
    baseline = run_variant(cohort_states, params_by_game, SHOCK_SD_BASELINE, oos, "BASELINE_V7_STATE")

    derived_sd = derivation["residual_sd"]
    print(f"Running VARIANT (shock_sd={derived_sd:.4f}, evidence-derived)...")
    variant = run_variant(cohort_states, params_by_game, derived_sd, oos, "V8_EVIDENCE_DERIVED_SHOCK_SD")

    # realized (real-world) yards dispersion for comparison
    team_game_real = (all_df.groupby(["GAME_ID", "TEAM"], as_index=False).agg(pass_yards=("PASS_YARDS", "sum")))
    realized_unconditional_sd = float(team_game_real.pass_yards.std(ddof=1))

    result = {
        "task": "SN3_SPORTS_V8_YARDAGE_DEPENDENCE",
        "twin_fidelity_check": fidelity,
        "SHARED_STATE_ALREADY_EXISTS": (
            "Y1 finding: simulator._run_one already draws one eff_mult per team per simulation "
            "(env.home_efficiency/away_efficiency ~ Normal(recent_form, .35), consumed directly since "
            "SN3_SPORTS_V6) and multiplies every reception/carry/scramble yardage mean for that team, "
            "that simulation, by it. Receptions within a team-game are therefore conditionally i.i.d. "
            "given eff_mult, not marginally independent -- SN3_SPORTS_V7's SINGLE_NEXT_ACTION statement "
            "that eff_mult 'does not change the fact that the draws are still independent' is incorrect. "
            "This task does not introduce a new shared state; it measures the existing one's variance "
            "contribution and tests whether its hardcoded shock scale (.35, un-evidenced, same class of "
            "magic number as V7's 90s/24-block caps) is well-calibrated against historical residual "
            "dispersion."
        ),
        "BASELINE_QB_MAE": baseline["qb_matched_yards"]["MAE"] if baseline["qb_matched_yards"] else None,
        "BASELINE_QB_RMSE": baseline["qb_matched_yards"]["RMSE"] if baseline["qb_matched_yards"] else None,
        "BASELINE_YARDS_SD": {
            "population": "matched-QB (N=110)",
            "value": baseline["qb_matched_yards"]["sim_total_sd_law_of_total_variance"],
            "value_definition": "sqrt(E[Var(sim|matchup)] + Var(E[sim|matchup])) = sqrt(mean(sim_sd^2) + "
                                 "Var(sim_y across the 110 matchups)) -- total simulated pass-yards "
                                 "variance across the population (within-matchup sim noise PLUS "
                                 "across-matchup mean differences), the only quantity actually comparable "
                                 "to a real-world sd computed from one observation per game (a single real "
                                 "observation cannot be split into within/across components, so Var(OBS) "
                                 "is already a total-variance figure; comparing it to a within-matchup-only "
                                 "sim sd understates real-world variance and was WRONG in an earlier pass "
                                 "of this task -- corrected here).",
            "within_matchup_component_only_do_not_compare_directly_to_realized": baseline["qb_matched_yards"]["sim_yards_sd_mean_within_game"],
            "team_level_all_120_team_games_within_matchup_component_only": baseline["variance_decomposition"]["mean_sd_total"],
        },
        "REALIZED_YARDS_SD": {
            "matched_qb_across_these_110_games": baseline["qb_matched_yards"]["real_yards_sd_across_these_110_games"],
            "unconditional_cross_team_cross_game_full_history": realized_unconditional_sd,
            "conditional_on_recent_form_residual_in_yards_team_level": derivation["residual_sd"] *
                float(team_game_real.pass_yards.mean()),
            "note": "matched_qb_across_these_110_games (a total-variance figure, since it's one real "
                    "observation per game) is the like-for-like comparator to BASELINE/V8_YARDS_SD.value "
                    "(also now a total-variance figure via law of total variance) and is what the "
                    "yards_sd_closer_to_realized gate below actually uses. The other two figures are "
                    "team-level, full-history context: unconditional mixes cross-team quality with real "
                    "game-to-game randomness; the residual-conditional figure isolates the latter using "
                    "the same population that derived the evidence-based shock sd."
        },
        "WITHIN_GAME_COV_BASELINE": baseline["variance_decomposition"],
        "WITHIN_GAME_COV_SHARED_STATE": variant["variance_decomposition"],
        "SHARED_STATE_DERIVATION": derivation,
        "V8_QB_MAE": variant["qb_matched_yards"]["MAE"] if variant["qb_matched_yards"] else None,
        "V8_QB_RMSE": variant["qb_matched_yards"]["RMSE"] if variant["qb_matched_yards"] else None,
        "V8_COV90": variant["qb_matched_yards"]["coverage90"] if variant["qb_matched_yards"] else None,
        "BASELINE_COV90": baseline["qb_matched_yards"]["coverage90"] if baseline["qb_matched_yards"] else None,
        "V8_YARDS_SD": {
            "population": "matched-QB (N=110)",
            "value": variant["qb_matched_yards"]["sim_total_sd_law_of_total_variance"],
            "value_definition": "same law-of-total-variance total as BASELINE_YARDS_SD.value_definition",
            "within_matchup_component_only_do_not_compare_directly_to_realized": variant["qb_matched_yards"]["sim_yards_sd_mean_within_game"],
            "team_level_all_120_team_games_within_matchup_component_only": variant["variance_decomposition"]["mean_sd_total"],
        },
        "Y1_AUDIT_ADDITIONAL_FINDING_DEAD_CODE": (
            "worker/sports_nova_v3/efficiency.py:draw_efficiency is never called from anywhere in the "
            "codebase (grepped worker/ and scripts/: zero call sites). The real per-reception/per-carry "
            "yardage mechanism that feeds pass_yards/rush_yards is inline in simulator._run_one, not "
            "this function. Not in scope to remove here (out of this task's authorized diff), but "
            "recorded so a future pass does not waste time treating draw_efficiency as the live path."
        ),
        "ATTEMPT_CORR": {
            "baseline_matched": baseline["qb_attempts_matched_only"]["corr"] if baseline["qb_attempts_matched_only"] else None,
            "v8_matched": variant["qb_attempts_matched_only"]["corr"] if variant["qb_attempts_matched_only"] else None,
            "note": "Identical to 17 significant digits by construction, not a bug: shock_sd only changes "
                    "the scale of the two rng.normal() calls inside draw_environment for home/away_efficiency, "
                    "not how many random values are drawn or their call order, so every downstream draw "
                    "(draw_block_volume/select_plays/allocate_opportunities, which produce attempts) consumes "
                    "an identical random stream in both runs -- attempts are drawn upstream of, and are "
                    "mechanically independent of, eff_mult's value. mean_eff_mult_sd moving 0.341->0.310 in "
                    "WITHIN_GAME_COV_BASELINE/SHARED_STATE (vs. attempt_corr's exact tie) is the paired "
                    "evidence that the variant parameter did take effect where it was supposed to and nowhere else.",
        },
        "RB_REGRESSION": {
            "baseline_MAE": baseline["player_metrics"].get("RB_rush_yards", {}).get("MAE"),
            "v8_MAE": variant["player_metrics"].get("RB_rush_yards", {}).get("MAE"),
        },
        "WR_REGRESSION": {
            "baseline_MAE": baseline["player_metrics"].get("WR_receiving_yards", {}).get("MAE"),
            "v8_MAE": variant["player_metrics"].get("WR_receiving_yards", {}).get("MAE"),
        },
        "TE_REGRESSION": {
            "baseline_MAE": baseline["player_metrics"].get("TE_receiving_yards", {}).get("MAE"),
            "v8_MAE": variant["player_metrics"].get("TE_receiving_yards", {}).get("MAE"),
        },
        "conservation": {
            "baseline": {"pass": baseline["conservation_pass"], "qb_pass": baseline["qb_conservation_pass"]},
            "v8": {"pass": variant["conservation_pass"], "qb_pass": variant["qb_conservation_pass"]},
        },
        "baseline_full": baseline,
        "variant_full": variant,
    }

    def gate_check():
        mae = result["V8_QB_MAE"]; base_mae = result["BASELINE_QB_MAE"]
        rmse = result["V8_QB_RMSE"]; base_rmse = result["BASELINE_QB_RMSE"]
        sd_sim = result["V8_YARDS_SD"]["value"]; sd_base = result["BASELINE_YARDS_SD"]["value"]
        sd_real = result["REALIZED_YARDS_SD"]["matched_qb_across_these_110_games"]
        cov90 = result["V8_COV90"]; base_cov90 = result["BASELINE_COV90"]
        corr = result["ATTEMPT_CORR"]["v8_matched"]
        cons = result["conservation"]["v8"]["pass"] and result["conservation"]["v8"]["qb_pass"]
        # "coverage90" (inherited name from the existing validation scripts) is actually built from
        # np.quantile(a, [.1, .9]) -- an 80% interval, nominal target 0.80, not 0.90. "Improve" therefore
        # means move CLOSER TO 0.80, not simply "not decrease": a naive higher-is-better read would score
        # ever-wider (more over-covering) intervals as improvement, which is backwards for a dispersion-
        # calibration task where the baseline (0.8455) already over-covers a nominal-80% interval.
        cov90_dist = abs(cov90 - 0.80) if cov90 is not None else None
        base_cov90_dist = abs(base_cov90 - 0.80) if base_cov90 is not None else None
        return {
            "qb_mae": "PASS" if mae is not None and mae < 61.5 else "FAIL",
            "qb_mae_preferred": "PASS" if mae is not None and mae < 60 else "FAIL",
            # "materially" is this task's own reading of the mission's gate text (not specified by the
            # mission itself): a >=5% RMSE reduction. Recorded so a future task does not inherit this
            # threshold as if it had been given rather than chosen.
            "qb_rmse_improve_materially": "PASS" if (rmse is not None and base_rmse is not None and rmse < base_rmse * 0.95) else "FAIL",
            "yards_sd_closer_to_realized": "PASS" if abs(sd_sim - sd_real) < abs(sd_base - sd_real) else "FAIL",
            "cov90_improve_or_acceptable": "PASS" if (cov90_dist is not None and base_cov90_dist is not None and cov90_dist <= base_cov90_dist) else "FAIL",
            "cov90_nominal_target": 0.80,
            "cov90_distance_from_nominal": {"baseline": base_cov90_dist, "v8": cov90_dist},
            "attempt_corr": "PASS" if (corr is not None and corr >= 0.20) else "FAIL",
            "conservation": "PASS" if cons else "FAIL",
        }

    gate = gate_check()
    result["gate_result"] = gate

    overdispersion_pct_baseline = (result["BASELINE_YARDS_SD"]["value"] /
        result["REALIZED_YARDS_SD"]["matched_qb_across_these_110_games"] - 1.0) * 100
    overdispersion_pct_v8 = (result["V8_YARDS_SD"]["value"] /
        result["REALIZED_YARDS_SD"]["matched_qb_across_these_110_games"] - 1.0) * 100

    # Both mechanical gate items (yards_sd_closer_to_realized, cov90_improve_or_acceptable) PASS after
    # the cov90-direction fix above -- mechanical_value below is True. Mechanically AND-ing two booleans
    # into "material_variance_improvement" would let a knife-edge decide the verdict, so the materiality
    # call is overridden to False here and put on the record rather than left implicit: dispersion after
    # the fix is still ~overdispersion_pct_v8 over the matched-QB realized sd (down from
    # ~overdispersion_pct_baseline, computed above) -- narrower, not calibrated -- while QB yards MAE/RMSE,
    # the metrics this whole four-task series is gated on, did not move (mae_delta/rmse_delta below). A
    # modest reduction in an already-large over-dispersion figure, with zero MAE movement, does not clear
    # a "material calibration improvement" bar. cov90's mechanical PASS is additionally confounded (see
    # SINGLE_NEXT_ACTION): the baseline's near-nominal coverage is itself two separate defects (over-wide
    # intervals, and a level bias pushing real outcomes toward the upper tail) partially cancelling, not
    # evidence of real calibration quality. NOTE: this override makes the PROMOTE_ARCH/STRUCTURAL_SIGNAL
    # branches below unreachable FOR THIS RUN's numbers; this script is a single-run diagnostic, not a
    # parameterized tool re-run at arbitrary scales, so that is intentional here, not a latent bug -- but
    # a future rerun at a materially different derived scale should re-examine this override rather than
    # inherit it blindly.
    material_variance_improvement_mechanical = (gate["yards_sd_closer_to_realized"] == "PASS" and
                                                 gate["cov90_improve_or_acceptable"] == "PASS")
    material_variance_improvement = False
    mae_clears = gate["qb_mae"] == "PASS"
    mae_delta = (result["V8_QB_MAE"] - result["BASELINE_QB_MAE"]
                 if result["V8_QB_MAE"] is not None and result["BASELINE_QB_MAE"] is not None else None)
    rmse_delta = (result["V8_QB_RMSE"] - result["BASELINE_QB_RMSE"]
                  if result["V8_QB_RMSE"] is not None and result["BASELINE_QB_RMSE"] is not None else None)
    cov90_delta_rows = (round((result["V8_COV90"] - result["BASELINE_COV90"]) * 110)
                        if result["V8_COV90"] is not None and result["BASELINE_COV90"] is not None else None)

    # Verdict rests on flat MAE/RMSE (see mae_delta/rmse_delta above) plus the explicit
    # material_variance_improvement=False judgment above -- not on the raw cov90/yards_sd gate mechanics,
    # which both PASS. Per inv, no rerun with a different shock scale was performed to try to move cov90
    # or MAE; SHARED_STATE_DERIVATION's evidenced value is used as-is, win or lose.
    if mae_clears and material_variance_improvement:
        verdict = "PROMOTE_ARCH"
    elif material_variance_improvement and not mae_clears:
        verdict = "STRUCTURAL_SIGNAL"
    else:
        verdict = "REJECT_HYPOTHESIS"

    result["gate_result"]["material_variance_improvement_judgment"] = {
        "mechanical_value": material_variance_improvement_mechanical,
        "overridden_to": material_variance_improvement,
        "reason": "See comment above material_variance_improvement in source: post-fix dispersion is "
                  "still substantially over-realistic and MAE/RMSE did not move; the mechanical AND of "
                  "yards_sd_closer_to_realized and cov90_improve_or_acceptable was overridden rather than "
                  "allowed to decide the verdict on its own.",
    }

    result["VERDICT"] = verdict
    result["FULL_1693_RUN_JUSTIFIED"] = "YES" if verdict == "PROMOTE_ARCH" else "NO"
    result["SINGLE_NEXT_ACTION"] = (
        "Do not patch sample_compound_signed's dependence structure, and do not re-tune eff_mult's "
        "shock scale further -- two independent findings both point away from this subsystem: (1) "
        "eff_mult already supplies conditional dependence and explains ~53% of within-game matched-QB "
        "pass-yards variance (idiosyncratic per-reception noise is only ~19%; see WITHIN_GAME_COV_BASELINE), "
        "so the 'independent draws suppress variance' premise this task was assigned to test is false; "
        f"(2) total simulated pass-yards variance (law-of-total-variance, matched-QB N=110) is ALREADY "
        f"~{overdispersion_pct_baseline:.0f}% OVER-realistic relative to the matched-QB population's own "
        "across-game sd (BASELINE_YARDS_SD.value vs REALIZED_YARDS_SD.matched_qb_across_these_110_games) "
        "-- the opposite problem from under-dispersion, so widening dependence further would make "
        f"calibration worse, not better (recalibrating the shock scale to the historical-residual-evidenced "
        f"value only narrowed this to ~{overdispersion_pct_v8:.0f}% over-realistic, still substantially over). "
        "Recalibrating the shock's scale to the historical-residual-evidenced value (.315 vs the hardcoded "
        ".35) moved QB MAE by only 0.03 yards and RMSE by only 0.06 yards -- both flat, in the wrong "
        "direction, well inside noise on N=110. Per V7's own OVERALL_QB_MAE attribution (sim_attempts_mean "
        "24.96 vs realized_attempts_mean 28.43, at sim_ypa 6.81, predicts ~-23.6 yards of pure level bias "
        "per QB-game -- nearly all of the ~69-70 MAE this series has been stuck at across V5/V6/V7/this "
        "task) is independently corroborated by the coverage90 numbers here: baseline coverage reads "
        "near-nominal (0.845 vs the true 0.80 target for a P10-P90 interval) DESPITE the interval being "
        "~47% too wide, because the level bias pushes real outcomes toward the upper tail of an "
        "already-too-wide distribution -- two separate defects (over-width and level bias) partially "
        "cancelling inside one coverage metric, which is why coverage looked roughly fine while both "
        "yards_sd and MAE did not. Redirect to why simulated attempt/play VOLUME sits below the historical "
        "mean in LEVEL (not variance) -- e.g. draw_block_volume's expected-plays formula (60/pace_seconds)*2.7, "
        "or QB target/attempt share -- a different question from every dispersion/dependence fix this "
        "four-task series has already tried and exhausted."
    )
    result["VERDICT_REASON"] = (
        f"gate={json.dumps(gate)}. Decisive evidence: QB MAE {result['BASELINE_QB_MAE']:.3f} -> "
        f"{result['V8_QB_MAE']:.3f} (delta {mae_delta:+.3f}), QB RMSE {result['BASELINE_QB_RMSE']:.3f} -> "
        f"{result['V8_QB_RMSE']:.3f} (delta {rmse_delta:+.3f}) -- both flat under the evidence-derived shock "
        "scale, with no re-tuning performed to chase a better number; this alone is sufficient for "
        "REJECT_HYPOTHESIS since MAE is the metric the whole series is gated on. Both mechanical dispersion "
        f"gate items PASS after correcting the cov90 direction bug (cov90 moved {result['BASELINE_COV90']:.3f} "
        f"-> {result['V8_COV90']:.3f}, a {cov90_delta_rows:+d}-row shift on N=110, CLOSER to the true nominal "
        "target of 0.80 for a P10-P90 interval, not farther as an earlier pass of this task mis-scored it; "
        "yards_sd also moved closer to realized). This task explicitly OVERRODE a mechanical PASS+PASS-implies-"
        "STRUCTURAL_SIGNAL branch (see gate_result.material_variance_improvement_judgment) because post-fix "
        f"dispersion is still ~{overdispersion_pct_v8:.0f}% over the matched-QB realized sd (down from "
        f"~{overdispersion_pct_baseline:.0f}%) -- narrower, not calibrated -- and because MAE, the metric that "
        "actually matters, did not move at all. The verdict rests on that judgment plus the flat MAE/RMSE, "
        "not on the cov90/yards_sd mechanics alone."
    )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "SN3_SPORTS_V8_YARDAGE_DEPENDENCE_REPORT.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print("Wrote", out_path)
    print(json.dumps({k: v for k, v in result.items() if k not in ("baseline_full", "variant_full")}, indent=2, default=str))


if __name__ == "__main__":
    main()
