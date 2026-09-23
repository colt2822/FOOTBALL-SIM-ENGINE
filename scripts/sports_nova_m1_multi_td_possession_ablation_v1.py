"""SPORTS_NOVA MULTI_TD_POSSESSION causal ablation V1 -- run harness (NOT a new engine version; no worker/sports_nova_v28*).

  python scripts/sports_nova_m1_multi_td_possession_ablation_v1.py freeze        # writes the pre-registration (refuses if any output exists)
  python scripts/sports_nova_m1_multi_td_possession_ablation_v1.py run [--limit N] [--workers W]
  (analysis: sports_nova_m1_multi_td_possession_ablation_v1_analysis.py)

Per pilot game (the FG_CAUSAL_ABLATION_V1 / V27 60-game cohort, 128 sims, seed = sha256(game)[:8]), four runs on the SAME PregameState / seed:
  REF   : unpatched worker.sports_nova_v27_causal_fg_rate.simulator.simulate_scoring          (the incumbent, untouched)
  BASE  : same V27 outer loop, `_v23._run_one` replaced by an instrumented copy with cap=False   -> must be np.array_equal to REF
  ABL   : same, cap=True ("at most one offensive TD per possession"), score feedback LIVE        -> the ablation of record
  CTRL  : cap=True with the script-pass-rate score differential replayed from BASE's trajectory  -> every random draw identical to BASE;
          isolates the pure scoring effect of the cap (component isolation proof + exact block pairing for the >8-pt ledger)
Engine sources are never edited (hashes before/after).  Scope: V27 SCORING BOUNDARY only (historical games have no RosterSnapshot), same as the
V27 60-game reproduction.  No market input is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.sports_nova_m1_fg_causal_ablation_v1 as H  # noqa: E402  (pilot cohort, seed rule, state builder)
from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import assert_path  # noqa: E402
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v21.config import PAT_MAKE_RATE  # noqa: E402
from worker.sports_nova_v23 import simulator as _v23  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import config as cfg  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import simulator as v27  # noqa: E402

OUT = ROOT / "data" / "sports_nova_v3" / "MULTI_TD_POSSESSION_ABLATION_V1"
ARR = OUT / "arrays"
PREREG = OUT / "MULTI_TD_POSSESSION_ABLATION_V1_PREREG.json"
FG_ABL = ROOT / "data" / "sports_nova_v3" / "FG_CAUSAL_ABLATION_V1"
V27_PKG = ROOT / "worker" / "sports_nova_v27_causal_fg_rate"
AUX_TAG = 0xCA9
FIELDS = ["sim", "block", "off", "sec_before", "pre_h", "pre_a", "plays", "pass_att", "designed", "scrambles", "targets", "rec", "pass_yds", "rush_yds",
          "rec_sites", "carry_sites", "trial_plays", "raw_td", "raw_rec_td", "raw_rush_td", "raw_pts", "td", "rec_td", "rush_td", "supp_td",
          "fg_elig", "fg", "pts"]
FI = {k: i for i, k in enumerate(FIELDS)}
TEAM_KEYS = ("score", "pass_attempts", "pass_yards", "rush_attempts", "rush_yards", "blocks")
PLAYER_KEYS = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds", "targets", "receptions", "receiving_yards",
               "receiving_tds", "scored_tds")
GUARDED_FILES = [
    "worker/sports_nova_v23/simulator.py", "worker/sports_nova_v23/allocation.py", "worker/sports_nova_v23/config.py",
    "worker/sports_nova_v3/distributions.py", "worker/sports_nova_v3/game_state.py", "worker/sports_nova_v3/game_script.py",
    "worker/sports_nova_v3/play_volume.py", "worker/sports_nova_v3/play_selection.py", "worker/sports_nova_v3/scoring.py",
    "worker/sports_nova_v19/simulator.py", "worker/sports_nova_v21/config.py", "worker/sports_nova_v25_role_aware/simulator.py",
    "worker/sports_nova_v26_active_skill_state/simulator.py",
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guarded_hashes() -> dict:
    d = {f: sha256_file(ROOT / f) for f in GUARDED_FILES}
    d.update({f"worker/sports_nova_v27_causal_fg_rate/{p.name}": sha256_file(p) for p in sorted(V27_PKG.glob("*.py"))})
    return d


# --------------------------------------------------------------------------------------------------------------------------------------
# Instrumented copy of worker.sports_nova_v23.simulator._run_one (statement order and every rng call identical; see BASE == REF check).
# --------------------------------------------------------------------------------------------------------------------------------------
def run_one(pregame, sim_id: int, seed: int, params, *, cap: bool, trace, out: dict):
    m = _v23
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), int(sim_id)])))
    aux = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), int(sim_id), AUX_TAG]))) if cap else None   # separate stream: main stream untouched
    player_ids = sorted(p.player_id for p in pregame.players)
    stats = {pid: {name: 0 for name in m.STAT_NAMES} for pid in player_ids}
    team_ids = (pregame.home.team_id, pregame.away.team_id)
    team = {tid: {name: 0 for name in m.TEAM_STAT_NAMES} for tid in team_ids}
    state = m.GameState(game_id=pregame.game_id, simulation_id=sim_id, block_index=0, seconds_remaining=3600, home_score=0, away_score=0,
                        possession=pregame.home.team_id, field_position=25)
    env = m.draw_environment(pregame, rng, epistemic_id=sim_id)
    eff_mult = {pregame.home.team_id: float(np.clip(1.0 + env.home_efficiency, 0.3, 2.5)),
                pregame.away.team_id: float(np.clip(1.0 + env.away_efficiency, 0.3, 2.5))}
    player_by_id = {p.player_id: p for p in pregame.players}
    qb_shares = {tid: m._qb_shares(pregame, tid) for tid in team_ids}
    qb_draw_probs = {}
    for tid, (ids, shares) in qb_shares.items():
        qb_draw_probs[tid] = (ids, rng.dirichlet(np.maximum(shares * 80.0, 1e-6)) if len(ids) else shares)
    blocks: list = []
    events: list = []
    for _ in range(60):
        if state.seconds_remaining <= 0:
            break
        offense = state.possession
        offense_team = pregame.home if offense == pregame.home.team_id else pregame.away
        team[offense]["blocks"] += 1
        plays = m.draw_block_volume(state, env, rng)
        if trace is None:
            script_state = state
        else:                                                   # CTRL: replay BASE's pre-block score differential into the script only
            bi = state.block_index
            if bi >= len(trace):
                raise AssertionError("CTRL block stream diverged from BASE (more blocks than BASE)")
            ph, pa = trace[bi]
            script_state = state.model_copy(update={"home_score": int(ph), "away_score": int(pa)})
        selection = m.select_plays(plays, m.script_pass_rate(pregame, script_state, env), rng,
                                   sack_rate=m._feature(offense_team, "sack_rate", .065), scramble_rate=m._feature(offense_team, "scramble_rate", .07))
        alloc = m.allocate_opportunities(pregame, offense, selection.pass_attempts, selection.designed_rushes, rng)
        ids, probs = qb_draw_probs[offense]
        qb = player_by_id[ids[int(rng.choice(len(ids), p=probs))]] if len(ids) else None
        b_rush_yds = 0
        if qb is not None:
            m._inc(stats, qb.player_id, "pass_attempts", selection.pass_attempts)
            m._inc(stats, qb.player_id, "rush_attempts", selection.scrambles)
            scramble_yards = m.sample_compound_signed(selection.scrambles, 4.0 * eff_mult[offense], 5.0, rng)
            m._inc(stats, qb.player_id, "rush_yards", scramble_yards)
            team[offense]["rush_yards"] += scramble_yards
            b_rush_yds += scramble_yards
        team[offense]["pass_attempts"] += selection.pass_attempts
        team[offense]["rush_attempts"] += selection.rush_attempts
        block_points = 0
        # --- per-block TD bookkeeping (observer only) ---
        td_awarded = False
        b_targets = b_rec = b_pass_yds = 0
        rec_sites = carry_sites = trial_plays = 0
        raw_td = raw_rec_td = raw_rush_td = raw_pts = 0
        td_n = rec_td_n = rush_td_n = supp_n = 0
        for pid, targets in alloc.target_counts.items():
            m._inc(stats, pid, "targets", targets)
            receiver = next(p for p in pregame.players if p.player_id == pid)
            catch_rate = np.clip(m._feature(receiver, "catch_rate", .64), .05, .95)
            rec = int(rng.binomial(targets, catch_rate))
            ypr_mean = m._feature(receiver, "yards_per_reception", params.receiving_yards_mean)
            rec_yards = m.sample_compound_signed(rec, ypr_mean * eff_mult[offense], params.receiving_yards_sd, rng)
            m._inc(stats, pid, "receptions", rec)
            m._inc(stats, pid, "receiving_yards", rec_yards)
            if qb is not None:
                m._inc(stats, qb.player_id, "pass_yards", rec_yards)
            team[offense]["pass_yards"] += rec_yards
            b_targets += targets; b_rec += rec; b_pass_yds += rec_yards
            if rec > 0:
                rec_sites += 1; trial_plays += rec
            pass_td_rate = m._feature(offense_team, "pass_td_rate", params.td_rate)
            td_count = int(rng.binomial(rec, np.clip(pass_td_rate, .001, .2)))
            if td_count:
                made_pats = int(rng.binomial(td_count, PAT_MAKE_RATE))          # drawn in BOTH arms with identical arguments
                raw_td += td_count; raw_rec_td += td_count; raw_pts += 6 * td_count + made_pats
                if not cap:
                    m._inc(stats, pid, "receiving_tds", td_count)
                    m._inc(stats, pid, "scored_tds", td_count)
                    if qb is not None:
                        m._inc(stats, qb.player_id, "pass_tds", td_count)
                    block_points += 6 * td_count + made_pats
                    td_n += td_count; rec_td_n += td_count
                else:
                    if not td_awarded:
                        made = made_pats if td_count == 1 else int(aux.random() < PAT_MAKE_RATE)
                        m._inc(stats, pid, "receiving_tds", 1)
                        m._inc(stats, pid, "scored_tds", 1)
                        if qb is not None:
                            m._inc(stats, qb.player_id, "pass_tds", 1)
                        block_points += 6 + made
                        td_awarded = True; td_n += 1; rec_td_n += 1
                        sup = td_count - 1
                    else:
                        sup = td_count
                    if sup:
                        supp_n += sup
                        events.append({"sim": sim_id, "block": state.block_index, "site": "REC", "player": pid, "raw_td": td_count, "pat_made_drawn": made_pats,
                                       "suppressed_td": sup, "kept": td_count - sup})
        for pid, carries in alloc.carry_counts.items():
            m._inc(stats, pid, "rush_attempts", carries)
            rush_yards = m.sample_compound_signed(carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng)
            m._inc(stats, pid, "rush_yards", rush_yards)
            team[offense]["rush_yards"] += rush_yards
            b_rush_yds += rush_yards
            if carries > 0:
                carry_sites += 1; trial_plays += carries
            rush_td_rate = m._feature(offense_team, "rush_td_rate", params.td_rate * .65)
            rush_td = int(rng.binomial(carries, np.clip(rush_td_rate, .0005, .15)))
            if rush_td:
                made_pats = int(rng.binomial(rush_td, PAT_MAKE_RATE))
                raw_td += rush_td; raw_rush_td += rush_td; raw_pts += 6 * rush_td + made_pats
                if not cap:
                    m._inc(stats, pid, "rush_tds", rush_td)
                    m._inc(stats, pid, "scored_tds", rush_td)
                    block_points += 6 * rush_td + made_pats
                    td_n += rush_td; rush_td_n += rush_td
                else:
                    if not td_awarded:
                        made = made_pats if rush_td == 1 else int(aux.random() < PAT_MAKE_RATE)
                        m._inc(stats, pid, "rush_tds", 1)
                        m._inc(stats, pid, "scored_tds", 1)
                        block_points += 6 + made
                        td_awarded = True; td_n += 1; rush_td_n += 1
                        sup = rush_td - 1
                    else:
                        sup = rush_td
                    if sup:
                        supp_n += sup
                        events.append({"sim": sim_id, "block": state.block_index, "site": "RUSH", "player": pid, "raw_td": rush_td, "pat_made_drawn": made_pats,
                                       "suppressed_td": sup, "kept": rush_td - sup})
        if alloc.residual_carries:
            resid = m.sample_compound_signed(alloc.residual_carries, params.rush_yards_mean * eff_mult[offense], params.rush_yards_sd, rng)
            team[offense]["rush_yards"] += resid
            b_rush_yds += resid
        fg_elig = int(block_points == 0 and selection.plays >= 3)
        fg = 0
        if block_points == 0 and selection.plays >= 3 and rng.random() < params.fg_rate:
            block_points = 3
            fg = 1
        elapsed = max(1, min(500, int(round(plays * env.pace_seconds))))
        blocks.append((sim_id, state.block_index, 0 if offense == team_ids[0] else 1, state.seconds_remaining, state.home_score, state.away_score,
                       selection.plays, selection.pass_attempts, selection.designed_rushes, selection.scrambles, b_targets, b_rec, b_pass_yds, b_rush_yds,
                       rec_sites, carry_sites, trial_plays, raw_td, raw_rec_td, raw_rush_td, raw_pts, td_n, rec_td_n, rush_td_n, supp_n, fg_elig, fg, block_points))
        if offense == team_ids[0]:
            state = state.advance(seconds=elapsed, possession=team_ids[1], field_position=int(rng.integers(15, 86)), home_score=state.home_score + block_points)
        else:
            state = state.advance(seconds=elapsed, possession=team_ids[0], field_position=int(rng.integers(15, 86)), away_score=state.away_score + block_points)
    for tid in team_ids:
        team[tid]["score"] = state.home_score if tid == team_ids[0] else state.away_score
        team[tid]["total"] = team[tid]["score"]
    out[sim_id] = (blocks, events)
    return stats, team, m.winner(state.home_score, state.away_score)


def patched_run(state, n: int, seed: int, cap: bool, trace_by_sim: dict | None):
    """Run V27's UNCHANGED simulate_scoring with `_v23._run_one` swapped for the instrumented copy; always restored."""
    out: dict = {}
    orig = _v23._run_one

    def patched(pregame, sim_id, seed_, params):
        return run_one(pregame, sim_id, seed_, params, cap=cap, trace=None if trace_by_sim is None else trace_by_sim[sim_id], out=out)

    _v23._run_one = patched
    try:
        batch = v27.simulate_scoring(state, n, seed, cfg.MODEL_VERSION)
    finally:
        _v23._run_one = orig
    assert _v23._run_one is orig
    return batch, out


def batches_equal(a, b) -> dict:
    ok_p = {k: bool(np.array_equal(a.player_stats[k], b.player_stats[k])) for k in a.player_stats}
    ok_t = {k: bool(np.array_equal(a.team_stats[k], b.team_stats[k])) for k in a.team_stats}
    return {"player": ok_p, "team": ok_t, "winner": bool(np.array_equal(a.winner, b.winner)), "all": all(ok_p.values()) and all(ok_t.values()) and bool(np.array_equal(a.winner, b.winner))}


def block_matrix(out: dict, n: int) -> np.ndarray:
    rows = [r for i in range(n) for r in out[i][0]]
    return np.asarray(rows, dtype=np.int32).reshape(-1, len(FIELDS))


def arm_arrays(batch, bm: np.ndarray) -> dict:
    n = batch.n_sims
    a = {"team_" + k: np.asarray(batch.team_stats[k], np.int64) for k in TEAM_KEYS}
    for k in PLAYER_KEYS:
        arr = np.asarray(batch.player_stats[k])
        cols = np.zeros((n, 2), np.int64)
        for j, tid in enumerate(batch.team_ids):
            idx = [i for i, pid in enumerate(batch.player_ids) if batch.player_team[pid] == tid]
            cols[:, j] = arr[:, idx].sum(axis=1) if idx else 0
        a["psum_" + k] = cols
    for k in ("targets", "receptions", "rush_attempts", "receiving_yards", "rush_yards", "scored_tds"):
        a["player_" + k] = np.asarray(batch.player_stats[k], np.int64)
    a["winner"] = np.asarray(batch.winner)
    a["blocks"] = bm
    return a


def accounting(batch, bm: np.ndarray, state, cap: bool) -> dict:
    n = batch.n_sims
    ix = FI
    score = np.zeros((n, 2), np.int64); tds = score.copy(); nblk = score.copy()
    bad = 0
    for r in bm:
        s, j = int(r[ix["sim"]]), int(r[ix["off"]])
        score[s, j] += r[ix["pts"]]; tds[s, j] += r[ix["td"]]; nblk[s, j] += 1
        t, p = int(r[ix["td"]]), int(r[ix["pts"]])
        if cap:
            if t > 1 or p not in (0, 3, 6, 7):
                bad += 1
        elif t == 0:
            bad += int(p not in (0, 3))
        else:
            bad += int(not (6 * t <= p <= 7 * t))
        if p == 3 and t:
            bad += 1
        bad += int(r[ix["fg"]] == 1 and (t > 0 or p != 3))
    acct = {"score_eq_block_points": int((score != np.asarray(batch.team_stats["score"])).sum()),
            "tds_eq_player_scored_tds": int((tds != _psum(batch, "scored_tds")).sum()),
            "blocks_eq_team_blocks": int((nblk != np.asarray(batch.team_stats["blocks"])).sum()),
            "block_point_shape_violations": int(bad)}
    fails: list = []
    for i in range(n):
        stats = {pid: {k: int(batch.player_stats[k][i, j]) for k in PLAYER_KEYS} for j, pid in enumerate(batch.player_ids)}
        team = {tid: {k: int(batch.team_stats[k][i, j]) for k in TEAM_KEYS} for j, tid in enumerate(batch.team_ids)}
        assert_path(state.game_id, state, {"result": (stats, team, None), "sim_id": i, "blocks": [], "residual_by_team": {}}, "V23", fails)
    acct["assert_path_failures"] = len(fails)
    if cap:
        acct["max_td_per_block"] = int(bm[:, ix["td"]].max()) if len(bm) else 0
    return acct


def _psum(batch, key):
    arr = np.asarray(batch.player_stats[key]); n = batch.n_sims
    cols = np.zeros((n, 2), np.int64)
    for j, tid in enumerate(batch.team_ids):
        idx = [i for i, pid in enumerate(batch.player_ids) if batch.player_team[pid] == tid]
        cols[:, j] = arr[:, idx].sum(axis=1) if idx else 0
    return cols


def prefix_check(bm_base: np.ndarray, bm_abl: np.ndarray, n: int) -> dict:
    """RNG alignment evidence.  Volume/yield columns must be identical in every block up to and including the first block that suppressed a TD
    (draws are byte-identical inside a block); any difference before that point is a hard failure.  Also block 0 of every sim, and whole-sim identity for
    sims that never suppressed."""
    vol = [FI[k] for k in ("off", "sec_before", "plays", "pass_att", "designed", "scrambles", "targets", "rec", "pass_yds", "rush_yds", "rec_sites", "carry_sites",
                           "trial_plays", "raw_td", "raw_rec_td", "raw_rush_td", "raw_pts", "fg_elig")]
    res = {"sims": n, "sims_with_suppression": 0, "sims_identical_whole": 0, "prefix_violations": 0, "block0_violations": 0, "first_divergence_before_first_suppression": 0,
           "first_divergence_block_index_hist": {}, "first_suppression_block_hist": {}, "sims_no_suppression_but_differ": 0}
    for s in range(n):
        a = bm_base[bm_base[:, FI["sim"]] == s]; b = bm_abl[bm_abl[:, FI["sim"]] == s]
        supp_blocks = np.nonzero(b[:, FI["supp_td"]] > 0)[0]
        if len(a) and len(b) and not np.array_equal(a[0][vol], b[0][vol]):
            res["block0_violations"] += 1
        first_sup = int(supp_blocks[0]) if len(supp_blocks) else None
        L = min(len(a), len(b))
        diff = np.nonzero((a[:L][:, vol] != b[:L][:, vol]).any(axis=1))[0]
        first_div = int(diff[0]) if len(diff) else (None if len(a) == len(b) else L)
        if first_sup is None:
            if not (len(a) == len(b) and np.array_equal(a, b)):
                res["sims_no_suppression_but_differ"] += 1
            else:
                res["sims_identical_whole"] += 1
        else:
            res["sims_with_suppression"] += 1
            key = str(first_sup); res["first_suppression_block_hist"][key] = res["first_suppression_block_hist"].get(key, 0) + 1
            if first_div is not None:
                res["first_divergence_block_index_hist"][str(first_div)] = res["first_divergence_block_index_hist"].get(str(first_div), 0) + 1
                if first_div <= first_sup:
                    res["first_divergence_before_first_suppression"] += 1
                    res["prefix_violations"] += 1
    res["frac_sims_with_suppression"] = res["sims_with_suppression"] / n
    return res


def run_game(args) -> dict:
    game, n = args
    t0 = time.time()
    season, week, away, home = H.game_parts(game)
    all_df = pd.read_parquet(H.PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    prior = all_df[(all_df._key < season * 100 + week) & (all_df.SEASON >= max(1999, season - 5))]
    state = H.make_state(game, prior, H.surrogate_kickoff(season, week), None)
    seed = H.game_seed(game)
    pre = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    ref = v27.simulate_scoring(state, n, seed, cfg.MODEL_VERSION)                               # incumbent, unpatched
    res: dict = {"game": game, "seed": seed, "home": home, "away": away, "n_sims": n, "fg_rate": ref.runtime["fg_rate"],
                 "fg_rate_equals_ablation_prereg": bool(abs(ref.runtime["fg_rate"] - pre["PER_GAME_ESTIMATES"][game]["rate"]) < 1e-15)}
    base, ob = patched_run(state, n, seed, False, None)
    res["base_equals_ref"] = batches_equal(base, ref)
    stored = np.load(H.ARR / f"{game}.npz")
    res["base_team_score_equals_stored_v27_ablation_arm"] = bool(np.array_equal(stored["abl__team_score"], np.asarray(base.team_stats["score"])))
    trace = {i: [(r[FI["pre_h"]], r[FI["pre_a"]]) for r in ob[i][0]] for i in range(n)}
    abl, oa = patched_run(state, n, seed, True, None)
    ctrl, oc = patched_run(state, n, seed, True, trace)
    bm = {"base": block_matrix(ob, n), "abl": block_matrix(oa, n), "ctrl": block_matrix(oc, n)}
    bt = {"base": base, "abl": abl, "ctrl": ctrl}
    res["accounting"] = {arm: accounting(bt[arm], bm[arm], state, arm != "base") for arm in bt}
    res["prefix_abl_vs_base"] = prefix_check(bm["base"], bm["abl"], n)
    vol_cols = [FI[k] for k in ("sim", "block", "off", "sec_before", "plays", "pass_att", "designed", "scrambles", "targets", "rec", "pass_yds", "rush_yds",
                                "rec_sites", "carry_sites", "trial_plays", "raw_td", "raw_rec_td", "raw_rush_td", "raw_pts", "fg_elig", "fg")]
    res["ctrl_block_stream_identical_to_base"] = bool(bm["ctrl"].shape == bm["base"].shape and np.array_equal(bm["ctrl"][:, vol_cols], bm["base"][:, vol_cols]))
    res["ctrl_vs_base_stats_equal_except_td_and_score"] = {
        "player": {k: bool(np.array_equal(ctrl.player_stats[k], base.player_stats[k])) for k in ctrl.player_stats},
        "team": {k: bool(np.array_equal(ctrl.team_stats[k], base.team_stats[k])) for k in ctrl.team_stats}}
    events = {"abl": [e for i in range(n) for e in oa[i][1]], "ctrl": [e for i in range(n) for e in oc[i][1]]}
    res["events"] = events
    ARR.mkdir(parents=True, exist_ok=True)
    flat = {}
    for arm in bt:
        for k, v in arm_arrays(bt[arm], bm[arm]).items():
            flat[f"{arm}__{k}"] = v
    flat["player_ids"] = np.asarray(base.player_ids)
    flat["player_team_home"] = np.asarray([base.player_team[p] == base.team_ids[0] for p in base.player_ids])
    np.savez_compressed(ARR / f"{game}.npz", **flat)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def concurrency() -> dict:
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'codex' } | Select-Object -ExpandProperty Name"], capture_output=True, text=True)
    recent = [p.relative_to(ROOT).as_posix() for p in (ROOT / "worker").rglob("*.py") if time.time() - p.stat().st_mtime < 3600 and "__pycache__" not in p.parts]
    return {"codex_processes_running": sorted(set(ps.stdout.split())), "worker_py_files_modified_last_hour": recent,
            "STATEMENT": "a Codex app-server IS running on this machine; this ablation writes only under scripts/sports_nova_m1_multi_td_possession_ablation_v1*.py and "
                         "data/sports_nova_v3/MULTI_TD_POSSESSION_ABLATION_V1/ and edits no worker/ file"}


def cmd_freeze() -> None:
    if PREREG.exists() or (ARR.exists() and any(ARR.glob("*.npz"))):
        raise SystemExit("REFUSING: prereg or outputs already exist (a rule change ships as a new version, never an edit)")
    abl_pre = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    games = H.pilot_games(list(json.loads(H.MANIFEST.read_text())["GAME_IDS"]))
    assert games == abl_pre["PILOT"]["game_ids"] and len(games) == 60
    v23_src = (ROOT / "worker/sports_nova_v23/simulator.py").read_text()
    rec = {
        "SCHEMA": "MULTI_TD_POSSESSION_ABLATION_V1_PREREG", "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "BASE": cfg.MODEL_VERSION, "MODE": "SINGLE_DEFECT_CAUSAL_ABLATION",
        "SCRIPT_SHA256_AT_FREEZE": sha256_file(Path(__file__)), "GUARDED_FILE_SHA256": guarded_hashes(), "V27_FG_RATE_SOURCE": "causal estimator A via v27.simulate_scoring (unchanged)",
        "COHORT": {"games": len(games), "game_ids": games, "sims_per_game": H.N_SIMS, "seed_rule": "sha256(game_id)[:8] (H.game_seed)", "scope": "V27 SCORING BOUNDARY only (no RosterSnapshot exists for historical games)"},
        "PHASE1_TD_PATH_STATIC": {
            "score_mutation_sites_in_V23_run_one": ["receiver loop (simulator.py:78-98) block_points += 6*td_count + made_pats", "carrier loop (:99-110) block_points += 6*rush_td + made_pats",
                                                    "FG gate (:114) block_points = 3 only when block_points == 0 and plays >= 3"],
            "receiver_td_trials": "one rng.binomial(rec, clip(pass_td_rate,.001,.2)) per entry of alloc.target_counts; each call may return >1",
            "carrier_td_trials": "one rng.binomial(carries, clip(rush_td_rate,.0005,.15)) per entry of alloc.carry_counts; each call may return >1",
            "qb_td_trials": 0, "qb_note": "QB scrambles add rush_attempts/yards only (no TD draw; deferred QB_SCRAMBLE_TD_OMISSION); QB pass_tds is derivative credit",
            "other_td_trials": 0, "other_note": "residual_carries: yards only; no defensive/ST/turnover scoring; no 2-pt attempt; scoring.py apply-events code is not imported by V23 _run_one (only `winner`)",
            "mutual_exclusion_present": False, "mutual_exclusion_note": "none among TD sites; FG is the only site guarded (block_points == 0)",
            "max_possible_tds_per_block": "sum(receptions)+sum(carries) <= plays <= 12 (draw_block_volume caps a block at 12 plays)", "max_possible_points_per_block": "7 * that <= 84",
            "paths_permitting_gt1_td": ["receiver loop: single receiver with rec>=2 and binomial>=2", "receiver loop: two or more receivers each with >=1", "carrier loop: same two forms",
                                        "receiver x carrier combination (receiver loop runs first, then carrier loop, no shared flag)"],
            "V23_SIMULATOR_SHA256": sha256_file(ROOT / "worker/sports_nova_v23/simulator.py"), "V23_run_one_line_count": v23_src.count("\n")},
        "RULE": "Once an offensive TD is awarded in a possession (block) no additional offensive TD is awarded in that block.  TD probabilities, allocation, loop order and the FG rule "
                "(block_points == 0 and plays >= 3) are unchanged.  Every rng call (each TD binomial and each PAT binomial) is still made with identical arguments; only the crediting "
                "of TDs after the first is suppressed.",
        "LOOP_ORDER_CONSEQUENCE": "the awarded TD is the FIRST TD in existing order: receivers in alloc.target_counts order, then carriers in alloc.carry_counts order.  A receiving TD therefore "
                                  "always pre-empts a rushing TD in the same block, and within a site the first receiver/carrier with td_count>0 wins.  This is recorded, not corrected (the ablation tests "
                                  "'one TD max', not 'better allocation').",
        "RNG_DESIGN": {"main_stream": "identical draws in both arms inside every block (FG rng.random() is consumed in the same blocks because a block has >=1 TD in baseline iff in ablation)",
                       "aux_stream": "SeedSequence([seed, sim_id, 0xCA9]) supplies ONE Bernoulli(PAT_MAKE_RATE) for the retained TD only when its raw td_count>1 (single-TD sites reuse the baseline PAT "
                                     "draw exactly).  Using min(made_pats,1) instead would raise the retained-PAT hit rate to 1-0.06^2 for those sites.",
                       "expected_divergence": "none until the first suppressed block; afterwards score differential changes script_pass_rate (game_script.py:16-20), which changes select_plays draws and "
                                              "desynchronises the main stream (numpy variable-consumption binomials).  Hence RNG_ALIGNMENT is PARTIAL by construction for ABL; CTRL replays the baseline "
                                              "score differential into the script so every draw is identical (isolation proof)."},
        "ARMS": {"REF": "unpatched V27 simulate_scoring", "BASE": "instrumented copy cap=False; MUST be np.array_equal to REF (all player/team arrays + winners, all 60 games) or the comparison is VOID",
                 "ABL": "cap=True, live score feedback (arm of record)", "CTRL": "cap=True, script differential replayed from BASE (component-isolation proof + exact block pairing for the >8 ledger)"},
        "ASSERTED_CHECKS": ["BASE == REF on all 60 games", "BASE team_score == stored FG_CAUSAL_ABLATION_V1 abl__team_score", "per-game causal fg_rate == FG ablation prereg rate to 1e-15",
                            "ABL/CTRL: max TDs per block == 1 and block points in {0,3,6,7}; GT8 == 0", "ABL: every block up to and including the first suppressed block identical to BASE in "
                            "volume/yields/targets/receptions (prefix_violations == 0); sims with no suppression bit-identical", "CTRL: block stream, every player/team stat except "
                            "TD-derived fields and score bit-identical to BASE", "accounting exact (score==sum block pts, TDs==player scored_tds, blocks, assert_path)", "guarded file hashes unchanged"],
        "PREDICTIONS_REGISTERED_BEFORE_ANY_CAPPED_RUN": {
            "BASIS": "FG_CAUSAL_ABLATION_V1 GT8 ledger (V27-equivalent arm): 4352 multi-TD blocks (3967x2, 359x3, 24x4, 1x5, 1x6) = 0.2833 blocks/team-game and 0.3103 extra TD/team-game; "
                     "reducing each to 6+0.94 removes 2.153 points/team-game.  Block-level P(>=1 TD) is unchanged by construction.",
            "TD_PER_TEAM_GAME": {"baseline": 2.60, "ablation": 2.29, "range": [2.24, 2.34], "observed_pilot": 2.575, "note": "material UNDERSHOOT of history predicted"},
            "POINTS_PER_TEAM_GAME": {"baseline": 23.005, "ablation": 20.85, "range": [20.6, 21.1], "observed_pilot": 22.683},
            "TOTAL_BIAS_VS_PILOT_OBSERVED": {"baseline": 0.644, "ablation": -3.7, "range": [-4.4, -3.0]},
            "FG_PER_TEAM_GAME": {"baseline": 1.630, "ablation": "unchanged +-0.03 (FG gate never re-opens)"},
            "GT8_BLOCKS_ABLATION": 0, "GT8_falsifier": "any nonzero count means the invariant is not enforced or another scoring path exists",
            "TIE_RATE": "rises from 3.85% (lower scoring, lower variance)",
            "TAILS": "P(team>=45) below baseline 4.6% and plausibly below history 1.8% (0.8-2.5%); P(total>=70) below baseline 8.6% (2-5%); the fall is partly the mean shift, not only shape",
            "PASS_RUSH_TD_MIX": "rush share of TDs falls (receiver loop pre-empts carrier loop)",
            "DISCRIMINATION": "corr(pred_total, obs_total) changes by <0.03 in absolute value (a near-constant shift); RMSE gets WORSE by 0.3-1.5 because of the new negative bias; TOTAL stays RED",
            "COMPONENTS": "CTRL: exactly 0 difference in plays, attempts, targets, receptions, yards, FG opportunities/rate, per-player allocations.  ABL: max |relative change| < 0.5% (game-script feedback only)",
            "PREFIX_ALIGNMENT": "prefix_violations == 0; sims without suppression 100% bit-identical",
            "FORECAST_STATUS": "WINNER YELLOW, TOTAL RED, TEAM_TOTAL RED unchanged"},
        "VERDICT_CLAUSES_PREREGISTERED": {
            "C1_GT8_removed": "ablation GT8 <= 5% of baseline GT8",
            "C2_tail_calibration_improves": "mean |P_sim - P_hist| over the 8 upper-tail ladder points (team>=35/40/45/50, total>=55/60/65/70) falls by >= 50% vs baseline",
            "C3_no_destructive_mean_bias": "|ablation total_mean - historical 2020-25 regulation total_mean| <= 1.5 pts AND ablation TD/team-game within +-5% of historical TD/team-game",
            "C4_accounting_integrity": "all accounting zero, BASE==REF, invariants hold, guarded hashes unchanged",
            "LABEL_RULE": "CAUSAL_MAJOR iff C1&C2&C3&C4.  If C1&C4 and (C2 or shape diagnostics) hold but C3 fails, the literal label is CAUSAL_MINOR and is annotated 'shape repair real, cap not mean-safe': "
                          "C3 failing measures the cap's collateral removal of TD mass, not the size of the defect.  NOT_PRIMARY if C1 holds but tails do not improve; INCONCLUSIVE if C4 or BASE==REF fails.",
            "DISCRIMINATION_IMPROVES": "Y only if the paired-bootstrap (over games) 95% CI of corr_abl - corr_base or of the bias-removed-RMSE change excludes 0 in the improving direction; otherwise N.  "
                                       "TOTAL_FORWARD_STATUS may leave RED only if ablation RMSE < league-mean RMSE with the CI excluding equality.",
            "PERMANENT_FIX_READY": "N if C3 fails or discrimination does not improve"},
        "CONCURRENCY_CHECK": concurrency(), "LIVE_CAPITAL_AUTHORIZED": False, "MARKET_INPUTS_READ": 0, "SEALED_OOS_INSPECTED": False, "KALSHI_PRICES_USED": False,
    }
    assert_market_free({k: rec[k] for k in ("SCHEMA", "BASE", "MODE", "RULE", "COHORT", "PHASE1_TD_PATH_STATIC")})
    OUT.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print("frozen", PREREG, "script", rec["SCRIPT_SHA256_AT_FREEZE"][:16], "codex procs", rec["CONCURRENCY_CHECK"]["codex_processes_running"])


def cmd_run(limit: int | None, workers: int) -> None:
    pre = json.loads(PREREG.read_text())
    games = pre["COHORT"]["game_ids"]
    assert games == H.pilot_games(list(json.loads(H.MANIFEST.read_text())["GAME_IDS"])), "pilot cohort drifted"
    before = guarded_hashes()
    assert before == pre["GUARDED_FILE_SHA256"], "a guarded engine file changed after the freeze"
    if limit:
        games = games[:limit]
    t0 = time.time()
    with Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(run_game, [(g, H.N_SIMS) for g in games]):
            results.append(r)
            a_abl = sum(v for k, v in r["accounting"]["abl"].items() if k != "max_td_per_block")
            print(f"{r['game']} {r['seconds']}s base==ref:{r['base_equals_ref']['all']} ctrl_stream:{r['ctrl_block_stream_identical_to_base']} "
                  f"prefix_viol:{r['prefix_abl_vs_base']['prefix_violations']} acct_abl:{a_abl}", flush=True)
    after = guarded_hashes()
    meta = {"prereg_sha256": sha256_file(PREREG), "script_sha256_at_run": sha256_file(Path(__file__)), "guarded_hashes_before": before, "guarded_hashes_after": after,
            "engine_unchanged": before == after, "n_games": len(results), "n_sims": H.N_SIMS, "wall_seconds": round(time.time() - t0, 1),
            "results": sorted(results, key=lambda r: r["game"])}
    (OUT / ("run_meta.json" if not limit else "run_meta_smoke.json")).write_text(json.dumps(meta, indent=1, default=str))
    print("DONE", meta["wall_seconds"], "engine_unchanged", meta["engine_unchanged"], "all_base_eq_ref", all(r["base_equals_ref"]["all"] for r in results))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "run"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmd_freeze() if a.cmd == "freeze" else cmd_run(a.limit, a.workers)
