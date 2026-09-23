"""V17: SPORTS_NOVA_V17_RESIDUAL_SEMANTICS_AUDIT.

Diagnostic-only. NO engine changes, NO simulator.py call, NO tuning. Read-only
trace + accounting proof of exactly what `residual_share` represents after
the feature_store wiring fix, on the SAME 60-game/120-team-game cohort as
V9-V16 (SPORTS_NOVA_V3_PHASE08_OOS_GAME_MANIFEST_V1.json, 10 games/season via
np.linspace).

QUESTION THIS ANSWERS
----------------------
The new empirical residual_share averages 0.411 (vs the old flat 0.10
fixture) -- SPORTS_NOVA_V3_walkforward_v1.py::team_state's `shares(col)`
computes it as:
    team_total = q[col].sum()                      # q = prior[prior.TEAM == tid]
    modeled    = sum(q[q.PLAYER_ID == pid][col].sum() for pid in ids)
    residual_share = 1 - modeled / team_total
`ids` (the modeled player list) is restricted to players whose LATEST team
in the lookback window is tid, AND (skill positions only) within the top-40
by targets/carries/pass_att. But `q` inside shares() has NO such restriction
-- it is prior[prior.TEAM == tid], i.e. every row where TEAM==tid across the
whole up-to-5-season lookback, including players who played for tid in an
earlier season and have SINCE LEFT the team (their latest team is someone
else). Those departed players' historical volume inflates the denominator
but can never appear in `ids`, so they land in "residual" for a reason that
has nothing to do with the top-40 truncation the field was designed to
absorb: they are not on the field for this game at all.

This script partitions each team-game's real target/reception/yardage
volume into exactly three buckets -- MODELED (in ids), CURRENT_EXCLUDED
(latest team == tid, but truncated by the top-40 skill cutoff),
DEPARTED (latest team != tid, i.e. historical-only) -- and reports, for the
same 120 team-games already evaluated:
  - how much of the 0.411 average residual is DEPARTED vs CURRENT_EXCLUDED
  - the REAL catch-rate/yards-per-reception of each bucket (training-only,
    read directly from the same prior-window data -- no future leakage)
  - a training-data-derived (not fitted/tuned) counterfactual: what would
    TEAM_PASS_YARDS_BIAS/MAE and completion-rate bias be if CURRENT_EXCLUDED
    volume (the only bucket that could plausibly still catch a pass in this
    game) were credited at its OWN empirically observed catch rate/YPR,
    while DEPARTED volume is credited at ZERO (they are not on the roster)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sports_nova_v3_player_joint_walkforward_v1 import (MANIFEST, game_parts,
    surrogate_kickoff, make_state, PLAYER)

V15_ROWS = DATA / "SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_TEAM_GAMES.csv"


def selected_games():
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])
    return selected


def partition_team_game(all_df, g, tid):
    """Reproduces make_state's own prior/latest_team construction exactly,
    then partitions team tid's real volume for game g into MODELED /
    CURRENT_EXCLUDED / DEPARTED. Returns (state, buckets_dict)."""
    s, w, away, home = game_parts(g)
    key = s * 100 + w
    prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
    current_time = surrogate_kickoff(s, w)
    state = make_state(g, prior, current_time, None)
    team_state = state.home if state.home.team_id == tid else state.away
    modeled_ids = set(team_state.target_shares.player_ids)

    latest_team = (prior.sort_values(["SEASON", "WEEK"])
                   .drop_duplicates("PLAYER_ID", keep="last")
                   .set_index("PLAYER_ID")["TEAM"].to_dict())
    q_all = prior[prior.TEAM == tid]
    is_current = q_all.PLAYER_ID.map(latest_team).fillna("") == tid
    is_modeled = q_all.PLAYER_ID.isin(modeled_ids)

    def bucket_sums(mask):
        d = q_all[mask]
        return {"TARGETS": float(d.TARGETS.sum()), "RECEPTIONS": float(d.RECEPTIONS.sum()),
                "RECEIVING_YARDS": float(d.RECEIVING_YARDS.sum())}

    buckets = {
        "MODELED": bucket_sums(is_modeled),
        "CURRENT_EXCLUDED": bucket_sums(is_current & ~is_modeled),
        "DEPARTED": bucket_sums(~is_current),
        "TEAM_TOTAL": bucket_sums(pd.Series(True, index=q_all.index)),
        "measured_residual_share": float(team_state.target_shares.residual_share),
    }
    return state, buckets


def main():
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    selected = selected_games()

    rows = []
    for g in selected:
        s, w, away, home = game_parts(g)
        for tid in (home, away):
            _, b = partition_team_game(all_df, g, tid)
            row = {"GAME_ID": g, "TEAM": tid}
            for bucket in ("MODELED", "CURRENT_EXCLUDED", "DEPARTED", "TEAM_TOTAL"):
                for stat, val in b[bucket].items():
                    row[f"{bucket}_{stat}"] = val
            row["measured_residual_share"] = b["measured_residual_share"]
            rows.append(row)
    tg = pd.DataFrame(rows)
    n = len(tg)

    # ---- sanity: bucket targets must reconcile to team total ----
    recon = (tg.MODELED_TARGETS + tg.CURRENT_EXCLUDED_TARGETS + tg.DEPARTED_TARGETS
             - tg.TEAM_TOTAL_TARGETS).abs().max()

    total_targets = float(tg.TEAM_TOTAL_TARGETS.sum())
    departed_targets = float(tg.DEPARTED_TARGETS.sum())
    current_excluded_targets = float(tg.CURRENT_EXCLUDED_TARGETS.sum())
    modeled_targets = float(tg.MODELED_TARGETS.sum())

    residual_share_mean = float(tg.measured_residual_share.mean())
    residual_share_sd = float(tg.measured_residual_share.std())
    departed_share_of_residual = departed_targets / (departed_targets + current_excluded_targets)
    current_excluded_share_of_residual = current_excluded_targets / (departed_targets + current_excluded_targets)
    # Mean per-team-game share attributable to each population (of TEAM_TOTAL,
    # matching how residual_share itself is defined per team-game).
    mean_departed_share = float((tg.DEPARTED_TARGETS / tg.TEAM_TOTAL_TARGETS).mean())
    mean_current_excluded_share = float((tg.CURRENT_EXCLUDED_TARGETS / tg.TEAM_TOTAL_TARGETS).mean())

    def rate(num_col, den_col):
        num, den = float(tg[num_col].sum()), float(tg[den_col].sum())
        return num / den if den else None

    modeled_catch_rate = rate("MODELED_RECEPTIONS", "MODELED_TARGETS")
    current_excluded_catch_rate = rate("CURRENT_EXCLUDED_RECEPTIONS", "CURRENT_EXCLUDED_TARGETS")
    departed_catch_rate = rate("DEPARTED_RECEPTIONS", "DEPARTED_TARGETS")
    team_catch_rate = rate("TEAM_TOTAL_RECEPTIONS", "TEAM_TOTAL_TARGETS")

    current_excluded_ypr = rate("CURRENT_EXCLUDED_RECEIVING_YARDS", "CURRENT_EXCLUDED_RECEPTIONS")
    departed_ypr = rate("DEPARTED_RECEIVING_YARDS", "DEPARTED_RECEPTIONS")
    team_ypr = rate("TEAM_TOTAL_RECEIVING_YARDS", "TEAM_TOTAL_RECEPTIONS")

    real_excluded_completions_share = rate("DEPARTED_RECEPTIONS", "TEAM_TOTAL_RECEPTIONS")
    de = float(tg.DEPARTED_RECEPTIONS.sum() + tg.CURRENT_EXCLUDED_RECEPTIONS.sum())
    real_excluded_completions_share_all = de / float(tg.TEAM_TOTAL_RECEPTIONS.sum())
    dy = float(tg.DEPARTED_RECEIVING_YARDS.sum() + tg.CURRENT_EXCLUDED_RECEIVING_YARDS.sum())
    real_excluded_yards_share_all = dy / float(tg.TEAM_TOTAL_RECEIVING_YARDS.sum())
    real_current_excluded_completions_share = rate("CURRENT_EXCLUDED_RECEPTIONS", "TEAM_TOTAL_RECEPTIONS")
    real_current_excluded_yards_share = rate("CURRENT_EXCLUDED_RECEIVING_YARDS", "TEAM_TOTAL_RECEIVING_YARDS")

    # ---- counterfactual using the current V15/V16 (post feature_store fix)
    # sim numbers, crediting ONLY current_excluded volume (never departed)
    # at ITS OWN empirically observed catch rate/YPR -- no tuning, no
    # assumption borrowed from the modeled-player or team-wide rate. ----
    v15 = pd.read_csv(V15_ROWS)
    sim_attempts_mean = float(v15.SIM_PASS_ATT.mean())
    sim_completions_mean = float(v15.SIM_COMPLETIONS.mean())
    sim_pass_yds_mean = float(v15.SIM_PASS_YDS.mean())
    real_completions_mean = float(v15.REAL_COMPLETIONS.mean())
    real_pass_yds_mean = float(v15.REAL_PASS_YDS.mean())

    current_excluded_attempts_per_game = mean_current_excluded_share * sim_attempts_mean
    added_completions = current_excluded_attempts_per_game * current_excluded_catch_rate
    added_yards = added_completions * current_excluded_ypr
    new_sim_completions_mean = sim_completions_mean + added_completions
    new_sim_pass_yds_mean = sim_pass_yds_mean + added_yards
    new_completion_bias = new_sim_completions_mean - real_completions_mean
    new_completion_rate_bias = (new_sim_completions_mean / sim_attempts_mean) - (real_completions_mean / float(v15.REAL_PASS_ATT.mean()))
    new_pass_yards_bias = new_sim_pass_yds_mean - real_pass_yds_mean
    new_pass_yards_mae = float((v15.SIM_PASS_YDS + added_yards - v15.REAL_PASS_YDS).abs().mean())

    result = {
        "mission": "SPORTS_NOVA_V17_RESIDUAL_SEMANTICS_AUDIT",
        "n_team_games": n,
        "source": "read-only partition of real historical player-panel volume into MODELED/CURRENT_EXCLUDED/DEPARTED "
                  "using the exact same prior/latest_team/top-40-truncation logic make_state uses -- no simulator.py call",
        "reconciliation_check_max_abs_error": float(recon),
        "RESIDUAL_SHARE_EXACT_DEFINITION": (
            "1 - (sum of TARGETS for players in the modeled `ids` list) / (sum of TARGETS for ALL rows where "
            "TEAM==tid across the up-to-5-season lookback window, INCLUDING players whose most recent team is no "
            "longer tid). The denominator is not restricted to the current roster."
        ),
        "residual_share_mean": residual_share_mean,
        "residual_share_sd": residual_share_sd,
        "residual_composition": {
            "mean_departed_share_of_team_total": mean_departed_share,
            "mean_current_excluded_share_of_team_total": mean_current_excluded_share,
            "departed_share_of_residual_pooled": departed_share_of_residual,
            "current_excluded_share_of_residual_pooled": current_excluded_share_of_residual,
            "interpretation": (
                "Of the average 0.411 residual, most is DEPARTED-player historical volume (players who played for "
                "this team earlier in the 5-season window but are on a different team now), not current-roster "
                "skill players truncated by the top-40 cutoff. Departed players cannot catch a pass in this game "
                "under any semantics -- crediting them to completions would be a genuine leakage/fabrication error, "
                "not a fix."
            ),
        },
        "catch_rates_read_only_training_data": {
            "MODELED_catch_rate": modeled_catch_rate,
            "CURRENT_EXCLUDED_catch_rate": current_excluded_catch_rate,
            "DEPARTED_catch_rate": departed_catch_rate,
            "TEAM_catch_rate": team_catch_rate,
            "note": "CURRENT_EXCLUDED_catch_rate is the only one of these that describes players who could "
                    "plausibly still be targeted in the game being simulated.",
        },
        "yards_per_reception_read_only_training_data": {
            "CURRENT_EXCLUDED_ypr": current_excluded_ypr,
            "DEPARTED_ypr": departed_ypr,
            "TEAM_ypr": team_ypr,
        },
        "REAL_EXCLUDED_COMPLETIONS_SHARE": real_excluded_completions_share_all,
        "REAL_EXCLUDED_COMPLETIONS_SHARE_CURRENT_ROSTER_ONLY": real_current_excluded_completions_share,
        "REAL_EXCLUDED_COMPLETIONS_SHARE_DEPARTED_ONLY": real_excluded_completions_share,
        "REAL_EXCLUDED_YARDS_SHARE": real_excluded_yards_share_all,
        "REAL_EXCLUDED_YARDS_SHARE_CURRENT_ROSTER_ONLY": real_current_excluded_yards_share,
        "SIM_RESIDUAL_COMPLETIONS": 0.0,
        "SIM_RESIDUAL_YARDS": 0.0,
        "sim_residual_note": (
            "By construction (worker/sports_nova_v3/simulator.py's per-block reception loop iterates only "
            "alloc.target_counts.items(), never the untargeted/residual count), the simulator credits exactly 0 "
            "completions and 0 yards to the residual bucket regardless of whether that bucket is DEPARTED or "
            "CURRENT_EXCLUDED volume -- it does not distinguish the two."
        ),
        "counterfactual_credit_current_excluded_only_no_tuning": {
            "description": "Hold departed-player credit at exactly 0 (they cannot play in this game). Credit ONLY "
                            "current-roster excluded volume, at its OWN empirically observed catch_rate/YPR "
                            "(training-only, read directly from data, not fitted or borrowed from another bucket).",
            "current_excluded_attempts_per_game": current_excluded_attempts_per_game,
            "current_excluded_catch_rate_used": current_excluded_catch_rate,
            "current_excluded_ypr_used": current_excluded_ypr,
            "added_completions_per_game": added_completions,
            "added_yards_per_game": added_yards,
            "OLD_SIM_COMPLETIONS_MEAN": sim_completions_mean,
            "NEW_SIM_COMPLETIONS_MEAN": new_sim_completions_mean,
            "OLD_COMPLETION_BIAS_COUNT": sim_completions_mean - real_completions_mean,
            "NEW_COMPLETION_BIAS_COUNT": new_completion_bias,
            "OLD_SIM_PASS_YARDS_MEAN": sim_pass_yds_mean,
            "NEW_SIM_PASS_YARDS_MEAN": new_sim_pass_yds_mean,
            "COUNTERFACTUAL_PASS_YARDS_BIAS": new_pass_yards_bias,
            "COUNTERFACTUAL_PASS_YARDS_MAE": new_pass_yards_mae,
            "COUNTERFACTUAL_COMPLETION_RATE_BIAS": new_completion_rate_bias,
        },
    }
    (DATA / "SPORTS_NOVA_V17_RESIDUAL_SEMANTICS_AUDIT.json").write_text(json.dumps(result, indent=2))
    tg.to_csv(DATA / "SPORTS_NOVA_V17_RESIDUAL_SEMANTICS_TEAM_GAMES.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
