"""Diagnostic-only pass over the same 60-game/128-sim cohort used by
SPORTS_NOVA_V3_ENGINE_REPAIR_VALIDATION_V1. Makes NO engine changes.

Goal: classify every zero-QB eval row and every >42-attempt QB row into a
concrete root cause, per SPORTS_NOVA_NODE3_V3_QB_ALLOCATION_REPAIR_V2 gates 1-2,
using the SAME cohort selection as sports_nova_v3_engine_repair_play_volume_qb_id_v1.py.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game, _primary_qb, _feature
from sports_nova_v3_player_joint_walkforward_v1 import (PLAYER, MANIFEST, game_parts,
    surrogate_kickoff, make_state)

OUT = ROOT / "data" / "sports_nova_v3"
N = 128

def main():
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    oos = all_df[all_df.GAME_ID.isin(selected)].copy()

    rows = []
    for g in selected:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        sim = simulate_game(state, N, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16), MODEL_VERSION)
        cur = oos[oos.GAME_ID == g]
        qcur = cur[cur.POSITION == "QB"]

        # what the engine actually resolved, per team, this game
        state_players = {p.player_id: p for p in state.players}
        primary = {}
        for tid in (home, away):
            pq = _primary_qb(state, tid)
            primary[tid] = pq.player_id if pq is not None else None

        for tid in (home, away):
            team_pass_att_sim = float(np.mean(sim.team(tid, "pass_attempts")))
            qb_att_sim_for_primary = (float(np.mean(sim.player(primary[tid], "pass_attempts")))
                                       if primary[tid] and primary[tid] in sim.player_team else None)
            team = state.home if state.home.team_id == tid else state.away
            team_pass_rate_feature = _feature(team, "pass_rate", None)
            raw_ratio = (float(prior[prior.TEAM == tid].PASS_ATTEMPTS.sum()) /
                         max(1.0, float(prior[prior.TEAM == tid].RUSH_ATTEMPTS.sum())))
            true_fraction = (float(prior[prior.TEAM == tid].PASS_ATTEMPTS.sum()) /
                             max(1.0, float(prior[prior.TEAM == tid].PASS_ATTEMPTS.sum() +
                                            prior[prior.TEAM == tid].RUSH_ATTEMPTS.sum())))

            for r in qcur[qcur.TEAM == tid].itertuples():
                pid = str(r.PLAYER_ID)
                in_roster = pid in state_players
                resolved_pos = state_players[pid].position if in_roster else None
                pass_rate_feat = _feature(state_players[pid], "pass_rate", None) if in_roster and resolved_pos == "QB" else None
                in_sim = pid in sim.player_team
                sim_att = float(np.mean(sim.player(pid, "pass_attempts"))) if in_sim else None
                is_primary = (pid == primary[tid])

                if not in_roster:
                    reason = "ROSTER_STATE_FAILURE"  # dropped by top-40 truncation or no prior history
                elif resolved_pos != "QB":
                    reason = "PLAYER_ID_MAPPING_FAILURE"  # position mode resolved away from QB
                elif primary[tid] is None:
                    reason = "NO_QB_SELECTED"
                elif not is_primary:
                    reason = "WRONG_QB_SELECTED"
                elif sim_att == 0:
                    reason = "TEAM_PASS_ATTEMPTS_NOT_ASSIGNED"
                else:
                    reason = "OK"

                rows.append({
                    "GAME_ID": g, "TEAM": tid, "EVAL_QB_ID": pid,
                    "REALIZED_ATTEMPTS": float(r.PASS_ATTEMPTS),
                    "IN_ROSTER": in_roster, "RESOLVED_POSITION": resolved_pos,
                    "IN_SIM": in_sim, "IS_PRIMARY": is_primary,
                    "PRIMARY_QB_ID": primary[tid], "PLAYER_PASS_RATE_FEATURE": pass_rate_feat,
                    "SIM_QB_ATTEMPTS": sim_att, "SIM_TEAM_PASS_ATTEMPTS": team_pass_att_sim,
                    "TEAM_PASS_RATE_FEATURE_STORED": team_pass_rate_feature,
                    "TEAM_PASS_RUSH_RAW_RATIO": raw_ratio, "TEAM_TRUE_PASS_FRACTION": true_fraction,
                    "ZERO": (sim_att == 0) if sim_att is not None else True,
                    "REASON": reason,
                })

    df = pd.DataFrame(rows)
    zero = df[df.ZERO]
    nonzero = df[~df.ZERO]
    high = nonzero[nonzero.SIM_QB_ATTEMPTS > 42]

    result = {
        "n_eval_rows": len(df),
        "zero_rate": float(df.ZERO.mean()),
        "zero_breakdown": zero.REASON.value_counts().to_dict(),
        "gt42_rate": float((nonzero.SIM_QB_ATTEMPTS > 42).mean()) if len(nonzero) else None,
        "gt50_rate": float((nonzero.SIM_QB_ATTEMPTS > 50).mean()) if len(nonzero) else None,
        "high_attempt_rows_is_primary_rate": float(high.IS_PRIMARY.mean()) if len(high) else None,
        "sim_team_pass_attempts_mean_overall": float(df.SIM_TEAM_PASS_ATTEMPTS.mean()),
        "team_pass_rate_feature_mean": float(df.TEAM_PASS_RATE_FEATURE_STORED.dropna().mean()),
        "team_pass_rush_raw_ratio_mean": float(df.TEAM_PASS_RUSH_RAW_RATIO.mean()),
        "team_true_pass_fraction_mean": float(df.TEAM_TRUE_PASS_FRACTION.mean()),
        "team_pass_rate_feature_saturated_at_08_rate": float((df.TEAM_PASS_RATE_FEATURE_STORED >= 0.8).mean()),
        "wrong_qb_rows_where_eval_qb_has_higher_pass_rate_than_primary": None,
    }

    # For WRONG_QB_SELECTED rows, check whether the eval QB actually had a
    # higher (more recent-relevant) pass_rate feature than the chosen primary,
    # to see if this is a pure tie-break/recency artifact.
    wrong = df[df.REASON == "WRONG_QB_SELECTED"].copy()
    if len(wrong):
        primary_rates = []
        for r in wrong.itertuples():
            g = r.GAME_ID; s, w, away, home = game_parts(g); key = s * 100 + w
            prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
            state = make_state(g, prior, surrogate_kickoff(s, w), None)
            sp = {p.player_id: p for p in state.players}
            pq = sp.get(r.PRIMARY_QB_ID)
            primary_rates.append(_feature(pq, "pass_rate", None) if pq else None)
        wrong["PRIMARY_PASS_RATE_FEATURE"] = primary_rates
        result["wrong_qb_rows_where_eval_qb_has_higher_pass_rate_than_primary"] = float(
            (wrong.PLAYER_PASS_RATE_FEATURE > wrong.PRIMARY_PASS_RATE_FEATURE).mean())
        wrong.to_csv(OUT / "SPORTS_NOVA_V3_QB_DIAG_WRONG_QB_ROWS_V1.csv", index=False)

    df.to_csv(OUT / "SPORTS_NOVA_V3_QB_DIAG_ALL_ROWS_V1.csv", index=False)
    (OUT / "SPORTS_NOVA_V3_QB_DIAG_SUMMARY_V1.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
