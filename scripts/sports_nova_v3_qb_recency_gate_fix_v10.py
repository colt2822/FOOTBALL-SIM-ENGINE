"""V10: SPORTS_NOVA_V10_QB_RECENCY_GATE_FIX post-fix validation.

Diagnostic-only script wrapping the SN3_V10_QB_RECENCY_GATE_FIX repair already
applied in worker/sports_nova_v3/simulator.py:_qb_shares (the hard
`<= min_gap` binary eligibility filter was replaced with a continuous,
parameter-free recency decay 1/(1+gap) multiplied into the existing
pass_rate*effective_sample_size volume score; **2.5 sharpening, _primary_qb,
and make_state are unchanged). This script makes NO further engine edits.

Identical 60-game/120-team-game/128-sim cohort as
sports_nova_v3_qb_attempt_bias_decomp_v9.py. Reclassifies each identity-miss
team-game against the SAME mechanism taxonomy traced for V9
(GAP_FILTER_EXCLUDED_REALIZED_STARTER, SURVIVED_FILTER_LOST_VOLUME_WEIGHT,
REALIZED_STARTER_NOT_QB_ROSTER, NO_REALIZED_STARTER_DATA), now evaluated
post-fix: a team-game is "gap-caused" post-fix if the realized starter's
prior-fix gap was > 0 (i.e. would have been hard-excluded under the removed
filter), regardless of whether the fix actually flipped that team-game to the
correct pick. REALIZED_STARTER_NOT_QB_ROSTER and NO_REALIZED_STARTER_DATA are
reported as UNKNOWN_QB_IDENTITY per the no-silent-substitution rule and
excluded from the identity-accuracy denominator, but counted explicitly.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game, _primary_qb, _feature
from sports_nova_v3_player_joint_walkforward_v1 import (PLAYER, MANIFEST,
    game_parts, surrogate_kickoff, make_state)

OUT = ROOT / "data" / "sports_nova_v3"
N = 128

BEFORE = json.loads((OUT / "SPORTS_NOVA_V9_QB_ATTEMPT_BIAS_REPORT.json").read_text())


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

    qb_rows, team_rows = [], []
    for g in selected:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        sim = simulate_game(state, N, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16), MODEL_VERSION)
        cur = oos[oos.GAME_ID == g]
        qcur = cur[cur.POSITION == "QB"]
        state_players = {p.player_id: p for p in state.players}
        primary = {}
        for tid in (home, away):
            pq = _primary_qb(state, tid)
            primary[tid] = pq.player_id if pq is not None else None

        for tid in (home, away):
            team_qb_rows = qcur[qcur.TEAM == tid]
            realized_starter = (str(team_qb_rows.sort_values("PASS_ATTEMPTS", ascending=False).iloc[0].PLAYER_ID)
                                 if len(team_qb_rows) else None)
            qbs_on_team = [p for p in state.players if p.team_id == tid and p.position == "QB"]
            ids_full = [p.player_id for p in qbs_on_team]
            gaps = {p.player_id: _feature(p, "games_since_last_team_game", 0.0) for p in qbs_on_team}
            min_gap = min(gaps.values()) if gaps else None

            if realized_starter is None:
                identity_status = "UNKNOWN_QB_IDENTITY_NO_DATA"
            elif realized_starter not in ids_full:
                identity_status = "UNKNOWN_QB_IDENTITY_NOT_IN_ROSTER"
            elif primary[tid] == realized_starter:
                identity_status = "OK"
            elif gaps.get(realized_starter, 0.0) > min_gap:
                identity_status = "MISS_GAP_CAUSED"
            else:
                identity_status = "MISS_TIED_WEIGHT"

            team_rows.append({"GAME_ID": g, "TEAM": tid, "SEASON": s,
                "MODEL_STARTER": primary[tid], "REALIZED_STARTER": realized_starter,
                "IDENTITY_STATUS": identity_status,
                "REALIZED_STARTER_GAP": gaps.get(realized_starter) if realized_starter in gaps else None,
                "MIN_GAP": min_gap})

            for r in team_qb_rows.itertuples():
                pid = str(r.PLAYER_ID)
                in_sim = pid in sim.player_team
                sim_att = float(np.mean(sim.player(pid, "pass_attempts"))) if in_sim else 0.0
                sim_yds = float(np.mean(sim.player(pid, "pass_yards"))) if in_sim else 0.0
                qb_rows.append({"GAME_ID": g, "TEAM": tid, "SEASON": s, "QB_ID": pid,
                    "REAL_ATT": float(r.PASS_ATTEMPTS), "SIM_ATT": sim_att,
                    "REAL_YDS": float(r.PASS_YARDS), "SIM_YDS": sim_yds,
                    "MATCHED": bool(pid == primary[tid]),
                    "IDENTITY_STATUS_TEAM_GAME": identity_status})

    qb = pd.DataFrame(qb_rows)
    ts = pd.DataFrame(team_rows)
    qb["ATT_ERR"] = qb.SIM_ATT - qb.REAL_ATT
    qb["YDS_AE"] = (qb.SIM_YDS - qb.REAL_YDS).abs()
    qb["YDS_SE"] = (qb.SIM_YDS - qb.REAL_YDS) ** 2

    resolved = ts[~ts.IDENTITY_STATUS.str.startswith("UNKNOWN_QB_IDENTITY")]
    unknown = ts[ts.IDENTITY_STATUS.str.startswith("UNKNOWN_QB_IDENTITY")]
    identity_miss_after = int((resolved.IDENTITY_STATUS != "OK").sum())
    gap_caused_after = int((resolved.IDENTITY_STATUS == "MISS_GAP_CAUSED").sum())
    tied_weight_after = int((resolved.IDENTITY_STATUS == "MISS_TIED_WEIGHT").sum())

    result = {
        "mission": "SPORTS_NOVA_V10_QB_RECENCY_GATE_FIX",
        "cohort": {"games": len(selected), "team_games": len(ts), "qb_eval_rows": len(qb)},
        "identity": {
            "resolved_team_games": int(len(resolved)),
            "unknown_team_games": int(len(unknown)),
            "unknown_breakdown": unknown.IDENTITY_STATUS.value_counts().to_dict(),
            "identity_miss_after": identity_miss_after,
            "identity_match_rate_after_resolved_only": float((resolved.IDENTITY_STATUS == "OK").mean()),
            "gap_caused_miss_after": gap_caused_after,
            "tied_weight_miss_after": tied_weight_after,
            "still_mismatched_games": ts[ts.IDENTITY_STATUS.isin(["MISS_GAP_CAUSED", "MISS_TIED_WEIGHT"])][
                ["GAME_ID", "TEAM", "IDENTITY_STATUS", "REALIZED_STARTER_GAP", "MIN_GAP"]].to_dict("records"),
        },
        "attempts": {
            "real_attempts_mean": float(qb.REAL_ATT.mean()),
            "sim_attempts_mean_after": float(qb.SIM_ATT.mean()),
            "attempt_bias_after": float(qb.ATT_ERR.mean()),
            "attempt_bias_matched_only_after": float(qb[qb.MATCHED].ATT_ERR.mean()) if qb.MATCHED.any() else None,
        },
        "qb_yards": {
            "mae_after": float(qb.YDS_AE.mean()),
            "rmse_after": float(np.sqrt(qb.YDS_SE.mean())),
            "mae_matched_only_after": float(qb[qb.MATCHED].YDS_AE.mean()) if qb.MATCHED.any() else None,
        },
    }

    # BEFORE numbers pulled from the frozen V9 report (same cohort, pre-fix code).
    before_bias = BEFORE["TOTAL_BIAS"]
    before_sim_mean = BEFORE["SIM_ATTEMPTS_MEAN"]
    before_real_mean = BEFORE["REAL_ATTEMPTS_MEAN"]
    before_identity_miss = BEFORE["by_reason"]["WRONG_QB_SELECTED_IDENTITY_MISS"]["N"] + \
        BEFORE["by_reason"]["WRONG_QB_SELECTED_CONCENTRATION_MISS"]["N"]
    gap_caused_before = 12  # traced directly against this same V9 cohort in the prior investigation turn

    attempt_bias_after = result["attempts"]["attempt_bias_after"]
    qb_mae_after = result["qb_yards"]["mae_after"]

    if before_real_mean:
        pass

    verdict = "FIX_CONFIRMED" if (gap_caused_after <= 2 and abs(attempt_bias_after) < abs(before_bias)) else "FIX_INCOMPLETE"
    if verdict == "FIX_CONFIRMED":
        if abs(attempt_bias_after) <= 1.0 and qb_mae_after <= 61.5:
            next_step = "ADVANCE_FULL_1693_OOS"
        else:
            next_step = "DIAGNOSE_NEXT_CENTERING_COMPONENT"
    else:
        next_step = "FIX_INCOMPLETE_INVESTIGATE_REMAINING_MISSES"

    result["comparison"] = {
        "IDENTITY_MISS_BEFORE": before_identity_miss,
        "IDENTITY_MISS_AFTER": identity_miss_after,
        "GAP_CAUSED_MISS_BEFORE": gap_caused_before,
        "GAP_CAUSED_MISS_AFTER": gap_caused_after,
        "UNKNOWN_IDENTITY": int(len(unknown)),
        "REAL_ATTEMPTS_MEAN": before_real_mean,
        "SIM_ATTEMPTS_MEAN_BEFORE": before_sim_mean,
        "SIM_ATTEMPTS_MEAN_AFTER": result["attempts"]["sim_attempts_mean_after"],
        "ATTEMPT_BIAS_BEFORE": before_bias,
        "ATTEMPT_BIAS_AFTER": attempt_bias_after,
        "QB_MAE_AFTER": qb_mae_after,
        "QB_RMSE_AFTER": result["qb_yards"]["rmse_after"],
        "VERDICT": verdict,
        "NEXT_STEP": next_step,
    }

    qb.to_csv(OUT / "SPORTS_NOVA_V10_QB_ATTEMPT_BIAS_ROWS.csv", index=False)
    ts.to_csv(OUT / "SPORTS_NOVA_V10_QB_IDENTITY_TEAM_STAGE.csv", index=False)
    (OUT / "SPORTS_NOVA_V10_QB_RECENCY_GATE_FIX_REPORT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["comparison"], indent=2))
    print(json.dumps(result["identity"]["unknown_breakdown"], indent=2))
    print(json.dumps(result["identity"]["still_mismatched_games"], indent=2))


if __name__ == "__main__":
    main()
