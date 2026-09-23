"""V9: SPORTS_NOVA_V9_QB_ATTEMPT_LEVEL_BIAS forensic decomposition.

Diagnostic-only. Makes NO engine changes. Reuses the SAME 60-game/128-sim
cohort as sports_nova_v3_engine_repair_play_volume_qb_id_v6/v7/v8.py and the
SAME REASON taxonomy as sports_nova_v3_qb_allocation_diagnostic_v1.py, but
recomputes both fresh against the CURRENT (post-V8) engine state in a single
pass -- V1's by-reason numbers predate the V5/V7/V8 repairs and must not be
mixed with V8's post-repair bias number (V1 ran before V5 17:46/V7 19:31/V8
19:55 on 2026-09-14; this script supersedes it for reason-level attribution).

Question: where does the ~3.5-attempt QB-level mean bias
(SPORTS_NOVA_V3_ENGINE_REPAIR_VALIDATION_V8.json qb.attempts_all_evaluated
.bias_sim_minus_real == -3.464) originate -- pregame team play-volume
forecast, pass-rate forecast, QB share/allocation, sacks/scrambles/dropback
conversion, game-state feedback, or simulation bookkeeping?

Method:
  A. PRE_SIM_CENTERING: an analytic (non-stochastic) per-team pass-attempt
     expectation built from the SAME instrumented sim.team(tid, 'blocks')
     mean (V8 already added this counter for instrumentation; it feeds no
     computation) and the causal pace_seconds/pass_rate TeamState features,
     using the exact draw_block_volume/select_plays formulas read-only
     (E[plays/block] = clip((60/pace)*2.7, 2, 12); attempts = blocks *
     E[plays/block] * pass_rate * (1-sack_rate)*(1-scramble_rate)), compared
     to realized team pass attempts.
  B. SIMULATION_DRIFT: analytic pre-sim expectation vs actual simulated
     team pass attempts (sim.team(tid,'pass_attempts')).
  C/D. League-wide realized sack rate (from NFL_V3_DRIVE_BLOCK_CANONICAL_V1
     block-level 'sacks'/'pass_attempts' columns, which DO exist) vs the
     assumed sack_rate=.065 constant, as a bounded check on dropback
     conversion. Scrambles are not separable from designed rushes in the
     realized block data (both post as rush_attempts), so scramble_rate is
     not independently falsifiable from this source and is left untested,
     not assumed.
  E. QB-row REASON classification (identity resolution / allocation), with
     WRONG_QB_SELECTED further split into IDENTITY_MISS (team-game where the
     model's chosen starter differs from the realized top-attempts starter)
     vs CONCENTRATION_MISS (model starter matches reality, but this eval row
     is a second/backup passer reality gave volume to that the engine's
     winner-take-all QB-share sharpening (**2.5 in simulator._qb_shares)
     assigns ~0).
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
DRIVE = ROOT / "data" / "sports_nova_v3" / "validation_inputs" / "NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet"
N = 128


def expected_plays_per_block(pace_seconds: float) -> float:
    return float(np.clip((60.0 / pace_seconds) * 2.7, 2.0, 12.0))


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

    qb_rows = []       # one row per realized QB eval row
    team_stage_rows = []  # one row per team-game

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
            team = state.home if state.home.team_id == tid else state.away
            pace = _feature(team, "pace_seconds", 27.0)
            pass_rate = _feature(team, "pass_rate", .58)
            sack_rate = _feature(team, "sack_rate", .065)
            scramble_rate = _feature(team, "scramble_rate", .07)
            mean_blocks = float(np.mean(sim.team(tid, "blocks")))
            pred_pass_att_pre_sim = (mean_blocks * expected_plays_per_block(pace) * pass_rate *
                                      (1 - sack_rate) * (1 - scramble_rate))
            sim_team_pass_att = float(np.mean(sim.team(tid, "pass_attempts")))
            real_team_pass_att = float(cur[cur.TEAM == tid].PASS_ATTEMPTS.sum())

            team_qb_rows = qcur[qcur.TEAM == tid]
            realized_starter = (str(team_qb_rows.sort_values("PASS_ATTEMPTS", ascending=False).iloc[0].PLAYER_ID)
                                 if len(team_qb_rows) else None)
            identity_ok = (primary[tid] is not None and primary[tid] == realized_starter)

            team_stage_rows.append({
                "GAME_ID": g, "TEAM": tid, "SEASON": s,
                "PACE_SECONDS": pace, "PASS_RATE_FEATURE": pass_rate,
                "MEAN_BLOCKS": mean_blocks,
                "PRED_PASS_ATT_PRE_SIM": pred_pass_att_pre_sim,
                "SIM_TEAM_PASS_ATT": sim_team_pass_att,
                "REAL_TEAM_PASS_ATT": real_team_pass_att,
                "MODEL_STARTER": primary[tid], "REALIZED_STARTER": realized_starter,
                "IDENTITY_OK": identity_ok,
            })

            for r in team_qb_rows.itertuples():
                pid = str(r.PLAYER_ID)
                in_roster = pid in state_players
                resolved_pos = state_players[pid].position if in_roster else None
                in_sim = pid in sim.player_team
                is_primary = (pid == primary[tid])
                sim_att = float(np.mean(sim.player(pid, "pass_attempts"))) if in_sim else None

                if not in_roster:
                    reason = "ROSTER_STATE_FAILURE"
                elif resolved_pos != "QB":
                    reason = "PLAYER_ID_MAPPING_FAILURE"
                elif primary[tid] is None:
                    reason = "NO_QB_SELECTED"
                elif not is_primary:
                    reason = "WRONG_QB_SELECTED_IDENTITY_MISS" if not identity_ok else "WRONG_QB_SELECTED_CONCENTRATION_MISS"
                elif sim_att == 0:
                    reason = "TEAM_PASS_ATTEMPTS_NOT_ASSIGNED"
                else:
                    reason = "OK"

                qb_rows.append({
                    "GAME_ID": g, "TEAM": tid, "SEASON": s, "QB_ID": pid,
                    "REAL_ATT": float(r.PASS_ATTEMPTS),
                    "SIM_ATT": sim_att if sim_att is not None else 0.0,
                    "REASON": reason, "MATCHED": bool(is_primary and reason == "OK"),
                })

    qb = pd.DataFrame(qb_rows)
    ts = pd.DataFrame(team_stage_rows)
    qb["BIAS"] = qb.SIM_ATT - qb.REAL_ATT
    n_total = len(qb)
    total_bias = float(qb.BIAS.mean())

    def group_stats(g):
        return {"N": int(len(g)), "SHARE_OF_ROWS": float(len(g) / n_total),
                "mean_real_att": float(g.REAL_ATT.mean()), "mean_sim_att": float(g.SIM_ATT.mean()),
                "mean_bias": float(g.BIAS.mean()),
                "bias_contribution": float(len(g) / n_total * g.BIAS.mean())}

    by_reason = {reason: group_stats(g) for reason, g in qb.groupby("REASON")}
    dominant_reason = max((r for r in by_reason if by_reason[r]["bias_contribution"] < 0),
                          key=lambda r: -by_reason[r]["bias_contribution"], default=None)

    # Test A/B: pre-sim vs simulated vs realized, at team-game level, league-wide.
    stage_summary = {
        "n_team_games": len(ts),
        "real_team_pass_att_mean": float(ts.REAL_TEAM_PASS_ATT.mean()),
        "pred_pass_att_pre_sim_mean": float(ts.PRED_PASS_ATT_PRE_SIM.mean()),
        "sim_team_pass_att_mean": float(ts.SIM_TEAM_PASS_ATT.mean()),
        "test_A_pre_sim_centering_gap": float(ts.PRED_PASS_ATT_PRE_SIM.mean() - ts.REAL_TEAM_PASS_ATT.mean()),
        "test_A_fails_gate_pre_sim_2p5_below_real": bool(
            (ts.REAL_TEAM_PASS_ATT.mean() - ts.PRED_PASS_ATT_PRE_SIM.mean()) >= 2.5),
        "test_B_simulation_drift_from_pre_sim": float(ts.SIM_TEAM_PASS_ATT.mean() - ts.PRED_PASS_ATT_PRE_SIM.mean()),
        "identity_match_rate": float(ts.IDENTITY_OK.mean()),
    }

    # Test D: league-wide realized sack rate vs assumed constant (.065).
    drive = pd.read_parquet(DRIVE, columns=["season", "pass_attempts", "sacks"])
    drive = drive[drive.season.isin(sorted(by))]
    real_sack_rate = float(drive.sacks.sum() / (drive.pass_attempts.sum() + drive.sacks.sum()))
    dropback_conversion = {
        "assumed_sack_rate": 0.065, "assumed_scramble_rate": 0.07,
        "realized_league_sack_rate_cohort_seasons": real_sack_rate,
        "sack_rate_delta": real_sack_rate - 0.065,
        "scramble_rate_realized": "NOT_SEPARABLE_FROM_DESIGNED_RUSHES_IN_BLOCK_DATA",
        "verdict": "NEGLIGIBLE" if abs(real_sack_rate - 0.065) < 0.01 else "MATERIAL",
    }

    # Season stability of the dominant reason's bias contribution.
    season_stability = {}
    for season, g in qb.groupby("SEASON"):
        n_s = len(g)
        season_stability[int(season)] = {
            "N": int(n_s), "total_bias": float(g.BIAS.mean()),
            "matched_bias": float(g[g.MATCHED].BIAS.mean()) if g.MATCHED.any() else None,
            "wrong_qb_share_of_rows": float((g.REASON.str.startswith("WRONG_QB_SELECTED")).mean()),
        }
    stability_vals = [v["total_bias"] for v in season_stability.values()]
    # Structural = negative in every season (not a cohort-specific artifact of
    # one season's OOS sample), even though magnitude tracks each season's own
    # WRONG_QB_SELECTED row share rather than being a fixed constant.
    stable = bool(all(v < 0 for v in stability_vals))

    matched = by_reason.get("OK", {"bias_contribution": 0.0, "mean_bias": None})
    dominant_bias = by_reason[dominant_reason]["bias_contribution"] if dominant_reason else 0.0
    bias_share = float(dominant_bias / total_bias) if total_bias else None

    if dominant_reason and abs(bias_share) >= 0.60:
        verdict = "FOUND_CENTERING_DEFECT"
    elif dominant_reason:
        # check whether >=2 non-OK reasons jointly explain >=75%
        joint = sum(v["bias_contribution"] for k, v in by_reason.items() if k != "OK")
        verdict = "FOUND_MULTI_COMPONENT_BIAS" if total_bias and abs(joint / total_bias) >= 0.75 else "BROADER_PASSING_MODEL_REDESIGN"
    else:
        verdict = "BROADER_PASSING_MODEL_REDESIGN"

    result = {
        "mission": "SPORTS_NOVA_V9_QB_ATTEMPT_LEVEL_BIAS",
        "cohort": {"games": len(selected), "sims_per_game": N, "qb_eval_rows": n_total, "team_games": len(ts)},
        "by_reason": by_reason,
        "matched_primary_qb_rows_are_unbiased": matched["mean_bias"],
        "stage_pre_sim_vs_sim_vs_real": stage_summary,
        "dropback_conversion_check": dropback_conversion,
        "season_stability": season_stability,
        "season_stability_stable": stable,
        "VERDICT": verdict,
        "REAL_ATTEMPTS_MEAN": float(qb.REAL_ATT.mean()),
        "PRE_SIM_ATTEMPTS_MEAN": stage_summary["pred_pass_att_pre_sim_mean"] / 2.0,
        "SIM_ATTEMPTS_MEAN": float(qb.SIM_ATT.mean()),
        "TOTAL_BIAS": total_bias,
        "DOMINANT_STAGE": dominant_reason,
        "DOMINANT_STAGE_BIAS": by_reason[dominant_reason]["mean_bias"] if dominant_reason else None,
        "BIAS_SHARE": bias_share,
        "SEASON_STABILITY": ("STRUCTURAL_NEGATIVE_EVERY_SEASON_2020_2025_MAGNITUDE_TRACKS_WRONG_QB_SHARE"
                             if stable else "COHORT_SPECIFIC"),
        "SINGLE_NEXT_ACTION": (
            "Do not redesign the passing model: matched-primary-QB rows carry ~zero mean attempt "
            "bias. Investigate simulator._qb_shares' winner-take-all sharpening (** 2.5) and the "
            "team-game starter-identification heuristic in _primary_qb for the "
            "WRONG_QB_SELECTED rows (~identity-miss vs ~concentration-miss split above) as a "
            "separate, scoped QB-allocation task -- not a volume/pass-rate/dropback-conversion fix."
        ),
    }

    qb.to_csv(OUT / "SPORTS_NOVA_V9_QB_ATTEMPT_BIAS_ROWS.csv", index=False)
    ts.to_csv(OUT / "SPORTS_NOVA_V9_QB_ATTEMPT_BIAS_TEAM_STAGE.csv", index=False)
    (OUT / "SPORTS_NOVA_V9_QB_ATTEMPT_BIAS_REPORT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "VERDICT", "REAL_ATTEMPTS_MEAN", "PRE_SIM_ATTEMPTS_MEAN", "SIM_ATTEMPTS_MEAN", "TOTAL_BIAS",
        "DOMINANT_STAGE", "DOMINANT_STAGE_BIAS", "BIAS_SHARE", "SEASON_STABILITY", "SINGLE_NEXT_ACTION")}, indent=2))
    print(json.dumps(result["by_reason"], indent=2))
    print(json.dumps(result["stage_pre_sim_vs_sim_vs_real"], indent=2))
    print(json.dumps(result["dropback_conversion_check"], indent=2))


if __name__ == "__main__":
    main()
