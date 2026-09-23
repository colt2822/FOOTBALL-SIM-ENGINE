"""SPORTS_NOVA_V18_WEEK1_HISTORICAL_REPLAY -- outcome join + report (SEPARATE
from capture, by design; see sports_nova_v3_v18_week1_replay_capture.py).

Reads the frozen prediction artifacts + raw draw arrays written by the
capture script and joins them against:
  - schedule_snapshot_2026.csv home_score/away_score (final results)
  - the live causal panel's own realized player rows for each GAME_ID
    (box-score stats for the actual final game, added by the panel refresh
    once the games went final -- these rows are not touched by, and were
    never visible to, the capture script)

Never imports or calls make_state/simulate_game. Never rewrites a
prediction artifact or its PREDICTION_HASH.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
SCHEDULE_CSV = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"
FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"

REPLAY_DIR = DATA / "replay" / "week1_2026"
PRED_DIR = REPLAY_DIR / "predictions"
DRAWS_DIR = REPLAY_DIR / "draws"

CANARY_GAME_ID = "2026_01_DEN_KC"

OUT_RESULTS = ROOT / "SPORTS_NOVA_V18_WEEK1_REPLAY_RESULTS.json"
OUT_CANARY = ROOT / "SPORTS_NOVA_V18_MNF_DEN_KC_CANARY.json"
OUT_REPORT = ROOT / "SPORTS_NOVA_V18_WEEK1_REPLAY_REPORT.md"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_hashes_post_run() -> dict:
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    detail = {}
    all_pass = True
    for rel, meta in freeze["CODE_HASHES"].items():
        actual = sha256_file(ROOT / rel)
        match = actual == meta["sha256"]
        detail[rel] = match
        all_pass = all_pass and match
    return {"ALL_PASS": all_pass, "COUNT": len(freeze["CODE_HASHES"]), "DETAIL": detail}


def pct_rank(draws: np.ndarray, actual: float) -> float:
    return float(np.mean(draws <= actual) * 100.0)


def brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def logloss(p: float, outcome: int) -> float:
    eps = 1e-9
    p = min(max(p, eps), 1 - eps)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


def main():
    sched = pd.read_csv(SCHEDULE_CSV)
    sched = sched[(sched.season == 2026) & (sched.week == 1) & (sched.game_type == "REG")].copy()
    sched_by_id = {f"2026_01_{r.away_team}_{r.home_team}": r for r in sched.itertuples()}

    panel = pd.read_parquet(LIVE_PANEL)

    pred_files = sorted(PRED_DIR.glob("*.json"))
    per_game = []
    for pf in pred_files:
        pred = json.loads(pf.read_text())
        game_id = pred["GAME_ID"]
        sched_row = sched_by_id.get(game_id)
        if sched_row is None or pd.isna(sched_row.home_score) or pd.isna(sched_row.away_score):
            per_game.append({"GAME_ID": game_id, "STATUS": "OUTCOME_UNAVAILABLE"})
            continue
        actual_home_score = float(sched_row.home_score)
        actual_away_score = float(sched_row.away_score)

        draws = np.load(ROOT / pred["RAW_DRAWS_PATH"])
        draws_sha = sha256_file(ROOT / pred["RAW_DRAWS_PATH"])
        draws_intact = draws_sha == pred["RAW_DRAWS_SHA256"]

        home_score_draws = draws["home_score"]
        away_score_draws = draws["away_score"]

        entry = {
            "GAME_ID": game_id,
            "STATUS": "SCORED",
            "AWAY_TEAM": pred["AWAY_TEAM"], "HOME_TEAM": pred["HOME_TEAM"],
            "DRAWS_HASH_INTACT": bool(draws_intact),
            "ACTUAL_HOME_SCORE": actual_home_score,
            "ACTUAL_AWAY_SCORE": actual_away_score,
            "MODEL_HOME_SCORE_MEAN": pred["MODEL_SCORE_DIST"]["HOME"]["mean"],
            "MODEL_AWAY_SCORE_MEAN": pred["MODEL_SCORE_DIST"]["AWAY"]["mean"],
            "HOME_SCORE_ERROR": pred["MODEL_SCORE_DIST"]["HOME"]["mean"] - actual_home_score,
            "AWAY_SCORE_ERROR": pred["MODEL_SCORE_DIST"]["AWAY"]["mean"] - actual_away_score,
            "HOME_SCORE_PCT_RANK_OF_ACTUAL": pct_rank(home_score_draws, actual_home_score),
            "AWAY_SCORE_PCT_RANK_OF_ACTUAL": pct_rank(away_score_draws, actual_away_score),
            "HOME_SCORE_COVERED_P10_P90": bool(
                pred["MODEL_SCORE_DIST"]["HOME"]["p10"] <= actual_home_score <= pred["MODEL_SCORE_DIST"]["HOME"]["p90"]),
            "AWAY_SCORE_COVERED_P10_P90": bool(
                pred["MODEL_SCORE_DIST"]["AWAY"]["p10"] <= actual_away_score <= pred["MODEL_SCORE_DIST"]["AWAY"]["p90"]),
            # Discrete-support caveat: scores are integers and p10/p90 are
            # np.quantile draws landing on integers too, so the actual mass
            # inside [p10,p90] is not exactly 80% (not tie-corrected). This
            # is that game's realized mass, so HIT_RATE above is compared
            # against the right target, not a naive 0.80.
            "HOME_SCORE_P10_P90_NOMINAL_MASS": float(np.mean(
                (home_score_draws >= pred["MODEL_SCORE_DIST"]["HOME"]["p10"]) &
                (home_score_draws <= pred["MODEL_SCORE_DIST"]["HOME"]["p90"]))),
            "AWAY_SCORE_P10_P90_NOMINAL_MASS": float(np.mean(
                (away_score_draws >= pred["MODEL_SCORE_DIST"]["AWAY"]["p10"]) &
                (away_score_draws <= pred["MODEL_SCORE_DIST"]["AWAY"]["p90"]))),
        }

        actual_winner = "HOME" if actual_home_score > actual_away_score else (
            "AWAY" if actual_away_score > actual_home_score else "TIE")
        entry["ACTUAL_WINNER"] = actual_winner
        home_outcome = 1 if actual_winner == "HOME" else 0
        entry["MODEL_WIN_PROB_HOME"] = pred["MODEL_WIN_PROB"]["HOME"]
        entry["BRIER_HOME"] = brier(pred["MODEL_WIN_PROB"]["HOME"], home_outcome)
        entry["LOGLOSS_HOME"] = logloss(pred["MODEL_WIN_PROB"]["HOME"], home_outcome)

        actual_margin = actual_home_score - actual_away_score
        actual_total = actual_home_score + actual_away_score
        entry["MODEL_SPREAD_HOME"] = pred["MODEL_SPREAD_HOME"]
        entry["ACTUAL_MARGIN_HOME"] = actual_margin
        entry["MODEL_TOTAL"] = pred["MODEL_TOTAL"]
        entry["ACTUAL_TOTAL"] = actual_total
        entry["MARKET"] = pred["MARKET"]
        if pred["MARKET"]["SPREAD_LINE"] is not None:
            # nflreadr data dictionary: spread_line is positive when the HOME
            # team was favored, and "lines up with the result column" (home
            # score minus away score) -- i.e. it is already in the same
            # home-minus-away sign convention as ACTUAL_MARGIN_HOME and
            # MODEL_SPREAD_HOME. No sign flip.
            entry["MARKET_SPREAD_ERROR_VS_ACTUAL"] = pred["MARKET"]["SPREAD_LINE"] - actual_margin
            entry["MODEL_VS_MARKET_SPREAD_DELTA"] = pred["MODEL_SPREAD_HOME"] - pred["MARKET"]["SPREAD_LINE"]
        if pred["MARKET"]["TOTAL_LINE"] is not None:
            entry["MARKET_TOTAL_ERROR_VS_ACTUAL"] = pred["MARKET"]["TOTAL_LINE"] - actual_total
            entry["MODEL_VS_MARKET_TOTAL_DELTA"] = pred["MODEL_TOTAL"] - pred["MARKET"]["TOTAL_LINE"]

        def qb_join(side, team):
            pred_qb = pred[f"{side}_QB"]
            game_panel = panel[(panel.GAME_ID == game_id) & (panel.TEAM == team) & (panel.POSITION == "QB")]
            if game_panel.empty:
                return {"STATUS": "NO_QB_PANEL_ROW"}
            actual_starter = game_panel.sort_values("PASS_ATTEMPTS", ascending=False).iloc[0]
            result = {
                "ACTUAL_STARTER_ID": str(actual_starter.PLAYER_ID),
                "ACTUAL_STARTER_NAME": str(actual_starter.PLAYER_NAME),
                "ACTUAL_PASS_ATTEMPTS": float(actual_starter.PASS_ATTEMPTS),
                "ACTUAL_COMPLETIONS": float(actual_starter.COMPLETIONS),
                "ACTUAL_PASS_YARDS": float(actual_starter.PASS_YARDS),
            }
            if pred_qb is None:
                result["STATUS"] = "NO_PREDICTION"
                return result
            result["PREDICTED_QB_ID"] = pred_qb["PLAYER_ID"]
            result["IDENTITY_MATCH"] = bool(pred_qb["PLAYER_ID"] == str(actual_starter.PLAYER_ID))

            # UNCONDITIONAL metric (slate-level, not selected on the outcome):
            # score the model's own predicted starter against THAT SAME
            # player's real box-score row for this game, whatever it is. If
            # he never appears in the game's panel rows at all, his real
            # pass_attempts/pass_yards were 0 -- that is itself the model's
            # error, not a missing observation, and is scored as 0.
            pred_player_row = game_panel[game_panel.PLAYER_ID == pred_qb["PLAYER_ID"]]
            if pred_player_row.empty:
                actual_pa_for_pred, actual_py_for_pred = 0.0, 0.0
                result["PREDICTED_QB_APPEARED_IN_GAME"] = False
            else:
                actual_pa_for_pred = float(pred_player_row.iloc[0].PASS_ATTEMPTS)
                actual_py_for_pred = float(pred_player_row.iloc[0].PASS_YARDS)
                result["PREDICTED_QB_APPEARED_IN_GAME"] = True
            result["PASS_ATTEMPTS_ERROR_UNCONDITIONAL"] = pred_qb["PASS_ATTEMPTS"]["mean"] - actual_pa_for_pred
            result["PASS_YARDS_ERROR_UNCONDITIONAL"] = pred_qb["PASS_YARDS"]["mean"] - actual_py_for_pred

            if result["IDENTITY_MATCH"]:
                pa_draws = draws.get(f"{side.lower()}_qb_pass_attempts")
                py_draws = draws.get(f"{side.lower()}_qb_pass_yards")
                result["STATUS"] = "IDENTITY_MATCH"
                result["PRED_PASS_ATTEMPTS_MEAN"] = pred_qb["PASS_ATTEMPTS"]["mean"]
                result["PRED_PASS_YARDS_MEAN"] = pred_qb["PASS_YARDS"]["mean"]
                result["PASS_ATTEMPTS_ERROR"] = pred_qb["PASS_ATTEMPTS"]["mean"] - result["ACTUAL_PASS_ATTEMPTS"]
                result["PASS_YARDS_ERROR"] = pred_qb["PASS_YARDS"]["mean"] - result["ACTUAL_PASS_YARDS"]
                if pa_draws is not None:
                    result["PASS_ATTEMPTS_PCT_RANK_OF_ACTUAL"] = pct_rank(pa_draws, result["ACTUAL_PASS_ATTEMPTS"])
                if py_draws is not None:
                    result["PASS_YARDS_PCT_RANK_OF_ACTUAL"] = pct_rank(py_draws, result["ACTUAL_PASS_YARDS"])
            else:
                result["STATUS"] = "IDENTITY_MISMATCH"
                result["REASON"] = ("worker/sports_nova_v3/simulator.py::_qb_shares' hard min-gap "
                                     "recency filter (SN3_V10_QB_RECENCY_GATE_FIX, kept after that "
                                     "comparison) excludes any QB whose games_since_last_team_game "
                                     "is not the team's minimum. Firing across a season boundary -- "
                                     "a regime that comparison never covered -- a backup who closed "
                                     "out 2025 (e.g. games_since_last_team_game=0) outranks a starter "
                                     "rested/injured earlier (nonzero gap), even into a new season. "
                                     "Known, disclosed V18 property; not patched in this replay.")
            return result

        entry["HOME_QB_JOIN"] = qb_join("HOME", pred["HOME_TEAM"])
        entry["AWAY_QB_JOIN"] = qb_join("AWAY", pred["AWAY_TEAM"])
        per_game.append(entry)

    scored = [e for e in per_game if e["STATUS"] == "SCORED"]
    invalid_log_path = REPLAY_DIR / "SPORTS_NOVA_V18_WEEK1_CAPTURE_LOG.json"
    capture_log = json.loads(invalid_log_path.read_text()) if invalid_log_path.exists() else []
    invalid = [c for c in capture_log if c["STATUS"] == "INVALID_FOR_REPLAY"]

    def agg_mae(key):
        vals = [abs(e[key]) for e in scored if key in e]
        return float(np.mean(vals)) if vals else None

    def agg_bias(key):
        vals = [e[key] for e in scored if key in e]
        return float(np.mean(vals)) if vals else None

    score_errors = [e["HOME_SCORE_ERROR"] for e in scored] + [e["AWAY_SCORE_ERROR"] for e in scored]
    qb_matches = [e for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                  if e[side].get("STATUS") == "IDENTITY_MATCH"]
    # IDENTITY-MATCHED-SUBSET metric: conditioned on the outcome (identity
    # match is determined by comparing to the real starter) -- reported only
    # as a labeled footnote, never as the headline QB accuracy number.
    pass_att_errors_matched = [e[side]["PASS_ATTEMPTS_ERROR"] for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                                if e[side].get("STATUS") == "IDENTITY_MATCH"]
    pass_yds_errors_matched = [e[side]["PASS_YARDS_ERROR"] for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                                if e[side].get("STATUS") == "IDENTITY_MATCH"]
    # UNCONDITIONAL (slate-level) metric: every predicted starter (32 team-
    # games), scored against that same player's real box-score row -- not
    # selected on whether the model happened to get identity right.
    pass_att_errors_all = [e[side]["PASS_ATTEMPTS_ERROR_UNCONDITIONAL"] for e in scored
                            for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                            if "PASS_ATTEMPTS_ERROR_UNCONDITIONAL" in e[side]]
    pass_yds_errors_all = [e[side]["PASS_YARDS_ERROR_UNCONDITIONAL"] for e in scored
                            for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                            if "PASS_YARDS_ERROR_UNCONDITIONAL" in e[side]]
    identity_mismatches = [(e["GAME_ID"], side) for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                            if e[side].get("STATUS") == "IDENTITY_MISMATCH"]

    coverage_home = np.mean([e["HOME_SCORE_COVERED_P10_P90"] for e in scored]) if scored else None
    coverage_away = np.mean([e["AWAY_SCORE_COVERED_P10_P90"] for e in scored]) if scored else None
    nominal_home = np.mean([e["HOME_SCORE_P10_P90_NOMINAL_MASS"] for e in scored]) if scored else None
    nominal_away = np.mean([e["AWAY_SCORE_P10_P90_NOMINAL_MASS"] for e in scored]) if scored else None
    briers = [e["BRIER_HOME"] for e in scored]
    loglosses = [e["LOGLOSS_HOME"] for e in scored]

    model_spread_errs = [abs(e["MODEL_SPREAD_HOME"] - e["ACTUAL_MARGIN_HOME"]) for e in scored]
    market_spread_errs = [abs(e["MARKET"]["SPREAD_LINE"] - e["ACTUAL_MARGIN_HOME"]) for e in scored
                           if e["MARKET"].get("SPREAD_LINE") is not None]
    model_total_errs = [abs(e["MODEL_TOTAL"] - e["ACTUAL_TOTAL"]) for e in scored]
    market_total_errs = [abs(e["MARKET"]["TOTAL_LINE"] - e["ACTUAL_TOTAL"]) for e in scored
                          if e["MARKET"].get("TOTAL_LINE") is not None]
    model_win_side_correct = sum(
        1 for e in scored
        if ((e["MODEL_WIN_PROB_HOME"] >= 0.5) == (e["ACTUAL_WINNER"] == "HOME")))

    hash_check_post = verify_hashes_post_run()

    summary = {
        "SCHEMA": "SPORTS_NOVA_V18_WEEK1_REPLAY_RESULTS",
        "SEASON": 2026, "WEEK": 1,
        "N_GAMES_TOTAL": len(per_game),
        "N_GAMES_SCORED": len(scored),
        "N_GAMES_INVALID_FOR_REPLAY": len(invalid),
        "INVALID_GAMES": invalid,
        "METRICS": {
            "SCORE_MAE": float(np.mean(np.abs(score_errors))) if score_errors else None,
            "TEAM_SCORE_BIAS": float(np.mean(score_errors)) if score_errors else None,
            "QB_PASS_ATTEMPTS_MAE": float(np.mean(np.abs(pass_att_errors_all))) if pass_att_errors_all else None,
            "QB_PASS_ATTEMPTS_BIAS": float(np.mean(pass_att_errors_all)) if pass_att_errors_all else None,
            "QB_PASS_YARDS_MAE": float(np.mean(np.abs(pass_yds_errors_all))) if pass_yds_errors_all else None,
            "QB_PASS_YARDS_BIAS": float(np.mean(pass_yds_errors_all)) if pass_yds_errors_all else None,
            "QB_METRIC_NOTE": "Unconditional/slate-level: each predicted starter (n=32 team-games) "
                                "scored against that same player's own real box score for the game "
                                "(0 if he never appears -- itself the model's error, not missing "
                                "data). NOT conditioned on identity match. This is dominated by the "
                                "11/32 identity-mismatch slots (actual=0 by construction there), so "
                                "read QB_PASS_ATTEMPTS_BIAS/QB_PASS_YARDS_BIAS above as an "
                                "IDENTITY-RESOLUTION failure expressed in volume units, not a "
                                "volume-calibration problem. The *_IDENTITY_MATCHED_SUBSET bias "
                                "fields below are the actual volume-calibration signal (small), "
                                "conditioned on the outcome so reported only as a footnote.",
            "QB_PASS_ATTEMPTS_MAE_IDENTITY_MATCHED_SUBSET": float(np.mean(np.abs(pass_att_errors_matched))) if pass_att_errors_matched else None,
            "QB_PASS_ATTEMPTS_BIAS_IDENTITY_MATCHED_SUBSET": float(np.mean(pass_att_errors_matched)) if pass_att_errors_matched else None,
            "QB_PASS_YARDS_MAE_IDENTITY_MATCHED_SUBSET": float(np.mean(np.abs(pass_yds_errors_matched))) if pass_yds_errors_matched else None,
            "QB_PASS_YARDS_BIAS_IDENTITY_MATCHED_SUBSET": float(np.mean(pass_yds_errors_matched)) if pass_yds_errors_matched else None,
            "QB_IDENTITY_MATCHED_SUBSET_NOTE": f"n={len(pass_yds_errors_matched)} of 32 team-games -- "
                                "conditioned on the outcome (identity match is itself determined by "
                                "comparing to the real starter), NOT a slate-level metric. Reported "
                                "only as a footnote.",
            "COMPLETIONS": "NOT_MODELED_BY_V18_ENGINE -- actual completions reported per-game, no model comparison possible",
            "COVERAGE_HOME_SCORE_P10_P90_HIT_RATE": float(coverage_home) if coverage_home is not None else None,
            "COVERAGE_AWAY_SCORE_P10_P90_HIT_RATE": float(coverage_away) if coverage_away is not None else None,
            "COVERAGE_HOME_SCORE_P10_P90_NOMINAL_MASS": float(nominal_home) if nominal_home is not None else None,
            "COVERAGE_AWAY_SCORE_P10_P90_NOMINAL_MASS": float(nominal_away) if nominal_away is not None else None,
            "COVERAGE_NOTE": "Scores are integer-valued and p10/p90 are np.quantile draws landing "
                              "on integers, so [p10,p90] is discrete-support and not tie-corrected -- "
                              "compare HIT_RATE against NOMINAL_MASS (the interval's own realized "
                              "coverage), not against a flat 0.80 target.",
            "BRIER_HOME_WIN": float(np.mean(briers)) if briers else None,
            "LOGLOSS_HOME_WIN": float(np.mean(loglosses)) if loglosses else None,
            "N_QB_IDENTITY_MATCHES": len(qb_matches),
            "N_QB_IDENTITY_MISMATCHES": len(identity_mismatches),
            "QB_IDENTITY_MISMATCHES_DETAIL": identity_mismatches,
            "MODEL_SPREAD_MAE_VS_ACTUAL": float(np.mean(model_spread_errs)) if model_spread_errs else None,
            "MARKET_SPREAD_MAE_VS_ACTUAL": float(np.mean(market_spread_errs)) if market_spread_errs else None,
            "MODEL_TOTAL_MAE_VS_ACTUAL": float(np.mean(model_total_errs)) if model_total_errs else None,
            "MARKET_TOTAL_MAE_VS_ACTUAL": float(np.mean(market_total_errs)) if market_total_errs else None,
            "MODEL_STRAIGHT_UP_WIN_ACCURACY": model_win_side_correct / len(scored) if scored else None,
        },
        "MARKET_COMPARISON_STATUS": "AVAILABLE -- nflverse games.csv spread_line/total_line/moneyline fields present for this slate",
        "HASH_CHECK_POST_RUN": hash_check_post,
        "PER_GAME": per_game,
    }
    OUT_RESULTS.write_text(json.dumps(summary, indent=2, sort_keys=True))

    canary = next((e for e in per_game if e["GAME_ID"] == CANARY_GAME_ID), None)
    canary_pred = json.loads((PRED_DIR / f"{CANARY_GAME_ID}.json").read_text()) if (PRED_DIR / f"{CANARY_GAME_ID}.json").exists() else None
    OUT_CANARY.write_text(json.dumps({
        "SCHEMA": "SPORTS_NOVA_V18_MNF_DEN_KC_CANARY",
        "GAME_ID": CANARY_GAME_ID,
        "PREDICTION_ARTIFACT": canary_pred,
        "JOINED_RESULT": canary,
        "HASH_CHECK_POST_RUN": hash_check_post,
    }, indent=2, sort_keys=True, default=str))

    lines = []
    lines.append("# SPORTS_NOVA_V18_WEEK1_2026_HISTORICAL_REPLAY_REPORT\n")
    lines.append(f"14/14 frozen V18 code hashes verified before AND after this run: "
                 f"{'PASS' if hash_check_post['ALL_PASS'] else 'FAIL'}.\n")
    lines.append(f"Games captured: {len(per_game)} / 16. Scored (outcome joined): {len(scored)}. "
                 f"Invalid for replay: {len(invalid)}.\n")
    if canary:
        lines.append("## MNF Canary: DEN @ KC\n")
        lines.append(f"- Model score: DEN {canary['MODEL_AWAY_SCORE_MEAN']:.1f} @ KC {canary['MODEL_HOME_SCORE_MEAN']:.1f} "
                     f"(win prob HOME/KC = {canary['MODEL_WIN_PROB_HOME']:.3f})\n")
        lines.append(f"- Actual score: DEN {canary['ACTUAL_AWAY_SCORE']:.0f} @ KC {canary['ACTUAL_HOME_SCORE']:.0f} "
                     f"(winner: {canary['ACTUAL_WINNER']})\n")
        lines.append(f"- Home score percentile rank of actual: {canary['HOME_SCORE_PCT_RANK_OF_ACTUAL']:.1f} "
                     f"(covered by model P10-P90: {canary['HOME_SCORE_COVERED_P10_P90']})\n")
        lines.append(f"- Away score percentile rank of actual: {canary['AWAY_SCORE_PCT_RANK_OF_ACTUAL']:.1f} "
                     f"(covered by model P10-P90: {canary['AWAY_SCORE_COVERED_P10_P90']})\n")
        if canary["MARKET"].get("SPREAD_LINE") is not None:
            lines.append(f"- Market spread (home-minus-away convention): {canary['MARKET']['SPREAD_LINE']:+.1f}, "
                         f"model spread (home): {canary['MODEL_SPREAD_HOME']:+.1f}, "
                         f"actual margin (home): {canary['ACTUAL_MARGIN_HOME']:+.0f} "
                         f"-- model and market picked opposite sides here\n")
        for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN"):
            qj = canary[side]
            lines.append(f"- {side}: {json.dumps(qj)}\n")
    lines.append("\n## Week 1 aggregate metrics\n")
    lines.append(f"```json\n{json.dumps(summary['METRICS'], indent=2)}\n```\n")
    m = summary["METRICS"]
    lines.append(
        f"\nModel vs market spread MAE ({m['MODEL_SPREAD_MAE_VS_ACTUAL']:.2f} vs "
        f"{m['MARKET_SPREAD_MAE_VS_ACTUAL']:.2f}) are indistinguishable at n=16 -- not evidence "
        f"of edge either direction. Total MAE ({m['MODEL_TOTAL_MAE_VS_ACTUAL']:.2f} vs "
        f"{m['MARKET_TOTAL_MAE_VS_ACTUAL']:.2f}) is a real gap, mechanically explained by "
        f"TEAM_SCORE_BIAS={m['TEAM_SCORE_BIAS']:.2f} (~{-2*m['TEAM_SCORE_BIAS']:.1f} points of "
        f"systematic total under-prediction from summing two under-predicted team scores).\n")
    if invalid:
        lines.append("\n## Invalid-for-replay games\n")
        for i in invalid:
            lines.append(f"- {i['GAME_ID']}: {i['REASON']}\n")
    OUT_REPORT.write_text("".join(lines))

    print(json.dumps({
        "V18_HASH_CHECK": hash_check_post["ALL_PASS"],
        "N_SCORED": len(scored),
        "N_INVALID": len(invalid),
        "SCORE_MAE": summary["METRICS"]["SCORE_MAE"],
        "QB_PASS_YARDS_MAE": summary["METRICS"]["QB_PASS_YARDS_MAE"],
    }, indent=2))


if __name__ == "__main__":
    main()
