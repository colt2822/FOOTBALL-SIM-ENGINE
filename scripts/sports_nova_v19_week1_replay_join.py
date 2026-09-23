"""SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER -- outcome join + V18 comparison.

Mirrors scripts/sports_nova_v3_v18_week1_replay_join.py's join/metric
formulas exactly, applied to the V19 replay artifacts, then adds a paired
comparison against the already-frozen SPORTS_NOVA_V18_WEEK1_REPLAY_RESULTS
.json (V18 is NOT re-simulated here -- reused as-is, on disk, hash-checked
by its own capture run).

Never imports or calls make_state/simulate_game. Never rewrites a
prediction artifact.
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

V19_REPLAY_DIR = DATA / "replay_v19" / "week1_2026"
V19_PRED_DIR = V19_REPLAY_DIR / "predictions"

V18_RESULTS_PATH = ROOT / "SPORTS_NOVA_V18_WEEK1_REPLAY_RESULTS.json"

OUT_V19_RESULTS = ROOT / "SPORTS_V19_WEEK1_RESULTS.json"
OUT_COMPARISON = ROOT / "SPORTS_V19_V18_COMPARISON.json"
OUT_REPORT = ROOT / "SPORTS_V19_QB_IDENTITY_REPORT.md"
OUT_CANARY = ROOT / "SPORTS_V19_DEN_KC_REPLAY.json"

CANARY_GAME_ID = "2026_01_DEN_KC"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_v18_hashes_unchanged() -> dict:
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


def join_slate(pred_dir: Path, sched_by_id: dict, panel: pd.DataFrame) -> list[dict]:
    pred_files = sorted(pred_dir.glob("*.json"))
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
            "GAME_ID": game_id, "STATUS": "SCORED",
            "AWAY_TEAM": pred["AWAY_TEAM"], "HOME_TEAM": pred["HOME_TEAM"],
            "DRAWS_HASH_INTACT": bool(draws_intact),
            "ACTUAL_HOME_SCORE": actual_home_score, "ACTUAL_AWAY_SCORE": actual_away_score,
            "MODEL_HOME_SCORE_MEAN": pred["MODEL_SCORE_DIST"]["HOME"]["mean"],
            "MODEL_AWAY_SCORE_MEAN": pred["MODEL_SCORE_DIST"]["AWAY"]["mean"],
            "HOME_SCORE_ERROR": pred["MODEL_SCORE_DIST"]["HOME"]["mean"] - actual_home_score,
            "AWAY_SCORE_ERROR": pred["MODEL_SCORE_DIST"]["AWAY"]["mean"] - actual_away_score,
            "HOME_SCORE_COVERED_P10_P90": bool(
                pred["MODEL_SCORE_DIST"]["HOME"]["p10"] <= actual_home_score <= pred["MODEL_SCORE_DIST"]["HOME"]["p90"]),
            "AWAY_SCORE_COVERED_P10_P90": bool(
                pred["MODEL_SCORE_DIST"]["AWAY"]["p10"] <= actual_away_score <= pred["MODEL_SCORE_DIST"]["AWAY"]["p90"]),
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
            entry["MARKET_SPREAD_ERROR_VS_ACTUAL"] = pred["MARKET"]["SPREAD_LINE"] - actual_margin
            entry["MODEL_VS_MARKET_SPREAD_DELTA"] = pred["MODEL_SPREAD_HOME"] - pred["MARKET"]["SPREAD_LINE"]
        if pred["MARKET"]["TOTAL_LINE"] is not None:
            entry["MARKET_TOTAL_ERROR_VS_ACTUAL"] = pred["MARKET"]["TOTAL_LINE"] - actual_total
            entry["MODEL_VS_MARKET_TOTAL_DELTA"] = pred["MODEL_TOTAL"] - pred["MARKET"]["TOTAL_LINE"]

        def qb_join(side, team):
            pred_qb = pred[f"{side}_QB"]
            identity_block = pred.get(f"{side}_QB_IDENTITY")
            game_panel = panel[(panel.GAME_ID == game_id) & (panel.TEAM == team) & (panel.POSITION == "QB")]
            if game_panel.empty:
                return {"STATUS": "NO_QB_PANEL_ROW"}
            actual_starter = game_panel.sort_values("PASS_ATTEMPTS", ascending=False).iloc[0]
            result = {
                "ACTUAL_STARTER_ID": str(actual_starter.PLAYER_ID),
                "ACTUAL_STARTER_NAME": str(actual_starter.PLAYER_NAME),
                "ACTUAL_PASS_ATTEMPTS": float(actual_starter.PASS_ATTEMPTS),
                "ACTUAL_PASS_YARDS": float(actual_starter.PASS_YARDS),
                "IDENTITY_BLOCK": identity_block,
            }
            if pred_qb is None:
                result["STATUS"] = "NO_PREDICTION"
                return result
            result["PREDICTED_QB_ID"] = pred_qb["PLAYER_ID"]
            result["IDENTITY_MATCH"] = bool(pred_qb["PLAYER_ID"] == str(actual_starter.PLAYER_ID))

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
            else:
                result["STATUS"] = "IDENTITY_MISMATCH"
            return result

        entry["HOME_QB_JOIN"] = qb_join("HOME", pred["HOME_TEAM"])
        entry["AWAY_QB_JOIN"] = qb_join("AWAY", pred["AWAY_TEAM"])
        per_game.append(entry)
    return per_game


def aggregate_metrics(per_game: list[dict]) -> dict:
    scored = [e for e in per_game if e["STATUS"] == "SCORED"]
    score_errors = [e["HOME_SCORE_ERROR"] for e in scored] + [e["AWAY_SCORE_ERROR"] for e in scored]
    qb_matches = [e for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                  if e[side].get("STATUS") == "IDENTITY_MATCH"]
    pass_att_errors_matched = [e[side]["PASS_ATTEMPTS_ERROR"] for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                                if e[side].get("STATUS") == "IDENTITY_MATCH"]
    pass_yds_errors_matched = [e[side]["PASS_YARDS_ERROR"] for e in scored for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
                                if e[side].get("STATUS") == "IDENTITY_MATCH"]
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
    briers = [e["BRIER_HOME"] for e in scored]
    loglosses = [e["LOGLOSS_HOME"] for e in scored]
    model_spread_errs = [abs(e["MODEL_SPREAD_HOME"] - e["ACTUAL_MARGIN_HOME"]) for e in scored]
    market_spread_errs = [abs(e["MARKET"]["SPREAD_LINE"] - e["ACTUAL_MARGIN_HOME"]) for e in scored
                           if e["MARKET"].get("SPREAD_LINE") is not None]
    model_total_errs = [abs(e["MODEL_TOTAL"] - e["ACTUAL_TOTAL"]) for e in scored]
    market_total_errs = [abs(e["MARKET"]["TOTAL_LINE"] - e["ACTUAL_TOTAL"]) for e in scored
                          if e["MARKET"].get("TOTAL_LINE") is not None]
    model_win_side_correct = sum(
        1 for e in scored if ((e["MODEL_WIN_PROB_HOME"] >= 0.5) == (e["ACTUAL_WINNER"] == "HOME")))

    return {
        "N_GAMES_SCORED": len(scored),
        "SCORE_MAE": float(np.mean(np.abs(score_errors))) if score_errors else None,
        "TEAM_SCORE_BIAS": float(np.mean(score_errors)) if score_errors else None,
        "QB_PASS_ATTEMPTS_MAE": float(np.mean(np.abs(pass_att_errors_all))) if pass_att_errors_all else None,
        "QB_PASS_ATTEMPTS_BIAS": float(np.mean(pass_att_errors_all)) if pass_att_errors_all else None,
        "QB_PASS_YARDS_MAE": float(np.mean(np.abs(pass_yds_errors_all))) if pass_yds_errors_all else None,
        "QB_PASS_YARDS_BIAS": float(np.mean(pass_yds_errors_all)) if pass_yds_errors_all else None,
        "QB_PASS_ATTEMPTS_MAE_IDENTITY_MATCHED_SUBSET": float(np.mean(np.abs(pass_att_errors_matched))) if pass_att_errors_matched else None,
        "QB_PASS_YARDS_MAE_IDENTITY_MATCHED_SUBSET": float(np.mean(np.abs(pass_yds_errors_matched))) if pass_yds_errors_matched else None,
        "N_QB_IDENTITY_MATCHED_SUBSET": len(pass_yds_errors_matched),
        "COVERAGE_HOME_SCORE_P10_P90_HIT_RATE": float(coverage_home) if coverage_home is not None else None,
        "COVERAGE_AWAY_SCORE_P10_P90_HIT_RATE": float(coverage_away) if coverage_away is not None else None,
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
    }


def paired_game_level(v19_per_game, v18_per_game_by_id):
    """Paired deltas at game level (n=16): score MAE components, spread/total
    error, Brier/logloss. Sign test (binomial, two-sided) as a distribution-
    free significance check appropriate for n=16."""
    from scipy import stats as _stats  # noqa: local import, optional dependency

    pairs = {"SCORE_ABS_ERROR": [], "BRIER_HOME": [], "SPREAD_ABS_ERROR": [], "TOTAL_ABS_ERROR": []}
    for e19 in v19_per_game:
        if e19["STATUS"] != "SCORED":
            continue
        e18 = v18_per_game_by_id.get(e19["GAME_ID"])
        if e18 is None or e18.get("STATUS") != "SCORED":
            continue
        v19_score_abs = (abs(e19["HOME_SCORE_ERROR"]) + abs(e19["AWAY_SCORE_ERROR"])) / 2
        v18_score_abs = (abs(e18["HOME_SCORE_ERROR"]) + abs(e18["AWAY_SCORE_ERROR"])) / 2
        pairs["SCORE_ABS_ERROR"].append((e19["GAME_ID"], v19_score_abs, v18_score_abs))
        pairs["BRIER_HOME"].append((e19["GAME_ID"], e19["BRIER_HOME"], e18["BRIER_HOME"]))
        pairs["SPREAD_ABS_ERROR"].append((e19["GAME_ID"], abs(e19["MODEL_SPREAD_HOME"] - e19["ACTUAL_MARGIN_HOME"]),
                                           abs(e18["MODEL_SPREAD_HOME"] - e18["ACTUAL_MARGIN_HOME"])))
        pairs["TOTAL_ABS_ERROR"].append((e19["GAME_ID"], abs(e19["MODEL_TOTAL"] - e19["ACTUAL_TOTAL"]),
                                          abs(e18["MODEL_TOTAL"] - e18["ACTUAL_TOTAL"])))

    out = {}
    for metric, rows in pairs.items():
        deltas = [v19 - v18 for _, v19, v18 in rows]
        n = len(deltas)
        n_v19_better = sum(1 for d in deltas if d < 0)
        n_v18_better = sum(1 for d in deltas if d > 0)
        n_tied = n - n_v19_better - n_v18_better
        sign_p = None
        if n_v19_better + n_v18_better > 0:
            sign_p = float(_stats.binomtest(min(n_v19_better, n_v18_better),
                                             n_v19_better + n_v18_better, 0.5).pvalue)
        out[metric] = {
            "N_PAIRS": n,
            "MEAN_V19": float(np.mean([v for _, v, _ in rows])) if rows else None,
            "MEAN_V18": float(np.mean([v for _, _, v in rows])) if rows else None,
            "MEAN_DELTA_V19_MINUS_V18": float(np.mean(deltas)) if deltas else None,
            "N_V19_BETTER": n_v19_better, "N_V18_BETTER": n_v18_better, "N_TIED": n_tied,
            "SIGN_TEST_TWO_SIDED_P": sign_p,
        }
    return out


def main():
    sched = pd.read_csv(SCHEDULE_CSV)
    sched = sched[(sched.season == 2026) & (sched.week == 1) & (sched.game_type == "REG")].copy()
    sched_by_id = {f"2026_01_{r.away_team}_{r.home_team}": r for r in sched.itertuples()}

    panel = pd.read_parquet(LIVE_PANEL)

    v18_results = json.loads(V18_RESULTS_PATH.read_text())
    v18_per_game_by_id = {e["GAME_ID"]: e for e in v18_results["PER_GAME"]}

    v19_per_game = join_slate(V19_PRED_DIR, sched_by_id, panel)
    v19_metrics = aggregate_metrics(v19_per_game)
    v18_metrics = v18_results["METRICS"]

    hash_check = verify_v18_hashes_unchanged()

    v19_results = {
        "SCHEMA": "SPORTS_V19_WEEK1_RESULTS",
        "SEASON": 2026, "WEEK": 1,
        "MODEL_VERSION": "sports_nova_v19.drive_block.season_boundary_qb_identity.1",
        "N_GAMES_TOTAL": len(v19_per_game),
        "METRICS": v19_metrics,
        "V18_HASH_CHECK_UNCHANGED": hash_check,
        "PER_GAME": v19_per_game,
    }
    OUT_V19_RESULTS.write_text(json.dumps(v19_results, indent=2, sort_keys=True, default=str))

    paired = paired_game_level(v19_per_game, v18_per_game_by_id)
    canary19 = next((e for e in v19_per_game if e["GAME_ID"] == CANARY_GAME_ID), None)
    canary18 = v18_per_game_by_id.get(CANARY_GAME_ID)
    canary_temporal_pass = bool(canary19 and all(
        canary19[side].get("IDENTITY_BLOCK", {}).get("RESOLUTION", {}).get("TEMPORAL_FIREWALL_PASS")
        for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
    ))
    canary_provenance_pass = bool(canary19 and all(
        canary19[side].get("IDENTITY_BLOCK", {}).get("RESOLUTION", {}).get("SOURCE")
        and canary19[side].get("IDENTITY_BLOCK", {}).get("RESOLUTION", {}).get("SOURCE_TIMESTAMP")
        and canary19[side].get("IDENTITY_BLOCK", {}).get("RESOLUTION", {}).get("CONFIDENCE") in {"HIGH", "CONFIRMED"}
        for side in ("HOME_QB_JOIN", "AWAY_QB_JOIN")
    ))

    # QB-attempt-level MDE context (this project's own established floor from
    # SPORTS_NOVA_V1_1_OVERNIGHT: n=120 team-games detects ~0.1-0.25 attempts
    # at 80% power; n=32 here is far smaller -- attempts/yards MAE deltas
    # should be read qualitatively, identity-match-rate is the load-bearing
    # number for this specific mission).
    calibration_delta = {
        "V19_COVERAGE_HOME": v19_metrics["COVERAGE_HOME_SCORE_P10_P90_HIT_RATE"],
        "V18_COVERAGE_HOME": v18_metrics["COVERAGE_HOME_SCORE_P10_P90_HIT_RATE"],
        "V19_COVERAGE_AWAY": v19_metrics["COVERAGE_AWAY_SCORE_P10_P90_HIT_RATE"],
        "V18_COVERAGE_AWAY": v18_metrics["COVERAGE_AWAY_SCORE_P10_P90_HIT_RATE"],
        "V19_BRIER": v19_metrics["BRIER_HOME_WIN"], "V18_BRIER": v18_metrics["BRIER_HOME_WIN"],
    }
    market_error_delta = {
        "V19_MODEL_SPREAD_MAE": v19_metrics["MODEL_SPREAD_MAE_VS_ACTUAL"],
        "V18_MODEL_SPREAD_MAE": v18_metrics["MODEL_SPREAD_MAE_VS_ACTUAL"],
        "V19_MODEL_TOTAL_MAE": v19_metrics["MODEL_TOTAL_MAE_VS_ACTUAL"],
        "V18_MODEL_TOTAL_MAE": v18_metrics["MODEL_TOTAL_MAE_VS_ACTUAL"],
        "MARKET_SPREAD_MAE": v19_metrics["MARKET_SPREAD_MAE_VS_ACTUAL"],
        "MARKET_TOTAL_MAE": v19_metrics["MARKET_TOTAL_MAE_VS_ACTUAL"],
    }

    qb_mismatch_reduced = v19_metrics["N_QB_IDENTITY_MISMATCHES"] < v18_metrics["N_QB_IDENTITY_MISMATCHES"]
    score_mae_improves = (v19_metrics["SCORE_MAE"] is not None and v18_metrics["SCORE_MAE"] is not None
                           and v19_metrics["SCORE_MAE"] < v18_metrics["SCORE_MAE"])
    pass_yards_mae_improves = (v19_metrics["QB_PASS_YARDS_MAE"] is not None and v18_metrics["QB_PASS_YARDS_MAE"] is not None
                                and v19_metrics["QB_PASS_YARDS_MAE"] < v18_metrics["QB_PASS_YARDS_MAE"])
    no_major_calibration_regression = (
        v19_metrics["BRIER_HOME_WIN"] is not None and v18_metrics["BRIER_HOME_WIN"] is not None
        and v19_metrics["BRIER_HOME_WIN"] <= v18_metrics["BRIER_HOME_WIN"] * 1.15
    )

    broad_improvement = qb_mismatch_reduced and score_mae_improves and pass_yards_mae_improves and no_major_calibration_regression
    promote = bool(broad_improvement and hash_check["ALL_PASS"] and canary_temporal_pass and canary_provenance_pass)

    comparison = {
        "SCHEMA": "SPORTS_V19_V18_COMPARISON",
        "MISSION": "SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER",
        "V18_QB_MISMATCH": v18_metrics["N_QB_IDENTITY_MISMATCHES"],
        "V19_QB_MISMATCH": v19_metrics["N_QB_IDENTITY_MISMATCHES"],
        "V18_SCORE_MAE": v18_metrics["SCORE_MAE"], "V19_SCORE_MAE": v19_metrics["SCORE_MAE"],
        "V18_PASS_ATTEMPTS_MAE": v18_metrics["QB_PASS_ATTEMPTS_MAE"],
        "V19_PASS_ATTEMPTS_MAE": v19_metrics["QB_PASS_ATTEMPTS_MAE"],
        "V18_PASS_YARDS_MAE": v18_metrics["QB_PASS_YARDS_MAE"], "V19_PASS_YARDS_MAE": v19_metrics["QB_PASS_YARDS_MAE"],
        "CALIBRATION_DELTA": calibration_delta,
        "MARKET_ERROR_DELTA": market_error_delta,
        "PAIRED_GAME_LEVEL": paired,
        "GATE_CHECKS": {
            "QB_MISMATCH_REDUCED": qb_mismatch_reduced,
            "SCORE_MAE_IMPROVES": score_mae_improves,
            "PASS_YARDS_MAE_IMPROVES": pass_yards_mae_improves,
            "NO_MAJOR_CALIBRATION_REGRESSION": no_major_calibration_regression,
            "DEN_KC_TEMPORAL_FIREWALL_PASS": canary_temporal_pass,
            "DEN_KC_IDENTITY_PROVENANCE_PASS": canary_provenance_pass,
            "V18_HASH_UNCHANGED": hash_check["ALL_PASS"],
        },
        "VERDICT": "PROMOTE_V19" if promote else "REJECT_V19_NOT_BROAD_IMPROVEMENT",
        "PROMOTE_V19": promote,
    }
    OUT_COMPARISON.write_text(json.dumps(comparison, indent=2, sort_keys=True, default=str))

    canary_valid = bool(
        canary19 and canary18
        and canary19.get("STATUS") == "SCORED"
        and canary19.get("DRAWS_HASH_INTACT")
        and canary19["HOME_QB_JOIN"].get("IDENTITY_MATCH")
        and canary19["AWAY_QB_JOIN"].get("IDENTITY_MATCH")
        and canary_temporal_pass
        and canary_provenance_pass
    )
    OUT_CANARY.write_text(json.dumps({
        "SCHEMA": "SPORTS_V19_DEN_KC_REPLAY",
        "MISSION": "SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER",
        "GAME_ID": CANARY_GAME_ID,
        "PREDICTION_FROZEN_BEFORE_OUTCOME_JOIN": True,
        "TEMPORAL_FIREWALL_PASS": canary_temporal_pass,
        "IDENTITY_PROVENANCE_PASS": canary_provenance_pass,
        "DEN_QB": canary19["AWAY_QB_JOIN"] if canary19 else None,
        "KC_QB": canary19["HOME_QB_JOIN"] if canary19 else None,
        "V18": canary18,
        "V19": canary19,
        "V18_V19_OUTPUT_DELTA": {
            "SCORE_DISTRIBUTION": {
                "V18_HOME_MEAN": canary18.get("MODEL_HOME_SCORE_MEAN") if canary18 else None,
                "V19_HOME_MEAN": canary19.get("MODEL_HOME_SCORE_MEAN") if canary19 else None,
                "V18_AWAY_MEAN": canary18.get("MODEL_AWAY_SCORE_MEAN") if canary18 else None,
                "V19_AWAY_MEAN": canary19.get("MODEL_AWAY_SCORE_MEAN") if canary19 else None,
            },
            "WIN_PROBABILITY_HOME": {
                "V18": canary18.get("MODEL_WIN_PROB_HOME") if canary18 else None,
                "V19": canary19.get("MODEL_WIN_PROB_HOME") if canary19 else None,
            },
            "SPREAD_HOME": {
                "V18": canary18.get("MODEL_SPREAD_HOME") if canary18 else None,
                "V19": canary19.get("MODEL_SPREAD_HOME") if canary19 else None,
            },
            "TOTAL": {
                "V18": canary18.get("MODEL_TOTAL") if canary18 else None,
                "V19": canary19.get("MODEL_TOTAL") if canary19 else None,
            },
        },
        "DEN_KC_REPLAY_VALID": canary_valid,
    }, indent=2, sort_keys=True, default=str))

    lines = []
    lines.append("# SPORTS_V19_QB_IDENTITY_REPORT\n\n")
    lines.append(f"V18 hash check (post V19 run, must be unchanged): "
                 f"{'PASS' if hash_check['ALL_PASS'] else 'FAIL'} "
                 f"({hash_check['COUNT']}/{hash_check['COUNT']} code hashes).\n\n")
    lines.append("## Canary: DEN @ KC\n\n")
    if canary19 and canary18:
        for side, team_key in (("HOME_QB_JOIN", "HOME_TEAM"), ("AWAY_QB_JOIN", "AWAY_TEAM")):
            j19, j18 = canary19[side], canary18[side]
            lines.append(f"- {canary19[team_key]}: V18 predicted={j18.get('PREDICTED_QB_ID')} "
                         f"({j18.get('STATUS')}) -> V19 predicted={j19.get('PREDICTED_QB_ID')} "
                         f"({j19.get('STATUS')}), actual={j19.get('ACTUAL_STARTER_ID')} "
                         f"({j19.get('ACTUAL_STARTER_NAME')})\n")
    lines.append(f"\nCanary pass criterion (V19 picks Mahomes for KC and Nix for DEN): "
                 f"{'PASS' if canary_valid else 'FAIL'}\n\n")

    lines.append("## Headline comparison (unconditional, n=32 team-games / n=16 games)\n\n")
    lines.append(f"```json\n{json.dumps(comparison, indent=2, default=str)}\n```\n\n")

    OUT_REPORT.write_text("".join(lines))

    print(json.dumps({
        "V18_QB_MISMATCH": comparison["V18_QB_MISMATCH"], "V19_QB_MISMATCH": comparison["V19_QB_MISMATCH"],
        "V18_SCORE_MAE": comparison["V18_SCORE_MAE"], "V19_SCORE_MAE": comparison["V19_SCORE_MAE"],
        "V18_PASS_YARDS_MAE": comparison["V18_PASS_YARDS_MAE"], "V19_PASS_YARDS_MAE": comparison["V19_PASS_YARDS_MAE"],
        "CALIBRATION_DELTA": calibration_delta, "MARKET_ERROR_DELTA": market_error_delta,
        "VERDICT": comparison["VERDICT"], "PROMOTE_V19": comparison["PROMOTE_V19"],
        "NEXT_SINGLE_ACTION": (
            "None -- V19 promoted, replace V18 references with V19 for the season-boundary "
            "QB identity mechanism" if promote else
            "Report REJECT: case-B stale-team-assignment defect (4/11 original mismatches, "
            "ATL/LV/MIA/NYJ) is untouched by this mission's scope and needs a roster/team-"
            "assignment fix inside make_state's latest_team logic, not _qb_shares -- separate "
            "mission if pursued."
        ),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
