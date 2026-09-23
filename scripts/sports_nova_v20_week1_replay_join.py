"""SPORTS_NOVA_V20_ROSTER_ASSIGNMENT_FIX -- outcome join + V19 comparison.

Mirrors scripts/sports_nova_v19_week1_replay_join.py's join/metric formulas
exactly, applied to V20's replay artifacts, with V19 (not V18) as the
baseline for the paired comparison -- V20 is a strict extension of V19
(same simulator, same resolutions, only roster construction changed), so
V19 is the correct "what did this specific change do" baseline. V18's code
hashes are still verified unchanged (V18 stays the frozen root reference).

PRE-COMMITTED BEFORE RUNNING (per SPORTS_NOVA_V1_1_EXPERIMENT_LEDGER.jsonl,
written before this script executed): team-score/spread/total/Brier deltas
are NOT evidence for or against V20. Injecting a player changes the player
axis (`player_ids = sorted(...)`) and allocator denominators for every
sim path in `_run_one`, which reshuffles `rng.dirichlet(...)`'s draw shape
and therefore every subsequent draw -- the same RNG-coupling V19 already
documented and confirmed via paired sign tests (p=0.69-1.00, pure noise) at
this same N_SIMS=2000/n=16. V20 changes MORE state per affected game than
V19 did (roster membership, not just which existing candidate wins), so
this coupling is *stronger* here, not weaker. QB identity match count is
the only metric this mission's promotion decision may weigh.
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

V20_REPLAY_DIR = DATA / "replay_v20" / "week1_2026"
V20_PRED_DIR = V20_REPLAY_DIR / "predictions"
V20_FIDELITY_PATH = V20_REPLAY_DIR / "SPORTS_NOVA_V20_FIDELITY_CHECK.json"

V19_RESULTS_PATH = ROOT / "SPORTS_V19_WEEK1_RESULTS.json"

OUT_V20_RESULTS = ROOT / "SPORTS_V20_WEEK1_RESULTS.json"
OUT_COMPARISON = ROOT / "SPORTS_V20_V19_COMPARISON.json"
OUT_REPORT = ROOT / "SPORTS_V20_ROSTER_ASSIGNMENT_REPORT.md"


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


def paired_game_level(v20_per_game, v19_per_game_by_id):
    from scipy import stats as _stats

    pairs = {"SCORE_ABS_ERROR": [], "BRIER_HOME": [], "SPREAD_ABS_ERROR": [], "TOTAL_ABS_ERROR": []}
    for e20 in v20_per_game:
        if e20["STATUS"] != "SCORED":
            continue
        e19 = v19_per_game_by_id.get(e20["GAME_ID"])
        if e19 is None or e19.get("STATUS") != "SCORED":
            continue
        v20_score_abs = (abs(e20["HOME_SCORE_ERROR"]) + abs(e20["AWAY_SCORE_ERROR"])) / 2
        v19_score_abs = (abs(e19["HOME_SCORE_ERROR"]) + abs(e19["AWAY_SCORE_ERROR"])) / 2
        pairs["SCORE_ABS_ERROR"].append((e20["GAME_ID"], v20_score_abs, v19_score_abs))
        pairs["BRIER_HOME"].append((e20["GAME_ID"], e20["BRIER_HOME"], e19["BRIER_HOME"]))
        pairs["SPREAD_ABS_ERROR"].append((e20["GAME_ID"], abs(e20["MODEL_SPREAD_HOME"] - e20["ACTUAL_MARGIN_HOME"]),
                                           abs(e19["MODEL_SPREAD_HOME"] - e19["ACTUAL_MARGIN_HOME"])))
        pairs["TOTAL_ABS_ERROR"].append((e20["GAME_ID"], abs(e20["MODEL_TOTAL"] - e20["ACTUAL_TOTAL"]),
                                          abs(e19["MODEL_TOTAL"] - e19["ACTUAL_TOTAL"])))

    out = {}
    for metric, rows in pairs.items():
        deltas = [v20 - v19 for _, v20, v19 in rows]
        n = len(deltas)
        n_v20_better = sum(1 for d in deltas if d < 0)
        n_v19_better = sum(1 for d in deltas if d > 0)
        n_tied = n - n_v20_better - n_v19_better
        sign_p = None
        if n_v20_better + n_v19_better > 0:
            sign_p = float(_stats.binomtest(min(n_v20_better, n_v19_better),
                                             n_v20_better + n_v19_better, 0.5).pvalue)
        out[metric] = {
            "N_PAIRS": n,
            "MEAN_V20": float(np.mean([v for _, v, _ in rows])) if rows else None,
            "MEAN_V19": float(np.mean([v for _, _, v in rows])) if rows else None,
            "MEAN_DELTA_V20_MINUS_V19": float(np.mean(deltas)) if deltas else None,
            "N_V20_BETTER": n_v20_better, "N_V19_BETTER": n_v19_better, "N_TIED": n_tied,
            "SIGN_TEST_TWO_SIDED_P": sign_p,
        }
    return out


def main():
    sched = pd.read_csv(SCHEDULE_CSV)
    sched = sched[(sched.season == 2026) & (sched.week == 1) & (sched.game_type == "REG")].copy()
    sched_by_id = {f"2026_01_{r.away_team}_{r.home_team}": r for r in sched.itertuples()}

    panel = pd.read_parquet(LIVE_PANEL)

    v19_results = json.loads(V19_RESULTS_PATH.read_text())
    v19_per_game_by_id = {e["GAME_ID"]: e for e in v19_results["PER_GAME"]}
    v19_metrics = v19_results["METRICS"]

    v20_per_game = join_slate(V20_PRED_DIR, sched_by_id, panel)
    v20_metrics = aggregate_metrics(v20_per_game)

    hash_check = verify_v18_hashes_unchanged()

    v20_results = {
        "SCHEMA": "SPORTS_V20_WEEK1_RESULTS",
        "SEASON": 2026, "WEEK": 1,
        "MODEL_VERSION": "sports_nova_v19.drive_block.season_boundary_qb_identity.1.v20_roster_injection",
        "N_GAMES_TOTAL": len(v20_per_game),
        "METRICS": v20_metrics,
        "V18_HASH_CHECK_UNCHANGED": hash_check,
        "PER_GAME": v20_per_game,
    }
    OUT_V20_RESULTS.write_text(json.dumps(v20_results, indent=2, sort_keys=True, default=str))

    paired = paired_game_level(v20_per_game, v19_per_game_by_id)

    qb_mismatch_reduced = v20_metrics["N_QB_IDENTITY_MISMATCHES"] < v19_metrics["N_QB_IDENTITY_MISMATCHES"]
    qb_mismatch_not_worse = v20_metrics["N_QB_IDENTITY_MISMATCHES"] <= v19_metrics["N_QB_IDENTITY_MISMATCHES"]
    no_major_calibration_regression = (
        v20_metrics["BRIER_HOME_WIN"] is not None and v19_metrics["BRIER_HOME_WIN"] is not None
        and v20_metrics["BRIER_HOME_WIN"] <= v19_metrics["BRIER_HOME_WIN"] * 1.15
    )
    # STRICT EXTENSION: V20's remaining mismatch set must be a SUBSET of
    # V19's, not merely a smaller count -- a run that fixed 3 slots while
    # breaking 3 different ones would pass a bare count check and would not
    # be a strict improvement.
    v20_mismatch_set = {tuple(m) for m in v20_metrics["QB_IDENTITY_MISMATCHES_DETAIL"]}
    v19_mismatch_set = {tuple(m) for m in v19_metrics["QB_IDENTITY_MISMATCHES_DETAIL"]}
    mismatch_set_is_strict_subset = v20_mismatch_set <= v19_mismatch_set
    fidelity_all_pass = None
    if V20_FIDELITY_PATH.exists():
        fidelity_all_pass = bool(json.loads(V20_FIDELITY_PATH.read_text()).get("FIDELITY_ALL_PASS"))
    # Score/spread/total/Brier are PRE-DECLARED noise for this mission (see
    # module docstring, written into the ledger before this script ran) --
    # they enter only as a "no major regression" guard, never as a
    # promotion-justifying improvement.
    promote = bool(qb_mismatch_reduced and mismatch_set_is_strict_subset and no_major_calibration_regression
                   and hash_check["ALL_PASS"] and fidelity_all_pass)

    comparison = {
        "SCHEMA": "SPORTS_V20_V19_COMPARISON",
        "MISSION": "SPORTS_NOVA_V20_ROSTER_ASSIGNMENT_FIX",
        "V19_QB_MISMATCH": v19_metrics["N_QB_IDENTITY_MISMATCHES"],
        "V20_QB_MISMATCH": v20_metrics["N_QB_IDENTITY_MISMATCHES"],
        "V19_QB_MISMATCH_DETAIL": v19_metrics["QB_IDENTITY_MISMATCHES_DETAIL"],
        "V20_QB_MISMATCH_DETAIL": v20_metrics["QB_IDENTITY_MISMATCHES_DETAIL"],
        "V19_SCORE_MAE": v19_metrics["SCORE_MAE"], "V20_SCORE_MAE": v20_metrics["SCORE_MAE"],
        "V19_PASS_YARDS_MAE": v19_metrics["QB_PASS_YARDS_MAE"], "V20_PASS_YARDS_MAE": v20_metrics["QB_PASS_YARDS_MAE"],
        "V19_BRIER": v19_metrics["BRIER_HOME_WIN"], "V20_BRIER": v20_metrics["BRIER_HOME_WIN"],
        "PAIRED_GAME_LEVEL": paired,
        "SCORE_SPREAD_TOTAL_BRIER_ARE_PRE_DECLARED_NOISE": True,
        "GATE_CHECKS": {
            "QB_MISMATCH_REDUCED": qb_mismatch_reduced,
            "QB_MISMATCH_NOT_WORSE": qb_mismatch_not_worse,
            "QB_MISMATCH_SET_IS_STRICT_SUBSET_OF_V19": mismatch_set_is_strict_subset,
            "NO_MAJOR_CALIBRATION_REGRESSION": no_major_calibration_regression,
            "V18_HASH_UNCHANGED": hash_check["ALL_PASS"],
            "FIDELITY_ALL_PASS": fidelity_all_pass,
        },
        "VERDICT": "PROMOTE_V20" if promote else "REJECT_V20",
        "PROMOTE_V20": promote,
    }
    OUT_COMPARISON.write_text(json.dumps(comparison, indent=2, sort_keys=True, default=str))

    lines = ["# SPORTS_V20_ROSTER_ASSIGNMENT_REPORT\n\n",
             f"V18 hash check (post V20 run, must be unchanged): "
             f"{'PASS' if hash_check['ALL_PASS'] else 'FAIL'} ({hash_check['COUNT']}/{hash_check['COUNT']}).\n\n",
             "## Per-team-slot mismatch detail (V19 -> V20)\n\n"]
    for e in v20_per_game:
        if e["STATUS"] != "SCORED":
            continue
        for side, team_key in (("HOME_QB_JOIN", "HOME_TEAM"), ("AWAY_QB_JOIN", "AWAY_TEAM")):
            j20 = e[side]
            e19 = v19_per_game_by_id.get(e["GAME_ID"], {})
            j19 = e19.get(side, {}) if e19 else {}
            if j20.get("STATUS") == "IDENTITY_MISMATCH" or j19.get("STATUS") == "IDENTITY_MISMATCH":
                lines.append(f"- {e[team_key]} ({e['GAME_ID']}): V19={j19.get('PREDICTED_QB_ID')} "
                             f"({j19.get('STATUS')}) -> V20={j20.get('PREDICTED_QB_ID')} ({j20.get('STATUS')}), "
                             f"actual={j20.get('ACTUAL_STARTER_ID')} ({j20.get('ACTUAL_STARTER_NAME')})\n")
    lines.append(f"\n## Headline comparison (n=32 team-games / n=16 games)\n\n```json\n"
                 f"{json.dumps(comparison, indent=2, default=str)}\n```\n\n")
    OUT_REPORT.write_text("".join(lines))

    print(json.dumps({
        "V19_QB_MISMATCH": comparison["V19_QB_MISMATCH"], "V20_QB_MISMATCH": comparison["V20_QB_MISMATCH"],
        "V19_SCORE_MAE": comparison["V19_SCORE_MAE"], "V20_SCORE_MAE": comparison["V20_SCORE_MAE"],
        "VERDICT": comparison["VERDICT"], "PROMOTE_V20": comparison["PROMOTE_V20"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
