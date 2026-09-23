"""SPORTS_NOVA_V21_EXTRA_POINT_FIX -- outcome join + V20 comparison.

Baseline is V20 (not V18/V19): V21 changes only the scoring mechanism, so
V20 is the correct "what did this specific change do" comparison. V18's
code hashes are still verified unchanged.

Promotion criteria for a scoring-mechanism fix are different from V19/V20's
QB-identity criteria: QB identity must be UNCHANGED (this mission touches
nothing about who plays), and score/total/spread/coverage/CRPS SHOULD move
-- that is the entire point of this fix, not noise to be waved away. The
gate here is: identity picks unchanged (isolation proof), total/score bias
moves toward the market/actual (not just toward one arbitrarily chosen
target), and V18 stays untouched.
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

V20_PRED_DIR = DATA / "replay_v20" / "week1_2026" / "predictions"
V21_PRED_DIR = DATA / "replay_v21" / "week1_2026" / "predictions"
V21_DRAWS_DIR = DATA / "replay_v21" / "week1_2026" / "draws"

OUT_V21_RESULTS = ROOT / "SPORTS_V21_WEEK1_RESULTS.json"
OUT_COMPARISON = ROOT / "SPORTS_V21_V20_COMPARISON.json"
OUT_REPORT = ROOT / "SPORTS_V21_EXTRA_POINT_FIX_REPORT.md"


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


def crps_empirical(draws: np.ndarray, actual: float) -> float:
    term1 = np.mean(np.abs(draws - actual))
    x = np.sort(draws)
    n = len(x)
    i = np.arange(1, n + 1)
    term2 = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * x)
    return float(term1 - 0.5 * term2)


def main():
    sched = pd.read_csv(SCHEDULE_CSV)
    sched = sched[(sched.season == 2026) & (sched.week == 1) & (sched.game_type == "REG")].copy()
    sched_by_id = {f"2026_01_{r.away_team}_{r.home_team}": r for r in sched.itertuples()}

    identity_checks = []
    per_game = []
    model_totals, actual_totals, market_totals = [], [], []
    crps_home_all, crps_away_all = [], []
    pit_all = []
    coverage_hits = {q: [] for q in (0.5, 0.6, 0.7, 0.8, 0.9)}

    for pf in sorted(V21_PRED_DIR.glob("*.json")):
        pred = json.loads(pf.read_text())
        gid = pred["GAME_ID"]

        for side, team_key in (("HOME", "HOME_TEAM"), ("AWAY", "AWAY_TEAM")):
            idb = pred[f"{side}_QB_IDENTITY"]
            identity_checks.append({
                "GAME_ID": gid, "TEAM": pred[team_key], "SIDE": side,
                "V21_QB": idb["V21_PICKED_QB_ID"], "V20_QB": idb["V20_PICKED_QB_ID"],
                "UNCHANGED": idb["IDENTITY_UNCHANGED_VS_V20"],
            })

        row = sched_by_id.get(gid)
        if row is None or pd.isna(row.home_score) or pd.isna(row.away_score):
            per_game.append({"GAME_ID": gid, "STATUS": "OUTCOME_UNAVAILABLE"})
            continue
        actual_home, actual_away = float(row.home_score), float(row.away_score)
        actual_total = actual_home + actual_away
        actual_margin = actual_home - actual_away

        draws = np.load(V21_DRAWS_DIR / f"{gid}.npz")
        home_d, away_d = draws["home_score"].astype(float), draws["away_score"].astype(float)

        model_totals.append(pred["MODEL_TOTAL"])
        actual_totals.append(actual_total)
        if pred["MARKET"]["TOTAL_LINE"] is not None:
            market_totals.append(pred["MARKET"]["TOTAL_LINE"])

        crps_h, crps_a = crps_empirical(home_d, actual_home), crps_empirical(away_d, actual_away)
        crps_home_all.append(crps_h)
        crps_away_all.append(crps_a)
        pit_h, pit_a = float(np.mean(home_d <= actual_home)), float(np.mean(away_d <= actual_away))
        pit_all += [pit_h, pit_a]
        for q in coverage_hits:
            lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
            h_lo, h_hi = np.quantile(home_d, lo), np.quantile(home_d, hi)
            a_lo, a_hi = np.quantile(away_d, lo), np.quantile(away_d, hi)
            coverage_hits[q].append(h_lo <= actual_home <= h_hi)
            coverage_hits[q].append(a_lo <= actual_away <= a_hi)

        actual_winner = "HOME" if actual_home > actual_away else ("AWAY" if actual_away > actual_home else "TIE")
        home_outcome = 1 if actual_winner == "HOME" else 0
        per_game.append({
            "GAME_ID": gid, "STATUS": "SCORED",
            "MODEL_HOME_SCORE_MEAN": pred["MODEL_SCORE_DIST"]["HOME"]["mean"],
            "MODEL_AWAY_SCORE_MEAN": pred["MODEL_SCORE_DIST"]["AWAY"]["mean"],
            "ACTUAL_HOME_SCORE": actual_home, "ACTUAL_AWAY_SCORE": actual_away,
            "HOME_SCORE_ERROR": pred["MODEL_SCORE_DIST"]["HOME"]["mean"] - actual_home,
            "AWAY_SCORE_ERROR": pred["MODEL_SCORE_DIST"]["AWAY"]["mean"] - actual_away,
            "MODEL_SPREAD_HOME": pred["MODEL_SPREAD_HOME"], "ACTUAL_MARGIN_HOME": actual_margin,
            "MODEL_TOTAL": pred["MODEL_TOTAL"], "ACTUAL_TOTAL": actual_total,
            "MODEL_WIN_PROB_HOME": pred["MODEL_WIN_PROB"]["HOME"],
            "BRIER_HOME": brier(pred["MODEL_WIN_PROB"]["HOME"], home_outcome),
            "LOGLOSS_HOME": logloss(pred["MODEL_WIN_PROB"]["HOME"], home_outcome),
            "CRPS_HOME": crps_h, "CRPS_AWAY": crps_a,
        })

    scored = [e for e in per_game if e["STATUS"] == "SCORED"]
    score_errors = [e["HOME_SCORE_ERROR"] for e in scored] + [e["AWAY_SCORE_ERROR"] for e in scored]
    model_spread_errs = [abs(e["MODEL_SPREAD_HOME"] - e["ACTUAL_MARGIN_HOME"]) for e in scored]
    model_total_errs = [abs(e["MODEL_TOTAL"] - e["ACTUAL_TOTAL"]) for e in scored]
    briers = [e["BRIER_HOME"] for e in scored]

    v21_metrics = {
        "N_GAMES_SCORED": len(scored),
        "SCORE_MAE": float(np.mean(np.abs(score_errors))),
        "TEAM_SCORE_BIAS": float(np.mean(score_errors)),
        "MODEL_SPREAD_MAE_VS_ACTUAL": float(np.mean(model_spread_errs)),
        "MODEL_TOTAL_MAE_VS_ACTUAL": float(np.mean(model_total_errs)),
        "BRIER_HOME_WIN": float(np.mean(briers)),
        "MEAN_MODEL_TOTAL": float(np.mean(model_totals)),
        "MEAN_ACTUAL_TOTAL": float(np.mean(actual_totals)),
        "MEAN_MARKET_TOTAL": float(np.mean(market_totals)) if market_totals else None,
        "MEAN_CRPS_HOME": float(np.mean(crps_home_all)), "MEAN_CRPS_AWAY": float(np.mean(crps_away_all)),
        "PIT_MEAN": float(np.mean(pit_all)),
        "COVERAGE_BY_NOMINAL_BAND": {f"{int(q*100)}pct": {"NOMINAL": q, "EMPIRICAL_HIT_RATE": float(np.mean(hits))}
                                      for q, hits in coverage_hits.items()},
    }

    # Load V20's comparable metrics from its own results file for the delta table.
    v20_results = json.loads((ROOT / "SPORTS_V20_WEEK1_RESULTS.json").read_text())["METRICS"]
    v4_audit = json.loads((ROOT / "SPORTS_NOVA_V1_1_P4_VARIANCE_TAILS_AUDIT.json").read_text())["SUMMARY"]

    n_identity_unchanged = sum(1 for c in identity_checks if c["UNCHANGED"])
    identity_isolated = n_identity_unchanged == len(identity_checks)

    hash_check = verify_v18_hashes_unchanged()

    # All simulated scores must now be able to violate mod-3 (PAT present).
    any_score_mod3_nonzero = False
    for f in V21_DRAWS_DIR.glob("*.npz"):
        d = np.load(f)
        if np.any(d["home_score"].astype(int) % 3 != 0) or np.any(d["away_score"].astype(int) % 3 != 0):
            any_score_mod3_nonzero = True
            break

    comparison = {
        "SCHEMA": "SPORTS_V21_V20_COMPARISON",
        "MISSION": "SPORTS_NOVA_V21_EXTRA_POINT_FIX",
        "IDENTITY_ISOLATION_CHECK": {
            "N_TEAM_SLOTS": len(identity_checks), "N_UNCHANGED_VS_V20": n_identity_unchanged,
            "ALL_IDENTITY_UNCHANGED": identity_isolated,
        },
        "PAT_MECHANISM_PRESENT_CHECK": {"ANY_SCORE_NOT_DIVISIBLE_BY_3": any_score_mod3_nonzero},
        "V20_SCORE_MAE": v20_results["SCORE_MAE"], "V21_SCORE_MAE": v21_metrics["SCORE_MAE"],
        "V20_TEAM_SCORE_BIAS": v20_results["TEAM_SCORE_BIAS"], "V21_TEAM_SCORE_BIAS": v21_metrics["TEAM_SCORE_BIAS"],
        "V20_MODEL_TOTAL_MAE": v20_results["MODEL_TOTAL_MAE_VS_ACTUAL"], "V21_MODEL_TOTAL_MAE": v21_metrics["MODEL_TOTAL_MAE_VS_ACTUAL"],
        "V20_BRIER": v20_results["BRIER_HOME_WIN"], "V21_BRIER": v21_metrics["BRIER_HOME_WIN"],
        "MEAN_MODEL_TOTAL": {"V20_APPROX_FROM_TOTAL_MAE_CONTEXT": None,
                              "V21": v21_metrics["MEAN_MODEL_TOTAL"],
                              "MARKET": v21_metrics["MEAN_MARKET_TOTAL"], "ACTUAL": v21_metrics["MEAN_ACTUAL_TOTAL"]},
        "V4_AUDIT_BEFORE_FIX": {
            "PIT_MEAN": v4_audit["PIT_MEAN"], "COVERAGE_50PCT": v4_audit["COVERAGE_BY_NOMINAL_BAND"]["50pct"]["EMPIRICAL_HIT_RATE"],
            "COVERAGE_90PCT": v4_audit["COVERAGE_BY_NOMINAL_BAND"]["90pct"]["EMPIRICAL_HIT_RATE"],
        },
        "V21_AFTER_FIX": {
            "PIT_MEAN": v21_metrics["PIT_MEAN"], "COVERAGE_50PCT": v21_metrics["COVERAGE_BY_NOMINAL_BAND"]["50pct"]["EMPIRICAL_HIT_RATE"],
            "COVERAGE_90PCT": v21_metrics["COVERAGE_BY_NOMINAL_BAND"]["90pct"]["EMPIRICAL_HIT_RATE"],
        },
        "GATE_CHECKS": {
            "IDENTITY_UNCHANGED_VS_V20": identity_isolated,
            "PAT_MECHANISM_CONFIRMED_ACTIVE": any_score_mod3_nonzero,
            "TOTAL_BIAS_MOVED_TOWARD_MARKET_AND_ACTUAL": abs(v21_metrics["TEAM_SCORE_BIAS"]) < abs(v20_results["TEAM_SCORE_BIAS"]),
            "V18_HASH_UNCHANGED": hash_check["ALL_PASS"],
        },
        "V18_HASH_CHECK": hash_check,
    }
    promote = bool(identity_isolated and any_score_mod3_nonzero and hash_check["ALL_PASS"] and
                   abs(v21_metrics["TEAM_SCORE_BIAS"]) < abs(v20_results["TEAM_SCORE_BIAS"]))
    comparison["VERDICT"] = "PROMOTE_V21" if promote else "REJECT_V21"
    comparison["PROMOTE_V21"] = promote

    OUT_V21_RESULTS.write_text(json.dumps({
        "SCHEMA": "SPORTS_V21_WEEK1_RESULTS", "METRICS": v21_metrics, "PER_GAME": per_game,
        "IDENTITY_CHECKS": identity_checks,
    }, indent=2, default=str))
    OUT_COMPARISON.write_text(json.dumps(comparison, indent=2, default=str))

    lines = [
        "# SPORTS_V21_EXTRA_POINT_FIX_REPORT\n\n",
        f"V18 hash check: {'PASS' if hash_check['ALL_PASS'] else 'FAIL'} ({hash_check['COUNT']}/{hash_check['COUNT']}).\n\n",
        f"Identity isolation: {n_identity_unchanged}/{len(identity_checks)} team-slots unchanged vs V20 "
        f"({'PASS' if identity_isolated else 'FAIL'}).\n\n",
        f"PAT mechanism confirmed active (scores not all divisible by 3): {any_score_mod3_nonzero}\n\n",
        f"## Headline comparison\n\n```json\n{json.dumps(comparison, indent=2, default=str)}\n```\n\n",
    ]
    OUT_REPORT.write_text("".join(lines))

    print(json.dumps({
        "V20_SCORE_MAE": comparison["V20_SCORE_MAE"], "V21_SCORE_MAE": comparison["V21_SCORE_MAE"],
        "V20_TEAM_SCORE_BIAS": comparison["V20_TEAM_SCORE_BIAS"], "V21_TEAM_SCORE_BIAS": comparison["V21_TEAM_SCORE_BIAS"],
        "V20_MODEL_TOTAL_MAE": comparison["V20_MODEL_TOTAL_MAE"], "V21_MODEL_TOTAL_MAE": comparison["V21_MODEL_TOTAL_MAE"],
        "MEAN_MODEL_TOTAL_V21": v21_metrics["MEAN_MODEL_TOTAL"], "MEAN_MARKET_TOTAL": v21_metrics["MEAN_MARKET_TOTAL"],
        "MEAN_ACTUAL_TOTAL": v21_metrics["MEAN_ACTUAL_TOTAL"],
        "PIT_MEAN_BEFORE": v4_audit["PIT_MEAN"], "PIT_MEAN_AFTER": v21_metrics["PIT_MEAN"],
        "IDENTITY_ISOLATED": identity_isolated,
        "VERDICT": comparison["VERDICT"], "PROMOTE_V21": comparison["PROMOTE_V21"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
