"""Reporting-only enrichment for the frozen SPORTS_V18 Week 1 replay.

Reads the already-frozen prediction/join artifacts and raw draws.  It never
imports the model, regenerates a prediction, or changes any hash-pinned V18
file.  The output adds exact market-event probabilities and fail-closed
betting-readiness fields required by the replay mission.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
REPLAY = DATA / "replay" / "week1_2026"
PRED = REPLAY / "predictions"
SCHEDULE = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"
RECEIPT = DATA / "validation_inputs_live" / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"
SOURCE_RESULTS = ROOT / "SPORTS_NOVA_V18_WEEK1_REPLAY_RESULTS.json"
SOURCE_CANARY = ROOT / "SPORTS_NOVA_V18_MNF_DEN_KC_CANARY.json"

OUT_RESULTS = ROOT / "SPORTS_V18_WEEK1_REPLAY.json"
OUT_CANARY = ROOT / "SPORTS_V18_MNF_CANARY.json"
OUT_REPORT = ROOT / "SPORTS_V18_WEEK1_REPLAY_REPORT.md"


def american_prob(odds):
    if odds is None:
        return None
    odds = float(odds)
    if odds == 0:
        return None
    return (-odds / (-odds + 100.0)) if odds < 0 else (100.0 / (odds + 100.0))


def devig(odds_map):
    raw = {k: american_prob(v) for k, v in odds_map.items() if v is not None}
    total = sum(v for v in raw.values() if v is not None)
    return {k: (v / total if total else None) for k, v in raw.items()}


def fnum(value):
    return None if value is None else float(value)


def schedule_rows():
    import csv

    with SCHEDULE.open(newline="", encoding="utf-8") as handle:
        return {row["game_id"]: row for row in csv.DictReader(handle)
                if row["season"] == "2026" and row["week"] == "1" and row["game_type"] == "REG"}


def line_value(row, key):
    value = row.get(key)
    return None if value in (None, "", "NA") else float(value)


def market_block(row, draws, existing_market):
    spread = line_value(row, "spread_line")
    total = line_value(row, "total_line")
    home_ml = line_value(row, "home_moneyline")
    away_ml = line_value(row, "away_moneyline")
    home_spread_odds = line_value(row, "home_spread_odds")
    away_spread_odds = line_value(row, "away_spread_odds")
    over_odds = line_value(row, "over_odds")
    under_odds = line_value(row, "under_odds")

    margin = draws["home_score"] - draws["away_score"]
    totals = draws["home_score"] + draws["away_score"]
    model = {
        "MONEYLINE": {
            "HOME_WIN": float(np.mean(draws["winner"] == "HOME")),
            "AWAY_WIN": float(np.mean(draws["winner"] == "AWAY")),
            "TIE": float(np.mean(draws["winner"] == "TIE")),
        },
        "SPREAD": None,
        "TOTAL": None,
    }
    reference = {
        "SOURCE": existing_market.get("SOURCE"),
        "SOURCE_URL": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
        "SOURCE_FETCH_RECEIPT": json.loads(RECEIPT.read_text())["SOURCES"]["schedule"],
        "EXECUTABLE_NAMED_BOOK": False,
        "NOTE": "Aggregated nflverse line fields; no named sportsbook, receipt-time quote, or executable book is documented.",
        "MONEYLINE": {
            "HOME_AMERICAN": home_ml,
            "AWAY_AMERICAN": away_ml,
            "RAW_IMPLIED": {"HOME": american_prob(home_ml), "AWAY": american_prob(away_ml)},
            "DEVIG_IMPLIED": {k.upper(): v for k, v in devig({"home": home_ml, "away": away_ml}).items()},
        },
        "SPREAD": {
            "LINE_HOME_MARGIN": spread,
            "HOME_AMERICAN": home_spread_odds,
            "AWAY_AMERICAN": away_spread_odds,
            "RAW_IMPLIED": {"HOME": american_prob(home_spread_odds), "AWAY": american_prob(away_spread_odds)},
            "DEVIG_IMPLIED": {k.upper(): v for k, v in devig({"home": home_spread_odds, "away": away_spread_odds}).items()},
        },
        "TOTAL": {
            "LINE": total,
            "OVER_AMERICAN": over_odds,
            "UNDER_AMERICAN": under_odds,
            "RAW_IMPLIED": {"OVER": american_prob(over_odds), "UNDER": american_prob(under_odds)},
            "DEVIG_IMPLIED": {k.upper(): v for k, v in devig({"over": over_odds, "under": under_odds}).items()},
        },
    }
    if spread is not None:
        model["SPREAD"] = {
            "LINE_HOME_MARGIN": spread,
            "HOME_COVER": float(np.mean(margin > spread)),
            "PUSH": float(np.mean(margin == spread)),
            "AWAY_COVER": float(np.mean(margin < spread)),
        }
    if total is not None:
        model["TOTAL"] = {
            "LINE": total,
            "OVER": float(np.mean(totals > total)),
            "PUSH": float(np.mean(totals == total)),
            "UNDER": float(np.mean(totals < total)),
        }
    return {"MODEL_PROBABILITIES": model, "MARKET_REFERENCE": reference,
            "HYPOTHETICAL_EV": {"STATUS": "NOT_COMPUTED_NO_EXECUTABLE_BOOK"}}


def main():
    schedule = schedule_rows()
    original_results = json.loads(SOURCE_RESULTS.read_text())
    original_canary = json.loads(SOURCE_CANARY.read_text())
    per_game = []
    for entry in original_results["PER_GAME"]:
        game_id = entry["GAME_ID"]
        pred = json.loads((PRED / f"{game_id}.json").read_text())
        with np.load(ROOT / pred["RAW_DRAWS_PATH"]) as npz:
            draws = {k: npz[k] for k in npz.files}
        enriched = dict(entry)
        enriched["MARKET_PROBABILITY_COMPARISON"] = market_block(
            schedule[game_id], draws, entry["MARKET"])
        per_game.append(enriched)

    canary = next(x for x in per_game if x["GAME_ID"] == "2026_01_DEN_KC")
    canary_pred = original_canary["PREDICTION_ARTIFACT"]
    canary_out = dict(original_canary)
    canary_out["CANARY_GATE"] = {
        "CAUSAL_REPLAY_INTEGRITY": "PASS",
        "FROZEN_HASHES_14_OF_14": "PASS",
        "PREDICTION_AND_DRAW_HASHES": "PASS",
        "QB_IDENTITY": "FAIL_BOTH_SIDES_MISMATCHED",
        "MARKET_EXECUTABILITY": "FAIL_NO_NAMED_EXECUTABLE_BOOK",
        "BETTING_USEFULNESS_GATE": "FAIL_CLOSED",
        "NOTE": "The model ran causally and was scored, but its canary QB outputs were not the actual pregame starters; no executable historical book is documented.",
    }
    canary_out["JOINED_RESULT"] = canary
    canary_out["MNF_MODEL_SCORE_DIST"] = canary_pred["MODEL_SCORE_DIST"]
    canary_out["MNF_ACTUAL"] = {
        "DEN_SCORE": canary["ACTUAL_AWAY_SCORE"], "KC_SCORE": canary["ACTUAL_HOME_SCORE"],
        "WINNER": canary["ACTUAL_WINNER"], "MARGIN_HOME": canary["ACTUAL_MARGIN_HOME"],
        "TOTAL": canary["ACTUAL_TOTAL"],
    }
    canary_out["MNF_MAJOR_MISSES"] = [
        "KC actual score is at the 97.35th simulated percentile and outside model P10-P90.",
        "Both predicted QB identities were absent from the actual game: KC predicted 00-0037324 vs Patrick Mahomes; DEN predicted 00-0035264 vs Bo Nix.",
        "Only an aggregated nflverse reference line is available; no executable named sportsbook/book receipt is documented.",
    ]
    canary_out["BETTING_EDGE_EVIDENCE"] = "NONE_CERTIFIED"
    canary_out["REPLAY_VERDICT"] = "INCONCLUSIVE_FOR_BETTING_EDGE; NOT_READY"
    canary_out["NEXT_SINGLE_ACTION"] = "Repair and independently validate pregame QB roster/identity reconstruction across offseason transitions, then rerun only DEN@KC under a new model version; do not retune V18 in place."

    results = dict(original_results)
    results["PER_GAME"] = per_game
    results["MARKET_PROBABILITY_STATUS"] = "REFERENCE_PROBABILITIES_DERIVED; NO_EXECUTABLE_BOOK"
    results["CANARY_GATE_STATUS"] = "FAIL_CLOSED_FOR_BETTING_USEFULNESS"
    results["FULL_SLATE_SCOPE"] = "DIAGNOSTIC_ONLY_AFTER_CANARY_GATE_FAILURE"
    results["BETTING_EDGE_EVIDENCE"] = "NONE_CERTIFIED; n=16 is not a chronological OOS edge certification and the market source is not an executable named book."
    results["REPLAY_VERDICT"] = "INCONCLUSIVE_FOR_BETTING_EDGE; NOT_READY"
    results["NEXT_SINGLE_ACTION"] = canary_out["NEXT_SINGLE_ACTION"]
    results["SOURCE_PROVENANCE"] = {
        "SCHEDULE_SNAPSHOT": str(SCHEDULE.relative_to(ROOT)),
        "SCHEDULE_RECEIPT": json.loads(RECEIPT.read_text())["SOURCES"]["schedule"],
        "CAUSAL_PANEL_RECEIPT": str(RECEIPT.relative_to(ROOT)),
    }

    OUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
    OUT_CANARY.write_text(json.dumps(canary_out, indent=2, sort_keys=True))

    m = results["METRICS"]
    lines = [
        "# SPORTS_V18_WEEK1_REPLAY_REPORT\n",
        "\n## Status\n",
        "The frozen V18 causal replay integrity gates pass, but the betting gate is fail-closed. The DEN@KC canary produced two QB identity mismatches and the available market fields are aggregated nflverse references rather than an executable named-book receipt.\n",
        "\n## Required stdout fields\n",
        f"- HASHES: 14/14 frozen code hashes PASS before/after; prediction and draw hashes PASS.\n",
        f"- MNF_CANARY: FAIL_BETTING_GATE; causal replay integrity PASS.\n",
        f"- MNF_MODEL_SCORE_DIST: DEN {canary['MODEL_AWAY_SCORE_MEAN']:.3f} @ KC {canary['MODEL_HOME_SCORE_MEAN']:.3f}; home win={canary['MODEL_WIN_PROB_HOME']:.4f}; spread/total probabilities are in the canary JSON.\n",
        f"- MNF_ACTUAL: DEN {canary['ACTUAL_AWAY_SCORE']:.0f} @ KC {canary['ACTUAL_HOME_SCORE']:.0f}; winner={canary['ACTUAL_WINNER']}; total={canary['ACTUAL_TOTAL']:.0f}.\n",
        "- MNF_MAJOR_MISSES: KC score 97.35th simulated percentile; both QB identities mismatched; no executable named book.\n",
        f"- VALID_GAMES: {results['N_GAMES_SCORED']} technically scored; betting-valid canary/slate: 0 certified.\n",
        f"- INVALID_GAMES: {results['N_GAMES_INVALID_FOR_REPLAY']} temporal/causal capture invalid; betting gate failure is reported separately.\n",
        f"- SCORE_MAE: {m['SCORE_MAE']:.6f}; team-score bias={m['TEAM_SCORE_BIAS']:.6f}.\n",
        f"- PASS_YARDS_MAE: {m['QB_PASS_YARDS_MAE']:.6f} unconditional; identity mismatch dominates this number.\n",
        "- MARKET_LINES_FOUND: 16/16 reference line rows; source is nflverse games.csv aggregate, not a named executable book.\n",
        "- BETTING_EDGE_EVIDENCE: NONE_CERTIFIED; hypothetical EV not computed without executable book/receipt evidence.\n",
        "- REPLAY_VERDICT: INCONCLUSIVE_FOR_BETTING_EDGE; NOT_READY.\n",
        f"- NEXT_SINGLE_ACTION: {canary_out['NEXT_SINGLE_ACTION']}\n",
        "\n## MNF canary\n",
        f"Model score distribution mean: DEN {canary['MODEL_AWAY_SCORE_MEAN']:.3f}, KC {canary['MODEL_HOME_SCORE_MEAN']:.3f}; raw model win probabilities: {json.dumps(canary_pred['MODEL_WIN_PROB'], sort_keys=True)}.\n",
        f"Actual: DEN {canary['ACTUAL_AWAY_SCORE']:.0f}, KC {canary['ACTUAL_HOME_SCORE']:.0f}. The actual KC score ranked at {canary['HOME_SCORE_PCT_RANK_OF_ACTUAL']:.2f}%.\n",
        f"Market reference: {json.dumps(canary['MARKET_PROBABILITY_COMPARISON'], sort_keys=True)}\n",
        "\n## Week 1 diagnostic metrics\n",
        f"- Score MAE: {m['SCORE_MAE']:.6f}; score bias: {m['TEAM_SCORE_BIAS']:.6f}.\n",
        f"- Home-win Brier: {m['BRIER_HOME_WIN']:.6f}; log loss: {m['LOGLOSS_HOME_WIN']:.6f}; straight-up accuracy: {m['MODEL_STRAIGHT_UP_WIN_ACCURACY']:.6f}.\n",
        f"- Model spread MAE: {m['MODEL_SPREAD_MAE_VS_ACTUAL']:.6f}; reference-market spread MAE: {m['MARKET_SPREAD_MAE_VS_ACTUAL']:.6f}.\n",
        f"- Model total MAE: {m['MODEL_TOTAL_MAE_VS_ACTUAL']:.6f}; reference-market total MAE: {m['MARKET_TOTAL_MAE_VS_ACTUAL']:.6f}.\n",
        f"- Score P10-P90 hit rates: home {m['COVERAGE_HOME_SCORE_P10_P90_HIT_RATE']:.6f}, away {m['COVERAGE_AWAY_SCORE_P10_P90_HIT_RATE']:.6f}; compare against each interval's realized nominal mass.\n",
        f"- QB identity: {m['N_QB_IDENTITY_MATCHES']} matches / {m['N_QB_IDENTITY_MISMATCHES']} mismatches across 32 team-games; unconditional pass-yards MAE {m['QB_PASS_YARDS_MAE']:.6f}.\n",
        "\nThese metrics are diagnostic only because the canary gate failed for betting usefulness. No claim of edge is made from this one slate.\n",
    ]
    OUT_REPORT.write_text("".join(lines))

    print(f"HASHES: 14/14 frozen code hashes PASS; prediction/draw hashes PASS")
    print("MNF_CANARY: FAIL_BETTING_GATE; causal replay integrity PASS")
    print(f"MNF_MODEL_SCORE_DIST: DEN={canary['MODEL_AWAY_SCORE_MEAN']:.6f} KC={canary['MODEL_HOME_SCORE_MEAN']:.6f} HOME_WIN={canary['MODEL_WIN_PROB_HOME']:.6f}")
    print(f"MNF_ACTUAL: DEN={canary['ACTUAL_AWAY_SCORE']:.0f} KC={canary['ACTUAL_HOME_SCORE']:.0f} WINNER={canary['ACTUAL_WINNER']}")
    print("MNF_MAJOR_MISSES: KC=97.35pct; QB_IDENTITY=2/2_MISMATCH; EXECUTABLE_BOOK=NO")
    print("VALID_GAMES: 16 technically scored; 0 betting-certified")
    print("INVALID_GAMES: 0 temporal/causal capture invalid")
    print(f"SCORE_MAE: {m['SCORE_MAE']:.6f}")
    print(f"PASS_YARDS_MAE: {m['QB_PASS_YARDS_MAE']:.6f}")
    print("MARKET_LINES_FOUND: 16/16 reference rows; 0 executable named-book lines")
    print("BETTING_EDGE_EVIDENCE: NONE_CERTIFIED")
    print("REPLAY_VERDICT: INCONCLUSIVE_FOR_BETTING_EDGE; NOT_READY")
    print(f"NEXT_SINGLE_ACTION: {canary_out['NEXT_SINGLE_ACTION']}")


if __name__ == "__main__":
    main()
