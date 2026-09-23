"""Reconcile the M1 V4 report with the completed raw-ledger battery."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
REPORT = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_DIAGNOSTIC_REPORT_V1.json"
BATTERY = DATA / "SPORTS_NOVA_M1_V4_VALIDATION_BATTERY_V1.json"
PILOT_REPORT = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_DIAGNOSTIC_PILOT_REPORT_V1.json"
V3 = DATA / "SPORTS_NOVA_V3_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet"
V21 = DATA / "SPORTS_NOVA_V21_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet"
V4 = DATA / "SPORTS_NOVA_V4_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet"


def common_metrics(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    keys = ["GAME_ID", "PLAYER_ID", "POSITION", "TARGET"]
    x = a.merge(b, on=keys, suffixes=("_A", "_B"))
    if x.empty:
        return {"N": 0}
    err = x.PRED_MEAN_A - x.OBSERVED_A
    return {"N": int(len(x)), "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt(np.mean(err * err))), "bias": float(err.mean()),
            "coverage90": float(((x.OBSERVED_A >= x.P10_A) & (x.OBSERVED_A <= x.P90_A)).mean())}


def main() -> None:
    report = json.loads(REPORT.read_text())
    battery = json.loads(BATTERY.read_text())
    pilot = json.loads(PILOT_REPORT.read_text()) if PILOT_REPORT.is_file() else {}
    v3 = pd.read_parquet(V3); v21 = pd.read_parquet(V21); v4 = pd.read_parquet(V4)
    common = {}
    for pos in ("QB", "RB", "WR", "TE"):
        a = v4[v4.POSITION == pos]; b = v21[v21.POSITION == pos]; c = v3[v3.POSITION == pos]
        common[pos] = {"V3_vs_V21_common": common_metrics(c, b),
                       "V4_vs_V21_common": common_metrics(a, b)}
    qb = battery["player_metrics"]["QB"]
    v21_qb = report["comparison_artifacts"]["V21"]["by_position"]["QB"]
    v3_qb = report["comparison_artifacts"]["V3"]["by_position"]["QB"]
    report.update({
        "V4_STATUS": "PARTIAL",
        "RAW_LEDGER_STATUS": "COMPLETE",
        "RAW_LEDGER_ROWS": 27088,
        "TEMPORAL_FIREWALL": "PASS_EVENT_ORDER_ONLY_PUBLICATION_FIREWALL_FAIL",
        "PUBLICATION_TIME_CERTIFICATION": "FAIL_NO_ROW_LEVEL_PUBLICATION_TIMESTAMPS",
        "ACCOUNTING_STATUS": "PASS_V4; V3_V21_COMPARATOR_FAILURES_PRESERVED",
        "FIRST_FAILURE_STAGE": "OPPORTUNITY_REALIZATION",
        "QB_UNDERPRODUCTION_ROOT_CAUSE": "OTHER: wrong QB selected and roster-state failure occur before QB yardage realization; the pilot reason ledger shows WRONG_QB_SELECTED/ROSTER_STATE_FAILURE, so this is not diagnosed as too few attempts, completion rate, or YPC.",
        "QB_PASS_YARD_MAE_V21": v21_qb["MAE"],
        "QB_PASS_YARD_MAE_V3": v3_qb["MAE"],
        "QB_PASS_YARD_MAE_V4": qb["MAE"],
        "JOINT_DELTA": battery["joint_delta_vs_V21"],
        "TAIL_DELTA": {"V4": {k: battery["player_metrics"][k] for k in ("QB", "RB", "WR", "TE")},
                       "note": "V4 tail quantiles are in the battery; V21 comparator tails remain in frozen comparison_artifacts."},
        "GAME_LEVEL_DELTA": battery["game_level"],
        "DISTRIBUTIONAL_DELTA": {k: v["coverage_50_80_90"] for k, v in battery["player_metrics"].items()},
        "pilot_comparison_stage_metrics": pilot.get("stage_metrics", {}),
        "PLAYER_ALLOCATION_V21": pilot.get("PLAYER_ALLOCATION_V21"),
        "PLAYER_ALLOCATION_V3": pilot.get("PLAYER_ALLOCATION_V3"),
        "PLAYER_ALLOCATION_V4": pilot.get("PLAYER_ALLOCATION_V4"),
        "common_population_metrics": common,
        "comparison_artifacts": {**report["comparison_artifacts"], "V4": battery},
        "PROMOTION_DECISION": "DO_NOT_PROMOTE: V4 raw/accounting gates pass, but publication certification fails, QB MAE is not materially better than V21, and game-score/joint distribution gates are not sufficient for promotion.",
        "CHAMPION_AFTER_TEST": "V21",
        "STATE_SPACE_CHALLENGER_NEEDED": "NO_NOT_YET",
        "REAL_BLOCKER": "No row-level publication/availability timestamps; incomplete 2020-2024 historical depth-chart coverage; no historical pregame injury input; and the player artifact does not expose final game scores for a complete game-level battery.",
        "NEXT_SINGLE_ACTION": "Acquire and hash a complete row-level publication/availability plus 2020-2024 roster/injury package, then rerun the frozen V4 battery without changing cohort, cutoffs, or mechanics.",
    })
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["V4_STATUS"], "qb_mae": report["QB_PASS_YARD_MAE_V4"],
                      "promotion": report["PROMOTION_DECISION"]}, indent=2))


if __name__ == "__main__":
    main()
