"""SPORTS_NOVA_V18_PROSPECTIVE_CAPTURE_GATE -- outcome join (SEPARATE from capture).

Reads the prospective ledger written by
sports_nova_v3_v18_prospective_capture.py and, for any row not yet SETTLED,
looks for a realized outcome to join. It NEVER imports the capture script,
NEVER calls make_state/simulate_game, and NEVER regenerates or overwrites a
prediction artifact or its PREDICTION_HASH -- it only appends outcome fields
to the ledger row once a real result exists. That separation is the point:
this script cannot leak a result back into a prediction because it has no
code path into prediction generation at all.

Outcome source: the same causal player panel used for feature construction
(NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet), once/if it is refreshed to include
the 2026 season's completed games. Until Node2 (or an equivalent ingestion
job) appends 2026 rows to that panel, this script correctly finds zero rows
to settle -- that is the honest state of the world today, not a bug.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
PROSPECTIVE = DATA / "prospective"
LEDGER_PATH = PROSPECTIVE / "ledger" / "SPORTS_NOVA_V18_PROSPECTIVE_LEDGER.json"
PRED_DIR = PROSPECTIVE / "predictions"
PLAYER_PANEL = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"

IMMUTABLE_LEDGER_FIELDS = {
    "GAME_ID", "TEAM", "OPPONENT", "QB", "PREDICTION_HASH",
    "PREDICTION_TIME", "KICKOFF_UTC",
}


def load_ledger() -> dict:
    if not LEDGER_PATH.exists():
        raise SystemExit(f"no ledger found at {LEDGER_PATH}")
    return json.loads(LEDGER_PATH.read_text())


def save_ledger(ledger: dict):
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def realized_row(panel: pd.DataFrame, game_id: str, team: str, qb_id: str):
    m = panel[(panel.GAME_ID == game_id) & (panel.TEAM == team) & (panel.PLAYER_ID == qb_id)]
    if m.empty:
        return None
    return m.iloc[0]


def score(pred_path: Path, realized) -> dict:
    artifact = json.loads(pred_path.read_text())
    actual = float(realized["PASS_YARDS"])
    pred_mean = float(artifact["EXPECTED_PASS_YARDS"])
    metrics = {
        "MAE": abs(actual - pred_mean),
        "ERROR": actual - pred_mean,
        "SQUARED_ERROR": (actual - pred_mean) ** 2,
    }
    for thr_str, p_over in artifact["THRESHOLD_PROBABILITIES"].items():
        thr = float(thr_str)
        outcome_over = 1.0 if actual > thr else 0.0
        p = min(max(p_over, 1e-6), 1 - 1e-6)
        metrics[f"BRIER_{thr_str}"] = (p - outcome_over) ** 2
        metrics[f"LOGLOSS_{thr_str}"] = -(outcome_over * np.log(p) + (1 - outcome_over) * np.log(1 - p))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="report what would settle without writing the ledger")
    args = ap.parse_args()

    ledger = load_ledger()
    if not PLAYER_PANEL.exists():
        raise SystemExit(f"no player panel at {PLAYER_PANEL}")
    panel = pd.read_parquet(PLAYER_PANEL)

    n_checked = n_settled = n_still_pending = n_no_panel_row = 0
    for key, row in ledger.items():
        if row.get("SETTLED"):
            continue
        n_checked += 1
        before = {k: row.get(k) for k in IMMUTABLE_LEDGER_FIELDS}
        realized = realized_row(panel, row["GAME_ID"], row["TEAM"], row["QB"])
        if realized is None:
            n_no_panel_row += 1
            n_still_pending += 1
            continue
        pred_path = PRED_DIR / f"{row['GAME_ID']}__{row['TEAM']}.json"
        metrics = score(pred_path, realized)
        row["OUTCOME_TIME"] = datetime.now(timezone.utc).isoformat()
        row["PASS_YARDS_ACTUAL"] = float(realized["PASS_YARDS"])
        row["ATTEMPT_ACTUAL"] = float(realized["PASS_ATTEMPTS"])
        row["COMPLETION_ACTUAL"] = float(realized["COMPLETIONS"])
        row["MODEL_METRICS"] = metrics
        row["SETTLED"] = True
        after = {k: row.get(k) for k in IMMUTABLE_LEDGER_FIELDS}
        assert before == after, f"outcome join mutated an immutable prediction field for {key}"
        n_settled += 1

    if not args.dry_run and n_settled:
        save_ledger(ledger)

    print(json.dumps({
        "LEDGER_ROWS_TOTAL": len(ledger),
        "ALREADY_SETTLED": sum(1 for r in ledger.values() if r.get("SETTLED")) - n_settled,
        "CHECKED_THIS_RUN": n_checked,
        "NEWLY_SETTLED_THIS_RUN": n_settled,
        "STILL_PENDING_NO_PANEL_ROW": n_no_panel_row,
        "DRY_RUN": args.dry_run,
        "PANEL_MAX_SEASON": int(panel.SEASON.max()) if len(panel) else None,
    }, indent=2))


if __name__ == "__main__":
    main()
