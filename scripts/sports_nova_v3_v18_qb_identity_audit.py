"""SPORTS_NOVA_V18_QB_UNCERTAINTY_POLICY_AND_WEEKLY_REFRESH -- AUDIT phase.

Mirrors sports_nova_v3_v18_prospective_outcome_join.py's separation
guarantee (read ledger, read panel, append fields, never touch a written
prediction artifact or its hash) but scores QB IDENTITY match rather than
pass-yards error, against the LIVE ledger
(SPORTS_NOVA_V18_PROSPECTIVE_LEDGER_LIVE.json), split by the COHORT the
uncertainty policy assigned at prediction time (PRIMARY vs
EXCLUDED_UNCERTAIN_IDENTITY -- see sports_nova_v3_v18_qb_uncertainty_policy.py).

Never imports the capture script or simulator; never regenerates or
overwrites a prediction artifact or its PREDICTION_HASH.

Rolling mismatch rate is computed ONLY over rows this script can actually
settle against a real completed 2026 game in the live panel. Run today (only
2026 week 1 final, and the current smoke predictions all target week 2+),
this settles nothing -- reported as an honest zero, same pattern as
prospective_outcome_join.py's own "that is the honest state of the world
today, not a bug" note.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
PROSPECTIVE = DATA / "prospective"
LEDGER_LIVE_PATH = PROSPECTIVE / "ledger" / "SPORTS_NOVA_V18_PROSPECTIVE_LEDGER_LIVE.json"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
PRED_DIR_LIVE = PROSPECTIVE / "predictions_live"
AUDIT_SUMMARY_PATH = PROSPECTIVE / "SPORTS_NOVA_V18_QB_IDENTITY_AUDIT_SUMMARY.json"

IMMUTABLE_LEDGER_FIELDS = {
    "GAME_ID", "TEAM", "OPPONENT", "QB", "PREDICTION_HASH",
    "PREDICTION_TIME", "KICKOFF_UTC", "COHORT", "IDENTITY_STATUS_AT_PREDICTION_TIME",
}


def realized_leader(panel: pd.DataFrame, game_id: str, team: str):
    rows = panel[(panel.GAME_ID == game_id) & (panel.TEAM == team)]
    if rows.empty:
        return None
    qb_rows = rows[rows.POSITION == "QB"]
    pick = qb_rows if not qb_rows.empty else rows
    top = pick.loc[pick.PASS_ATTEMPTS.idxmax()]
    return {"PLAYER_ID": top.PLAYER_ID, "PLAYER_NAME": top.PLAYER_NAME}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ledger = json.loads(LEDGER_LIVE_PATH.read_text()) if LEDGER_LIVE_PATH.exists() else {}
    panel = pd.read_parquet(LIVE_PANEL) if LIVE_PANEL.exists() else None

    n_checked = n_settled = n_pending = 0
    by_cohort = {"PRIMARY": {"n": 0, "match": 0, "mismatch": 0},
                 "EXCLUDED_UNCERTAIN_IDENTITY": {"n": 0, "resolved_to_candidate_in_list": 0,
                                                  "resolved_to_uncaptured_candidate": 0}}

    for key, row in ledger.items():
        if row.get("IDENTITY_SETTLED"):
            continue
        n_checked += 1
        before = {k: row.get(k) for k in IMMUTABLE_LEDGER_FIELDS}
        cohort = row.get("COHORT", "PRIMARY")

        realized = realized_leader(panel, row["GAME_ID"], row["TEAM"]) if panel is not None else None
        if realized is None:
            n_pending += 1
            continue

        row["IDENTITY_OUTCOME_TIME"] = datetime.now(timezone.utc).isoformat()
        row["REALIZED_QB_ID"] = realized["PLAYER_ID"]
        row["REALIZED_QB_NAME"] = realized["PLAYER_NAME"]

        if cohort == "PRIMARY":
            match = row.get("QB") == realized["PLAYER_ID"]
            row["IDENTITY_MATCH"] = match
            by_cohort["PRIMARY"]["n"] += 1
            by_cohort["PRIMARY"]["match" if match else "mismatch"] += 1
        else:
            row["IDENTITY_MATCH"] = None  # not scored as a miss -- no prediction was made
            by_cohort["EXCLUDED_UNCERTAIN_IDENTITY"]["n"] += 1
            # Informational only: did the realized starter appear in the
            # candidate list this policy already knew about pregame? Read
            # from the excluded artifact itself, never written to.
            art_path = PRED_DIR_LIVE / f"{row['GAME_ID']}__{row['TEAM']}.json"
            listed_ids = set()
            if art_path.exists():
                cands = json.loads(art_path.read_text()).get("IDENTITY_RESOLUTION", {}).get("CANDIDATES", [])
                listed_ids = {c.get("PLAYER_ID") for c in cands if isinstance(c, dict)}
            was_listed = realized["PLAYER_ID"] in listed_ids
            row["REALIZED_QB_WAS_A_LISTED_CANDIDATE"] = was_listed
            by_cohort["EXCLUDED_UNCERTAIN_IDENTITY"][
                "resolved_to_candidate_in_list" if was_listed else "resolved_to_uncaptured_candidate"] += 1

        row["IDENTITY_SETTLED"] = True
        after = {k: row.get(k) for k in IMMUTABLE_LEDGER_FIELDS}
        assert before == after, f"audit mutated an immutable prediction field for {key}"
        n_settled += 1

    if not args.dry_run and n_settled:
        LEDGER_LIVE_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))

    primary_n = by_cohort["PRIMARY"]["n"]
    rolling_mismatch_rate = (
        round(by_cohort["PRIMARY"]["mismatch"] / primary_n, 4) if primary_n else None
    )

    summary = {
        "SCHEMA": "SPORTS_NOVA_V18_QB_IDENTITY_AUDIT_SUMMARY",
        "RUN_AT_UTC": datetime.now(timezone.utc).isoformat(),
        "LEDGER_ROWS_TOTAL": len(ledger),
        "CHECKED_THIS_RUN": n_checked,
        "NEWLY_SETTLED_THIS_RUN": n_settled,
        "STILL_PENDING_NO_PANEL_ROW": n_pending,
        "ROLLING_PRIMARY_MISMATCH_RATE": rolling_mismatch_rate,
        "BY_COHORT": by_cohort,
        "NOTE": (
            "Zero settled is the honest state of the world when no ledger row's game "
            "has a completed 2026 box score in the live panel yet -- not a bug. This "
            "number is INDEPENDENT of, and should be watched alongside, the 2025 "
            "methodology backcheck in SPORTS_NOVA_V18_CURRENT_IDENTITY_REFRESH_REPORT.json "
            "-- once real 2026 PRIMARY rows settle, this becomes the live figure that "
            "matters; the 2025 number is the only one available until then."
        ),
        "DRY_RUN": args.dry_run,
    }
    AUDIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
