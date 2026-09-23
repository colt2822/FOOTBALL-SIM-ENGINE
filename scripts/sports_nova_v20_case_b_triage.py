"""SPORTS_NOVA_V20 -- pre-implementation triage (read-only, no model change).

For each of the 6 actual QB-identity mismatches from the V19 Week-1-2026
replay (SPORTS_V19_WEEK1_RESULTS.json), determine WHY that team's
constructed roster (`make_state`) does not contain the actual starter, using
the exact same panel/cutoff/make_state the replay harness uses. This decides
whether V20 needs a roster-injection layer, a `latest_team` tie-break fix, or
neither (resolver-wrong cases are out of scope for a roster fix).

Buckets:
  NO_ROWS_ANYWHERE       -- player has zero rows in `prior` for any team
                            (rookie / no causal history at all).
  ROWS_OTHER_TEAM_ONLY   -- player has prior rows, but none with TEAM==tid
                            (traded/signed away from the team `prior` last
                            saw him with -- genuine Case B, injection needed).
  ROWS_FOR_TID_BUT_STALE_LATEST_TEAM
                          -- player HAS rows with TEAM==tid, but his
                            latest-by-(SEASON,WEEK) row is a different team,
                            so the `latest_team` filter drops him -- NOT an
                            injection case, a tie-break/ordering defect.
  PRESENT_IN_ROSTER      -- player is actually in pregame.players for tid
                            (excluded only by _qb_shares' min-gap filter --
                            Case A, already in V19's scope, not V20's).
  DUPLICATE_SEASON_WEEK  -- flagged additionally if the player has >1 row at
                            the same (SEASON, WEEK) anywhere in `prior`
                            (makes `drop_duplicates(keep="last")` order-
                            dependent since pandas sort is not stable).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "sports_nova_v3"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
SCHEDULE_CSV = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
RESULTS_PATH = ROOT / "SPORTS_V19_WEEK1_RESULTS.json"
SEASON, WEEK = 2026, 1

MISMATCHES = [
    # (game_id, team, actual_starter_id, actual_starter_name)
]


def parse_kickoff_utc(gameday, gametime):
    if pd.isna(gametime):
        return None
    d = datetime.strptime(str(gameday), "%Y-%m-%d")
    hh, mm = str(gametime).split(":")
    local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def load_mismatches():
    d = json.loads(RESULTS_PATH.read_text())
    mismatches = set(tuple(m) for m in d["METRICS"]["QB_IDENTITY_MISMATCHES_DETAIL"])
    out = []
    for pg in d["PER_GAME"]:
        for side_key in ("AWAY_QB_JOIN", "HOME_QB_JOIN"):
            block = pg[side_key]
            gid = block["IDENTITY_BLOCK"]["RESOLUTION"]["GAME_ID"]
            if (gid, side_key) in mismatches:
                team = block["IDENTITY_BLOCK"]["RESOLUTION"]["TEAM"]
                out.append((gid, team, block["ACTUAL_STARTER_ID"], block["ACTUAL_STARTER_NAME"],
                            block["IDENTITY_BLOCK"]["RESOLUTION"]["QB_ID"],
                            block["IDENTITY_BLOCK"]["RESOLUTION"]["QB_NAME"]))
    return out


def main():
    spec = importlib.util.spec_from_file_location("wf_v20_triage", WALKFORWARD_SCRIPT)
    wf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wf)

    panel = pd.read_parquet(LIVE_PANEL)
    panel["_event_time"] = pd.to_datetime(panel["EVENT_TIME"], utc=True, errors="coerce")
    sched = pd.read_csv(SCHEDULE_CSV)

    mismatches = load_mismatches()
    report = []
    for gid, team, actual_id, actual_name, res_id, res_name in mismatches:
        parts = gid.split("_")
        season, week, away, home = int(parts[0]), int(parts[1]), parts[2], parts[3]
        r = sched[(sched.season == season) & (sched.week == week) &
                  (sched.away_team == away) & (sched.home_team == home)].iloc[0]
        kickoff = parse_kickoff_utc(r.gameday, r.gametime)
        prior = panel[(panel._event_time < kickoff) & (panel.SEASON >= season - 5)].copy()
        prior_for_state = prior.drop(columns=["_event_time"])

        state = wf.make_state(gid, prior_for_state, kickoff, None)
        roster_ids = {p.player_id for p in state.players if p.team_id == team}
        present_in_roster = actual_id in roster_ids

        rows_any = prior_for_state[prior_for_state.PLAYER_ID == actual_id]
        rows_tid = rows_any[rows_any.TEAM == team]
        latest_team_map = (prior_for_state.sort_values(["SEASON", "WEEK"])
                            .drop_duplicates("PLAYER_ID", keep="last")
                            .set_index("PLAYER_ID")["TEAM"].to_dict())
        latest_team_for_player = latest_team_map.get(actual_id)

        dup_sw = (rows_any.groupby(["SEASON", "WEEK"]).size() > 1).any() if len(rows_any) else False

        if present_in_roster:
            bucket = "PRESENT_IN_ROSTER"
        elif len(rows_any) == 0:
            bucket = "NO_ROWS_ANYWHERE"
        elif len(rows_tid) == 0:
            bucket = "ROWS_OTHER_TEAM_ONLY"
        elif latest_team_for_player != team:
            bucket = "ROWS_FOR_TID_BUT_STALE_LATEST_TEAM"
        else:
            bucket = "UNEXPLAINED"

        report.append({
            "GAME_ID": gid, "TEAM": team,
            "ACTUAL_STARTER_ID": actual_id, "ACTUAL_STARTER_NAME": actual_name,
            "RESOLUTION_QB_ID": res_id, "RESOLUTION_QB_NAME": res_name,
            "RESOLUTION_CORRECT": res_id == actual_id,
            "BUCKET": bucket,
            "N_ROWS_ANY_TEAM": int(len(rows_any)),
            "N_ROWS_FOR_TID": int(len(rows_tid)),
            "LATEST_TEAM_FOR_PLAYER": latest_team_for_player,
            "DUPLICATE_SEASON_WEEK_ROWS": bool(dup_sw),
            "ROWS_ANY_TEAM_DETAIL": rows_any[["SEASON", "WEEK", "TEAM"]].sort_values(["SEASON", "WEEK"]).to_dict("records"),
        })

    print(json.dumps(report, indent=2, default=str))
    out_path = DATA / "SPORTS_NOVA_V20_CASE_B_TRIAGE.json"
    out_path.write_text(json.dumps({
        "SCHEMA": "SPORTS_NOVA_V20_CASE_B_TRIAGE", "SEASON": SEASON, "WEEK": WEEK,
        "GENERATED_AT_UTC": datetime.now(timezone.utc).isoformat(),
        "RESULTS": report,
    }, indent=2, default=str))
    print("WROTE", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
