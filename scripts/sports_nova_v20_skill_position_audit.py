"""SPORTS_NOVA_V1_1 P0 item 5 -- audit whether the make_state roster-absence
defect (fixed for QB in V20) also affects RB/WR/TE.

Read-only measurement, no fix attempted here (goal explicitly wants one
interpretable change per experiment -- this decides whether a follow-up
mission is warranted, it does not implement one).

Method: for each of the 32 2026-Week1 team-slots, find the actual game's
leading rusher (by RUSH_ATTEMPTS) and leading receiver (by TARGETS) from
the causal panel's own 2026-Week1 rows (same panel, same method already
used for QB ground truth), then check whether that player is present in
the UNMODIFIED (V19-base, not V20-injected) constructed roster for his
team. This measures the size of the problem, not a QB-specific artifact --
V20's injection is QB-only and untouched here.
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
OUT_PATH = DATA / "SPORTS_NOVA_V1_1_SKILL_POSITION_AUDIT.json"
SEASON, WEEK = 2026, 1


def parse_kickoff_utc(gameday, gametime):
    if pd.isna(gametime):
        return None
    d = datetime.strptime(str(gameday), "%Y-%m-%d")
    hh, mm = str(gametime).split(":")
    local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def main():
    spec = importlib.util.spec_from_file_location("wf_skill_audit", WALKFORWARD_SCRIPT)
    wf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wf)

    panel = pd.read_parquet(LIVE_PANEL)
    panel["_event_time"] = pd.to_datetime(panel["EVENT_TIME"], utc=True, errors="coerce")
    sched = pd.read_csv(SCHEDULE_CSV)
    week1 = sched[(sched.season == SEASON) & (sched.week == WEEK) & (sched.game_type == "REG")].copy()

    results = []
    for r in week1.itertuples():
        away, home = r.away_team, r.home_team
        gid = f"{SEASON}_{WEEK:02d}_{away}_{home}"
        kickoff = parse_kickoff_utc(r.gameday, r.gametime)
        if kickoff is None:
            continue
        prior = panel[(panel._event_time < kickoff) & (panel.SEASON >= SEASON - 5)].copy()
        prior_for_state = prior.drop(columns=["_event_time"])
        state = wf.make_state(gid, prior_for_state, kickoff, None)

        game_rows = panel[panel.GAME_ID == gid]
        for team in (home, away):
            roster_ids = {p.player_id for p in state.players if p.team_id == team}
            team_rows = game_rows[game_rows.TEAM == team]

            top_rusher = team_rows.sort_values("RUSH_ATTEMPTS", ascending=False).iloc[0] if len(team_rows) else None
            top_receiver = team_rows.sort_values("TARGETS", ascending=False).iloc[0] if len(team_rows) else None

            def check(row, min_val_col):
                if row is None or row[min_val_col] <= 0:
                    return None
                pid = str(row.PLAYER_ID)
                present = pid in roster_ids
                rows_any = prior_for_state[prior_for_state.PLAYER_ID == pid]
                rows_tid = rows_any[rows_any.TEAM == team]
                return {
                    "PLAYER_ID": pid, "PLAYER_NAME": str(row.PLAYER_NAME), "POSITION": str(row.POSITION),
                    "ACTUAL_VALUE": float(row[min_val_col]),
                    "PRESENT_IN_BASE_ROSTER": present,
                    "N_ROWS_ANY_TEAM": int(len(rows_any)), "N_ROWS_FOR_TID": int(len(rows_tid)),
                }

            rusher_check = check(top_rusher, "RUSH_ATTEMPTS")
            receiver_check = check(top_receiver, "TARGETS")
            results.append({"GAME_ID": gid, "TEAM": team,
                             "TOP_RUSHER": rusher_check, "TOP_RECEIVER": receiver_check})

    n_rusher_checked = sum(1 for r in results if r["TOP_RUSHER"])
    n_rusher_absent = sum(1 for r in results if r["TOP_RUSHER"] and not r["TOP_RUSHER"]["PRESENT_IN_BASE_ROSTER"])
    n_receiver_checked = sum(1 for r in results if r["TOP_RECEIVER"])
    n_receiver_absent = sum(1 for r in results if r["TOP_RECEIVER"] and not r["TOP_RECEIVER"]["PRESENT_IN_BASE_ROSTER"])

    summary = {
        "N_TEAM_SLOTS": len(results),
        "TOP_RUSHER_N_CHECKED": n_rusher_checked, "TOP_RUSHER_N_ABSENT_FROM_ROSTER": n_rusher_absent,
        "TOP_RECEIVER_N_CHECKED": n_receiver_checked, "TOP_RECEIVER_N_ABSENT_FROM_ROSTER": n_receiver_absent,
    }
    print(json.dumps(summary, indent=2))
    absent_detail = [r for r in results if
                      (r["TOP_RUSHER"] and not r["TOP_RUSHER"]["PRESENT_IN_BASE_ROSTER"]) or
                      (r["TOP_RECEIVER"] and not r["TOP_RECEIVER"]["PRESENT_IN_BASE_ROSTER"])]
    print(json.dumps(absent_detail, indent=2, default=str))
    OUT_PATH.write_text(json.dumps({
        "SCHEMA": "SPORTS_NOVA_V1_1_SKILL_POSITION_AUDIT", "SEASON": SEASON, "WEEK": WEEK,
        "GENERATED_AT_UTC": datetime.now(timezone.utc).isoformat(),
        "SUMMARY": summary, "ABSENT_DETAIL": absent_detail, "ALL_RESULTS": results,
    }, indent=2, default=str))
    print("WROTE", OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
