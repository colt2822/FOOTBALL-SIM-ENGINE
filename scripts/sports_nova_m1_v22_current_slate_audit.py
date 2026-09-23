"""Read-only current-slate roster/QB input audit; no market columns are read."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
LIVE = DATA / "validation_inputs_live"
dc = pd.read_parquet(DATA / "validation_inputs" / "depth_charts_external" / "depth_charts_2026.parquet")
dc["dt"] = pd.to_datetime(dc.dt, utc=True)
sched = pd.read_csv(LIVE / "schedule_snapshot_2026.csv")
week = sched[(sched.season == 2026) & (sched.week == 2) & (sched.game_type == "REG")]

def kickoff(row):
    d = datetime.strptime(str(row.gameday), "%Y-%m-%d")
    hh, mm = map(int, str(row.gametime).split(":"))
    return pd.Timestamp(d.replace(hour=hh, minute=mm, tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc))

games = []
conflicts = []
for row in week.itertuples():
    k = kickoff(row)
    teams = {}
    timestamps = []
    for team, side in ((row.away_team, "away"), (row.home_team, "home")):
        q = dc[(dc.team == team) & (dc.dt < k)]
        ts = q.dt.max() if not q.empty else None
        snap = q[q.dt == ts] if ts is not None else q
        qbs = snap[snap.pos_name == "Quarterback"].sort_values("pos_rank")
        skill = snap[snap.pos_name.isin(["Running Back", "Wide Receiver", "Tight End"])]
        schedule_id = str(getattr(row, f"{side}_qb_id"))
        depth_id = str(qbs.iloc[0].gsis_id) if len(qbs) else None
        if schedule_id != depth_id:
            conflicts.append(f"{row.game_id}:{team}:SCHEDULE_QB={schedule_id}:DEPTH_QB1={depth_id}")
        if ts is not None:
            timestamps.append(ts)
        teams[team] = {"starting_qb_schedule_id": schedule_id,
                       "starting_qb_schedule_name": str(getattr(row, f"{side}_qb_name")),
                       "depth_qb1_id": depth_id,
                       "depth_qb1_name": str(qbs.iloc[0].player_name) if len(qbs) else None,
                       "depth_input_timestamp": str(ts) if ts is not None else None,
                       "rb_wr_te_pool_count": int(skill.gsis_id.nunique()),
                       "injury_status": "UNAVAILABLE",
                       "expected_inactive_players": "UNAVAILABLE"}
    games.append({"game_id": row.game_id, "kickoff_utc": k.isoformat(),
                  "input_timestamp": max(timestamps).isoformat() if timestamps else None, "teams": teams})

payload = {"SCHEMA": "SPORTS_NOVA_M1_V22_CURRENT_SLATE_INPUT_AUDIT", "WEEK": 2,
           "STATUS": "PARTIAL_NOT_USABLE", "GAMES": games,
           "QB_CONFLICTS": conflicts,
           "REASON": "No causal injury or expected-inactive source is present; ATL and SEA schedule/depth QB inputs conflict.",
           "NO_MARKET_DATA": True}
out = DATA / "SPORTS_NOVA_M1_V22_CURRENT_SLATE_INPUT_AUDIT.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
