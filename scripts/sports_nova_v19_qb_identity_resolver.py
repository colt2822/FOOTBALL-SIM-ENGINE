"""SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER -- resolver phase.

Resolves a causal, pre-kickoff-only QB identity for every 2026 Week 1
team-slot (32 total), to be installed into the isolated
worker.sports_nova_v19 simulator's `_qb_shares` eligibility gate. Never
reads outcome data (any 2026 box score, the live panel's post-kickoff rows,
week-1 injury/roster reports). Produces SOURCE + AS_OF (dt) + CONFIDENCE
per team; UNRESOLVED (not a guess) when no pre-kickoff evidence exists.

RULE SELECTION (done on a DEV set, before touching 2026 -- see
scripts/sports_nova_v19_dev_rule_selection.py / the scratch run this
mission's report cites):
  2025 Week 1 is the only complete season-boundary week where the
  timestamped nflverse depth-chart asset (dt-per-snapshot schema,
  2025-08-03 onward) exists, giving a real, non-circular DEV set (n=32
  boundary team-games, ground truth = actual 2025 Week 1 box-score
  starter). Two candidate signals were compared there:
    - PANEL_LEADER (this team's most recent PRIOR game's leading passer,
      the same mechanism `_qb_shares`' `games_since_last_team_game`
      encodes): 18/32 = 56.2% accurate. This IS the season-boundary bug,
      reproduced directly on the DEV set.
    - DEPTH_CHART_QB1 (newest depth-chart snapshot strictly before
      kickoff): 32/32 = 100% accurate on the DEV set.
  The current CONFIRMED rule used by SPORTS_V18_CURRENT_SEASON_IDENTITY_
  REFRESH (agreement between panel-leader and depth-chart) only covers
  18/32 on this same DEV set -- it requires agreement with the very signal
  that is broken at a season boundary, so it would abstain on exactly the
  cases this mission needs to fix. Rule chosen here: DEPTH_CHART_QB1 ALONE,
  pre-kickoff, for a team-game that is a season boundary (2026 Week 1 by
  construction for this replay). No injury/roster corroboration -- both
  helper functions in the current_identity_refresh script take the LATEST
  available week's report/status as a stand-in for "this week's", which is
  post-kickoff information for a Week-1 replay (a real leakage risk flagged
  on review); the DEV-set accuracy above was already 100% without them, so
  they are dropped rather than patched for this narrow replay use.
  Rule fixed here, BEFORE being scored against 2026 Week 1 -- not tuned
  after seeing the 2026 result (feedback_no_fake_improvement rule 3).

CONFIDENCE: stamped CONFIRMED only when a depth-chart snapshot exists
strictly before that game's own kickoff. UNRESOLVED (not a guess) when it
does not.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
SCHEDULE_CSV = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"
OUT_PATH = DATA / "prospective" / "identity" / "SPORTS_NOVA_V19_WEEK1_QB_IDENTITY_RESOLUTION.json"

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
SEASON, WEEK = 2026, 1


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_parquet(url: str) -> tuple[pd.DataFrame, str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    return pd.read_parquet(io.BytesIO(raw)), sha256_bytes(raw)


def depth_chart_qb1_as_of(dc: pd.DataFrame, team: str, cutoff: pd.Timestamp):
    rows = dc[(dc.team == team) & (dc.pos_name == "Quarterback") & (dc.dt < cutoff)]
    if rows.empty:
        return None
    latest_dt = rows.dt.max()
    snap = rows[rows.dt == latest_dt].sort_values("pos_rank")
    qb1 = snap[snap.pos_rank == snap.pos_rank.min()].iloc[0]
    return {"PLAYER_ID": qb1.gsis_id, "PLAYER_NAME": qb1.player_name, "AS_OF_DT": str(latest_dt)}


def main():
    sched = pd.read_csv(SCHEDULE_CSV)
    week1 = sched[(sched.season == SEASON) & (sched.week == WEEK) & (sched.game_type == "REG")].copy()
    if len(week1) != 16:
        raise SystemExit(f"EXPECTED_16_WEEK1_GAMES_GOT_{len(week1)}")

    # Re-derive kickoff exactly as the capture scripts do (schedule gameday +
    # gametime, America/New_York -> UTC) so the temporal-firewall cutoff used
    # here matches the one the replay harness itself uses.
    from zoneinfo import ZoneInfo

    def parse_kickoff_utc(gameday, gametime):
        if pd.isna(gametime):
            return None
        d = datetime.strptime(str(gameday), "%Y-%m-%d")
        hh, mm = str(gametime).split(":")
        local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
        return local.astimezone(timezone.utc)

    dc_2025, dc_2025_sha = fetch_parquet(f"{NFLVERSE_RELEASES}/depth_charts/depth_charts_2025.parquet")
    dc_2026, dc_2026_sha = fetch_parquet(f"{NFLVERSE_RELEASES}/depth_charts/depth_charts_2026.parquet")
    dc = pd.concat([dc_2025, dc_2026], ignore_index=True)
    dc["dt"] = pd.to_datetime(dc["dt"], utc=True)

    resolutions = {}
    coverage = {"CONFIRMED": 0, "UNRESOLVED": 0}
    for r in week1.itertuples():
        away, home = r.away_team, r.home_team
        game_id = f"{SEASON}_{WEEK:02d}_{away}_{home}"
        kickoff = parse_kickoff_utc(r.gameday, r.gametime)
        if kickoff is None:
            continue
        for team in (home, away):
            dcq = depth_chart_qb1_as_of(dc, team, pd.Timestamp(kickoff))
            key = f"{game_id}__{team}"
            if dcq is None:
                resolutions[key] = {
                    "GAME_ID": game_id, "TEAM": team, "QB_ID": None, "QB_NAME": None,
                    "STATUS": "UNRESOLVED", "SOURCE": [], "AS_OF": None, "CONFIDENCE": "NONE",
                    "REASON": "no depth-chart snapshot strictly before kickoff for this team",
                }
                coverage["UNRESOLVED"] += 1
            else:
                resolutions[key] = {
                    "GAME_ID": game_id, "TEAM": team,
                    "QB_ID": dcq["PLAYER_ID"], "QB_NAME": dcq["PLAYER_NAME"],
                    "STATUS": "CONFIRMED", "CONFIDENCE": "HIGH",
                    "SOURCE": ["nflverse_depth_chart_timestamped"],
                    "AS_OF": dcq["AS_OF_DT"],
                    "SOURCE_TIMESTAMP": dcq["AS_OF_DT"],
                    "KICKOFF_UTC": kickoff.isoformat(),
                    "KICKOFF_TIME": kickoff.isoformat(),
                    "TEMPORAL_FIREWALL_PASS": dcq["AS_OF_DT"] < kickoff.isoformat(),
                }
                coverage["CONFIRMED"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "SCHEMA": "SPORTS_NOVA_V19_WEEK1_QB_IDENTITY_RESOLUTION",
        "MISSION": "SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER",
        "SEASON": SEASON, "WEEK": WEEK,
        "RULE": "DEPTH_CHART_QB1_ALONE_PRE_KICKOFF -- selected on 2025 Week1 DEV set "
                "(32/32 boundary team-games), see module docstring. Not agreement-gated "
                "(the agreement rule needs the broken panel-leader signal to concur).",
        "SOURCES": {
            "depth_charts_2025": {"sha256": dc_2025_sha},
            "depth_charts_2026": {"sha256": dc_2026_sha},
        },
        "RESOLVED_AT_UTC": datetime.now(timezone.utc).isoformat(),
        "COVERAGE": coverage,
        "RESOLUTIONS": resolutions,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    contract_path = ROOT / "SPORTS_V19_IDENTITY_CONTRACT.md"
    contract_lines = [
        "# SPORTS V19 identity contract\n\n",
        "This contract is the only semantic input permitted to differ from frozen V18. "
        "It is injected at the isolated V19 simulator boundary; V18 files remain unchanged.\n\n",
        "Required fields per `(game_id, team)` record:\n\n",
        "- `team` / `TEAM`\n- `game_id` / `GAME_ID`\n- `kickoff_time` / `KICKOFF_TIME`\n",
        "- `qb_player_id` / `QB_ID`\n- `confidence` / `CONFIDENCE`\n",
        "- `source` / `SOURCE`\n- `source_timestamp` / `SOURCE_TIMESTAMP`\n\n",
        "Acceptance rules: `STATUS=CONFIRMED`; confidence is `HIGH` or `CONFIRMED`; "
        "`SOURCE_TIMESTAMP < KICKOFF_TIME`; source provenance is persisted; uncertain "
        "records are `INVALID_FOR_REPLAY` and are not guessed.\n\n",
        "Rule: `DEPTH_CHART_QB1_ALONE_PRE_KICKOFF`, selected on the pre-2026 DEV boundary "
        "set before outcome joining.\n\n",
        f"Coverage: `{coverage['CONFIRMED']} CONFIRMED`, `{coverage['UNRESOLVED']} UNRESOLVED`.\n\n",
        "Source byte hashes:\n\n",
        f"- depth_charts_2025.parquet: `{dc_2025_sha}`\n",
        f"- depth_charts_2026.parquet: `{dc_2026_sha}`\n\n",
        "DEN@KC contract records:\n\n",
        f"- DEN: Bo Nix (`{resolutions['2026_01_DEN_KC__DEN']['QB_ID']}`), "
        f"source timestamp `{resolutions['2026_01_DEN_KC__DEN']['SOURCE_TIMESTAMP']}`, `HIGH`.\n",
        f"- KC: Patrick Mahomes (`{resolutions['2026_01_DEN_KC__KC']['QB_ID']}`), "
        f"source timestamp `{resolutions['2026_01_DEN_KC__KC']['SOURCE_TIMESTAMP']}`, `HIGH`.\n",
    ]
    contract_path.write_text("".join(contract_lines), encoding="utf-8")
    print(json.dumps({"CONFIRMED": coverage["CONFIRMED"], "UNRESOLVED": coverage["UNRESOLVED"],
                       "OUT": str(OUT_PATH.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
