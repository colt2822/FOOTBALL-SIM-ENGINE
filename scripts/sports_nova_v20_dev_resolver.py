"""SPORTS_NOVA_V20 -- DEV-set QB1 resolver (2025 Week 1 boundary games).

Generalizes scripts/sports_nova_v19_qb_identity_resolver.py's already-DEV-
validated rule (DEPTH_CHART_QB1_ALONE_PRE_KICKOFF, 32/32 on 2025 Week1) to
produce the SAME resolutions for use as V20's DEV set -- this is not a new
rule, it is the existing V19 rule run on its own DEV season so the roster-
injection layer can be validated before touching 2026 Week1.

Ground truth for each team-slot is derived from the causal panel itself
(2025 Week1 rows already sit in NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet as
completed history for 2026 games), not a separate box-score fetch: actual
starter = the QB with the most PASS_ATTEMPTS in that (SEASON=2025, WEEK=1,
TEAM=tid) slice, same method scripts/sports_nova_v19_week1_replay_join.py
uses for 2026.

Never reads any row at/after a game's own kickoff for that game's own
resolution (temporal firewall, same as the 2026 resolver).
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
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
CACHE_DIR = DATA / "validation_inputs" / "depth_charts_external"
OUT_PATH = DATA / "SPORTS_NOVA_V20_DEV_2025_WEEK1_QB_RESOLUTION.json"

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
SEASON, WEEK = 2025, 1


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_or_cache_parquet(season: int) -> tuple[pd.DataFrame, str]:
    cache_path = CACHE_DIR / f"depth_charts_{season}.parquet"
    if cache_path.is_file():
        raw = cache_path.read_bytes()
    else:
        url = f"{NFLVERSE_RELEASES}/depth_charts/depth_charts_{season}.parquet"
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
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
    panel = pd.read_parquet(LIVE_PANEL)
    panel["_event_time"] = pd.to_datetime(panel["EVENT_TIME"], utc=True, errors="coerce")

    w1 = panel[(panel.SEASON == SEASON) & (panel.WEEK == WEEK)]
    games = sorted(w1.GAME_ID.unique())
    if len(games) != 16:
        raise SystemExit(f"EXPECTED_16_DEV_GAMES_GOT_{len(games)}")

    dc, dc_sha = fetch_or_cache_parquet(SEASON)
    dc["dt"] = pd.to_datetime(dc["dt"], utc=True)

    resolutions = {}
    ground_truth = {}
    coverage = {"CONFIRMED": 0, "UNRESOLVED": 0}
    for gid in games:
        gp = w1[w1.GAME_ID == gid]
        kickoff = gp._event_time.min()  # this game's own rows share one EVENT_TIME per the panel's construction
        parts = gid.split("_")
        away, home = parts[2], parts[3]
        for team in (home, away):
            key = f"{gid}__{team}"
            team_qb_rows = gp[(gp.TEAM == team) & (gp.POSITION == "QB")]
            if team_qb_rows.empty:
                continue
            actual = team_qb_rows.sort_values("PASS_ATTEMPTS", ascending=False).iloc[0]
            ground_truth[key] = {
                "GAME_ID": gid, "TEAM": team,
                "ACTUAL_STARTER_ID": str(actual.PLAYER_ID),
                "ACTUAL_STARTER_NAME": str(actual.PLAYER_NAME),
                "ACTUAL_PASS_ATTEMPTS": float(actual.PASS_ATTEMPTS),
            }
            dcq = depth_chart_qb1_as_of(dc, team, pd.Timestamp(kickoff))
            if dcq is None:
                resolutions[key] = {
                    "GAME_ID": gid, "TEAM": team, "QB_ID": None, "QB_NAME": None,
                    "STATUS": "UNRESOLVED", "SOURCE": [], "AS_OF": None, "CONFIDENCE": "NONE",
                }
                coverage["UNRESOLVED"] += 1
            else:
                resolutions[key] = {
                    "GAME_ID": gid, "TEAM": team,
                    "QB_ID": dcq["PLAYER_ID"], "QB_NAME": dcq["PLAYER_NAME"],
                    "STATUS": "CONFIRMED", "CONFIDENCE": "HIGH",
                    "SOURCE": ["nflverse_depth_chart_timestamped"],
                    "AS_OF": dcq["AS_OF_DT"], "SOURCE_TIMESTAMP": dcq["AS_OF_DT"],
                    "KICKOFF_UTC": str(kickoff), "KICKOFF_TIME": str(kickoff),
                    "TEMPORAL_FIREWALL_PASS": dcq["AS_OF_DT"] < str(kickoff),
                }
                coverage["CONFIRMED"] += 1

    payload = {
        "SCHEMA": "SPORTS_NOVA_V20_DEV_2025_WEEK1_QB_RESOLUTION",
        "SEASON": SEASON, "WEEK": WEEK,
        "RULE": "DEPTH_CHART_QB1_ALONE_PRE_KICKOFF (same rule as V19, reused unmodified on its own DEV season)",
        "SOURCES": {f"depth_charts_{SEASON}": {"sha256": dc_sha}},
        "GENERATED_AT_UTC": datetime.now(timezone.utc).isoformat(),
        "COVERAGE": coverage,
        "RESOLUTIONS": resolutions,
        "GROUND_TRUTH": ground_truth,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps({"N_TEAM_SLOTS": len(ground_truth), **coverage, "OUT": str(OUT_PATH.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
