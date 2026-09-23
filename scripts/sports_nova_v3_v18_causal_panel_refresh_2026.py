"""SPORTS_NOVA_V18_2026_CAUSAL_PANEL_REFRESH.

Incrementally appends completed-2026-week rows to a NEW copy of the V3
player-game causal panel, so V18's frozen feature code (which reads
NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet by SEASON*100+WEEK) can compute
recency/volume/shares features from real 2026 games instead of stopping at
end-of-2025.

Hard rule, verified against SPORTS_NOVA_V18_FREEZE_MANIFEST.json before this
script was written: prospective_capture.py hash-checks the FROZEN panel path
(data/sports_nova_v3/validation_inputs/NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet)
byte-for-byte against DATA_SCHEMA_HASHES at every run and hard-exits on any
drift. That frozen file is therefore never opened for writing here -- this
script only READS it, and writes the refreshed panel to a sibling directory
(validation_inputs_live/) under a different filename. A separate capture
script (sports_nova_v3_v18_prospective_capture_live.py) points at the new
path explicitly; the original capture script and the freeze hash it checks
are completely unaffected by this refresh.

Whole-week granularity, not per-game: the frozen walkforward's
build_prior_snapshot() keys strictly on SEASON*100+WEEK ("_key < target_key"),
with no within-week ordering. Appending a partial week would let one game in
a week look "prior" to another game in the SAME week the moment any row from
that week entered the panel with a lower row-index -- the frozen code has no
per-game timestamp comparison to prevent that. So this script only ingests
2026 weeks where every scheduled game in that week already has a final score.
Source for "final": nflverse/nfldata games.csv home_score/away_score
presence, same schedule feed the prospective-capture script already uses.

Source for the new rows: nflverse-data release tag "stats_player", asset
stats_player_week_<season>.parquet -- identical dataset family to the
existing panel's recorded SOURCE ("nflverse/nflverse-data:stats_player_week"
/ "release tag stats_player"), just a later snapshot of it.

CANONICAL_GAME_ID is recomputed with the exact formula recovered from
D:\\build_nfl_schedule_spine.py (canonical_id(); nflg_ + sha256(...)[:24] over
"|".join([season, game_type, week, home_team, away_team, kickoff_iso_Z])) and
verified byte-for-byte against a known frozen row (2025_22_SEA_NE ->
nflg_55b023e05fba19013ffa728a) before being trusted for new rows.

Columns NOT populated for new rows (SNAPS, ROUTE_PARTICIPATION,
STARTER_STATUS, ACTIVE_STATUS, ROSTER_STATUS): grep-verified that
make_state()/team_state()/shares() in the frozen
scripts/sports_nova_v3_player_joint_walkforward_v1.py never reference these
columns (STARTER_STATUS is already 100% null across all 475,586 existing
rows per SPORTS_NOVA_V12_QB_IDENTITY_SIGNAL_AVAILABILITY.json). Left null
here rather than fabricated; disclosed in the provenance file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
FROZEN_PANEL = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_DIR = DATA / "validation_inputs_live"
LIVE_PANEL = LIVE_DIR / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
RECEIPT_PATH = LIVE_DIR / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"
FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"

GAMES_CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
STATS_PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)

PANEL_COLUMNS = [
    "GAME_ID", "CANONICAL_GAME_ID", "SEASON", "WEEK", "GAME_TYPE", "EVENT_TIME",
    "PLAYER_ID", "PLAYER_NAME", "POSITION", "POSITION_GROUP", "TEAM", "OPPONENT",
    "PASS_ATTEMPTS", "COMPLETIONS", "PASS_YARDS", "PASS_TD", "INTERCEPTIONS",
    "RUSH_ATTEMPTS", "RUSH_YARDS", "RUSH_TD", "TARGETS", "RECEPTIONS",
    "RECEIVING_YARDS", "RECEIVING_TD", "SNAPS", "ROUTE_PARTICIPATION",
    "STARTER_STATUS", "ACTIVE_STATUS", "ROSTER_STATUS", "SOURCE",
    "SOURCE_VERSION", "SOURCE_PROVENANCE", "CAUSALITY_MODE", "TARGET_DATA",
    "PREGAME_FEATURE_DATA",
]

INT_COLS = [
    "PASS_ATTEMPTS", "COMPLETIONS", "PASS_YARDS", "PASS_TD", "INTERCEPTIONS",
    "RUSH_ATTEMPTS", "RUSH_YARDS", "RUSH_TD", "TARGETS", "RECEPTIONS",
    "RECEIVING_YARDS", "RECEIVING_TD",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sports-nova-v18-refresh/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def iso_z(v: datetime) -> str:
    return v.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_id(parts: list[str]) -> str:
    material = "|".join(parts)
    return "nflg_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def verify_canonical_id_formula(frozen_df: pd.DataFrame) -> None:
    """Fail loudly if the recovered canonical_id() formula stops matching."""
    row = frozen_df[frozen_df.GAME_ID == "2025_22_SEA_NE"].iloc[0]
    expected = row["CANONICAL_GAME_ID"]
    kickoff = row["EVENT_TIME"]
    got = canonical_id(["2025", "SB", "22", "NE", "SEA", kickoff])
    if got != expected:
        raise SystemExit(f"CANONICAL_ID_FORMULA_DRIFT: expected {expected} got {got}")


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    local = datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--cutoff-utc", type=str, default=None,
                     help="ISO8601 UTC; only weeks fully complete AND fully "
                          "before this timestamp are ingested. Default: no "
                          "extra cutoff beyond 'all games in the week have a "
                          "final score'.")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = datetime.fromisoformat(args.cutoff_utc.replace("Z", "+00:00")) if args.cutoff_utc else now

    if not FROZEN_PANEL.exists():
        raise SystemExit(f"MISSING_FROZEN_PANEL: {FROZEN_PANEL}")
    frozen_sha_before = sha256_file(FROZEN_PANEL)
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    expected_frozen_sha = freeze["DATA_SCHEMA_HASHES"]["NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"]
    if frozen_sha_before != expected_frozen_sha:
        raise SystemExit(
            f"FROZEN_PANEL_ALREADY_DRIFTED (refuse to build on top of it): "
            f"expected {expected_frozen_sha} got {frozen_sha_before}")

    frozen_df = pd.read_parquet(FROZEN_PANEL)
    verify_canonical_id_formula(frozen_df)
    existing_keys = set(zip(frozen_df.GAME_ID, frozen_df.PLAYER_ID))

    games_raw = fetch(GAMES_CSV_URL)
    games_sha = sha256_bytes(games_raw)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / f"schedule_snapshot_{args.season}.csv").write_bytes(games_raw)
    games = pd.read_csv(LIVE_DIR / f"schedule_snapshot_{args.season}.csv")
    season_games = games[games.season == args.season].copy()
    season_games["is_final"] = season_games.home_score.notna() & season_games.away_score.notna()
    season_games["kickoff_dt"] = season_games.apply(
        lambda r: kickoff_utc(r.gameday, r.gametime) if pd.notna(r.gametime) else None, axis=1)

    week_complete = season_games.groupby("week")["is_final"].all()
    week_max_kickoff = season_games.groupby("week")["kickoff_dt"].max()
    eligible_weeks = sorted(
        w for w in week_complete.index
        if week_complete[w] and week_max_kickoff[w] is not None and week_max_kickoff[w] < cutoff
    )

    stats_raw = fetch(STATS_PLAYER_WEEK_URL.format(season=args.season))
    stats_sha = sha256_bytes(stats_raw)
    stats_path = LIVE_DIR / f"stats_player_week_{args.season}_snapshot.parquet"
    stats_path.write_bytes(stats_raw)
    stats = pd.read_parquet(stats_path)

    stats = stats[stats.week.isin(eligible_weeks)].copy()

    sched_by_gid = season_games.set_index("game_id")
    stats = stats[stats.game_id.isin(sched_by_gid.index)].copy()

    def build_row(r):
        g = sched_by_gid.loc[r.game_id]
        kickoff = g["kickoff_dt"]
        event_time = iso_z(kickoff)
        game_type_raw = str(g["game_type"])
        panel_game_type = "REG" if game_type_raw == "REG" else "POST"
        cgid = canonical_id([
            str(int(g["season"])), game_type_raw, str(int(g["week"])),
            g["home_team"], g["away_team"], event_time,
        ])
        return pd.Series({
            "GAME_ID": r.game_id,
            "CANONICAL_GAME_ID": cgid,
            "SEASON": int(r.season),
            "WEEK": int(r.week),
            "GAME_TYPE": panel_game_type,
            "EVENT_TIME": event_time,
            "PLAYER_ID": r.player_id,
            "PLAYER_NAME": r.player_display_name,
            "POSITION": r.position,
            "POSITION_GROUP": r.position_group,
            "TEAM": r.team,
            "OPPONENT": r.opponent_team,
            "PASS_ATTEMPTS": int(r.attempts or 0),
            "COMPLETIONS": int(r.completions or 0),
            "PASS_YARDS": int(r.passing_yards or 0),
            "PASS_TD": int(r.passing_tds or 0),
            "INTERCEPTIONS": int(r.passing_interceptions or 0),
            "RUSH_ATTEMPTS": int(r.carries or 0),
            "RUSH_YARDS": int(r.rushing_yards or 0),
            "RUSH_TD": int(r.rushing_tds or 0),
            "TARGETS": int(r.targets or 0),
            "RECEPTIONS": int(r.receptions or 0),
            "RECEIVING_YARDS": int(r.receiving_yards or 0),
            "RECEIVING_TD": int(r.receiving_tds or 0),
            "SNAPS": None,
            "ROUTE_PARTICIPATION": None,
            "STARTER_STATUS": None,
            "ACTIVE_STATUS": None,
            "ROSTER_STATUS": None,
            "SOURCE": "nflverse/nflverse-data:stats_player_week",
            "SOURCE_VERSION": "release tag stats_player",
            "SOURCE_PROVENANCE": "weekly player statistics; outcome rows; event-causal only; V18_2026_CAUSAL_PANEL_REFRESH",
            "CAUSALITY_MODE": "EVENT_CAUSAL_ONLY",
            "TARGET_DATA": True,
            "PREGAME_FEATURE_DATA": False,
        })

    new_rows = stats.apply(build_row, axis=1) if len(stats) else pd.DataFrame(columns=PANEL_COLUMNS)
    if len(new_rows):
        new_rows = new_rows[PANEL_COLUMNS]
        # Match the frozen panel's exact per-column pandas/arrow dtypes so a
        # plain concat doesn't silently upcast (e.g. pyarrow string -> object
        # on the all-null STARTER_STATUS column), which would make the
        # historical-rows-unmutated equality check below fail on dtype alone
        # even though no value changed.
        for c in PANEL_COLUMNS:
            new_rows[c] = new_rows[c].astype(frozen_df[c].dtype)

    new_keys = list(zip(new_rows.GAME_ID, new_rows.PLAYER_ID)) if len(new_rows) else []
    collisions = [k for k in new_keys if k in existing_keys]
    dup_within_new = len(new_keys) - len(set(new_keys))
    if collisions:
        raise SystemExit(f"KEY_COLLISION_WITH_FROZEN_PANEL: {collisions[:5]} (+{len(collisions)-5} more)")
    if dup_within_new:
        raise SystemExit(f"DUPLICATE_KEYS_WITHIN_NEW_ROWS: {dup_within_new}")

    future_leak = new_rows[pd.to_datetime(new_rows.EVENT_TIME) >= cutoff] if len(new_rows) else new_rows
    if len(future_leak):
        raise SystemExit(f"FUTURE_ROW_LEAK: {len(future_leak)} rows at/after cutoff {cutoff.isoformat()}")

    combined = pd.concat([frozen_df, new_rows], ignore_index=True) if len(new_rows) else frozen_df.copy()

    hist_slice = combined.iloc[:len(frozen_df)].reset_index(drop=True)
    frozen_reset = frozen_df.reset_index(drop=True)
    if not hist_slice.equals(frozen_reset):
        diff_cols = [c for c in frozen_reset.columns if not hist_slice[c].equals(frozen_reset[c])]
        raise SystemExit(f"HISTORICAL_ROWS_MUTATED: columns differ {diff_cols}")

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, LIVE_PANEL, compression="zstd")
    live_sha_after = sha256_file(LIVE_PANEL)

    frozen_sha_after = sha256_file(FROZEN_PANEL)
    if frozen_sha_after != frozen_sha_before:
        raise SystemExit("FROZEN_PANEL_WAS_MODIFIED_DURING_THIS_RUN -- aborting, investigate immediately")

    receipt = {
        "SCHEMA": "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT",
        "BUILT_AT_UTC": now.isoformat(),
        "SEASON": args.season,
        "CUTOFF_UTC": cutoff.isoformat(),
        "FROZEN_PANEL_PATH": str(FROZEN_PANEL.relative_to(ROOT)),
        "FROZEN_PANEL_SHA256": frozen_sha_before,
        "FROZEN_PANEL_UNCHANGED": frozen_sha_after == frozen_sha_before,
        "LIVE_PANEL_PATH": str(LIVE_PANEL.relative_to(ROOT)),
        "LIVE_PANEL_SHA256": live_sha_after,
        "ROWS_FROZEN": int(len(frozen_df)),
        "ROWS_ADDED": int(len(new_rows)),
        "ROWS_LIVE_TOTAL": int(len(combined)),
        "WEEKS_ELIGIBLE_THIS_RUN": eligible_weeks,
        "WEEKS_SKIPPED_INCOMPLETE": sorted(w for w in week_complete.index if not week_complete[w]),
        "MAX_2026_EVENT_TIME": new_rows.EVENT_TIME.max() if len(new_rows) else None,
        "GAMES_ADDED": sorted(new_rows.GAME_ID.unique().tolist()) if len(new_rows) else [],
        "SOURCES": {
            "schedule": {"url": GAMES_CSV_URL, "fetched_at_utc": now.isoformat(), "sha256": games_sha},
            "stats_player_week": {
                "url": STATS_PLAYER_WEEK_URL.format(season=args.season),
                "fetched_at_utc": now.isoformat(), "sha256": stats_sha,
            },
        },
        "COLUMNS_NOT_POPULATED_FOR_NEW_ROWS": [
            "SNAPS", "ROUTE_PARTICIPATION", "STARTER_STATUS", "ACTIVE_STATUS", "ROSTER_STATUS"],
        "COLUMNS_NOT_POPULATED_REASON": (
            "grep-verified unused by make_state/team_state/shares in the frozen "
            "walkforward script; STARTER_STATUS is already 100% null across all "
            "existing rows. Left null rather than fabricated."),
        "CANONICAL_ID_FORMULA_VERIFIED_AGAINST": "2025_22_SEA_NE -> nflg_55b023e05fba19013ffa728a",
        "DEDUP_CHECK": "0 collisions with frozen (GAME_ID,PLAYER_ID) keys; 0 duplicate keys within new rows",
        "HISTORICAL_ROWS_MUTATED": False,
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
