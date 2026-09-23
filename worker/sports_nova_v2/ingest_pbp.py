"""Download nflverse play-by-play and reduce it to a team-game ledger.

Only football-derived fields are kept. Market-derived pbp columns (vegas_wp,
vegas_wpa, spread_line, total_line, ...) are never read.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "sports_nova_v2" / "pbp_agg"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "home_team", "away_team", "play_type", "pass", "rush", "down", "ydstogo",
    "yardline_100", "qtr", "game_seconds_remaining", "epa", "wp", "success",
    "yards_gained", "air_yards", "complete_pass", "sack", "qb_hit",
    "interception", "touchdown", "score_differential", "fixed_drive",
    "posteam_score", "penalty",
]

# Market-derived columns that must never enter the ledger.
FORBIDDEN = ("vegas", "spread_line", "total_line", "odds", "moneyline")


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den.replace(0, np.nan), np.nan)


def team_game_ledger(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].copy()
    p = p[p.posteam.notna() & p.defteam.notna()]
    p = p[p.play_type.isin(["pass", "run"])]

    p["is_pass"] = (p["pass"] == 1).astype(float)
    p["is_rush"] = (p["rush"] == 1).astype(float)
    p["neutral"] = ((p.wp.between(0.20, 0.80)) & (p.qtr <= 3)).astype(float)
    p["early"] = p.down.isin([1, 2]).astype(float)
    p["rz"] = (p.yardline_100 <= 20).astype(float)
    p["expl_pass"] = ((p.is_pass == 1) & (p.yards_gained >= 20)).astype(float)
    p["expl_rush"] = ((p.is_rush == 1) & (p.yards_gained >= 10)).astype(float)
    p["sack_f"] = p["sack"].fillna(0).astype(float)
    p["qb_hit_f"] = p["qb_hit"].fillna(0).astype(float)
    p["pass_epa"] = np.where(p.is_pass == 1, p.epa, np.nan)
    p["rush_epa"] = np.where(p.is_rush == 1, p.epa, np.nan)
    p["pass_yds"] = np.where(p.is_pass == 1, p.yards_gained, np.nan)
    p["neutral_pass"] = p.neutral * p.is_pass
    p["early_pass"] = p.early * p.is_pass
    p["rz_pass"] = p.rz * p.is_pass

    # Pace: seconds burned between consecutive snaps by the same offense in the
    # same drive. Clipped to drop halftime / review gaps.
    p = p.sort_values(["game_id", "fixed_drive", "game_seconds_remaining"],
                      ascending=[True, True, False])
    gap = p.groupby(["game_id", "fixed_drive"])["game_seconds_remaining"].diff(-1)
    p["sec_per_play"] = gap.clip(lower=0, upper=60)

    def agg(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
        g = df.groupby(["season", "week", "season_type", "game_id", key])
        out = pd.DataFrame({
            f"{prefix}plays": g.size(),
            f"{prefix}pass_plays": g.is_pass.sum(),
            f"{prefix}rush_plays": g.is_rush.sum(),
            f"{prefix}epa_per_play": g.epa.mean(),
            f"{prefix}pass_epa_per_db": g.pass_epa.mean(),
            f"{prefix}rush_epa_per_carry": g.rush_epa.mean(),
            f"{prefix}success_rate": g.success.mean(),
            f"{prefix}pass_yds": g.pass_yds.sum(),
            f"{prefix}yards": g.yards_gained.sum(),
            f"{prefix}adot": g.air_yards.mean(),
            f"{prefix}comp_rate": g.complete_pass.mean(),
            f"{prefix}sacks": g.sack_f.sum(),
            f"{prefix}qb_hits": g.qb_hit_f.sum(),
            f"{prefix}expl_pass": g.expl_pass.sum(),
            f"{prefix}expl_rush": g.expl_rush.sum(),
            f"{prefix}neutral_plays": g.neutral.sum(),
            f"{prefix}neutral_pass_plays": g.neutral_pass.sum(),
            f"{prefix}early_plays": g.early.sum(),
            f"{prefix}early_pass_plays": g.early_pass.sum(),
            f"{prefix}rz_plays": g.rz.sum(),
            f"{prefix}rz_pass_plays": g.rz_pass.sum(),
            f"{prefix}sec_per_play": g.sec_per_play.mean(),
            f"{prefix}points": g.posteam_score.max(),
        }).reset_index().rename(columns={key: "team"})
        return out

    off = agg(p, "posteam", "off_")
    dfn = agg(p, "defteam", "def_").drop(columns=["def_points"])
    led = off.merge(dfn, on=["season", "week", "season_type", "game_id", "team"], how="outer")

    for side in ("off_", "def_"):
        led[f"{side}pass_rate"] = _rate(led[f"{side}pass_plays"], led[f"{side}plays"])
        led[f"{side}neutral_pass_rate"] = _rate(led[f"{side}neutral_pass_plays"], led[f"{side}neutral_plays"])
        led[f"{side}early_pass_rate"] = _rate(led[f"{side}early_pass_plays"], led[f"{side}early_plays"])
        led[f"{side}rz_pass_rate"] = _rate(led[f"{side}rz_pass_plays"], led[f"{side}rz_plays"])
        led[f"{side}sack_rate"] = _rate(led[f"{side}sacks"], led[f"{side}pass_plays"])
        led[f"{side}qb_hit_rate"] = _rate(led[f"{side}qb_hits"], led[f"{side}pass_plays"])
        led[f"{side}expl_pass_rate"] = _rate(led[f"{side}expl_pass"], led[f"{side}pass_plays"])
        led[f"{side}expl_rush_rate"] = _rate(led[f"{side}expl_rush"], led[f"{side}rush_plays"])
        led[f"{side}yds_per_pass_play"] = _rate(led[f"{side}pass_yds"], led[f"{side}pass_plays"])
    return led


def run(seasons: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for s in seasons:
        dest = OUT / f"team_game_ledger_{s}.parquet"
        if dest.exists():
            print(f"skip {s} (exists)", flush=True)
            continue
        raw = urllib.request.urlopen(URL.format(season=s), timeout=300).read()
        pbp = pd.read_parquet(io.BytesIO(raw), columns=COLS)
        bad = [c for c in pbp.columns if any(t in c.lower() for t in FORBIDDEN)]
        assert not bad, f"market column leaked: {bad}"
        led = team_game_ledger(pbp)
        led.to_parquet(dest, index=False)
        manifest[str(s)] = {"pbp_rows": int(len(pbp)), "ledger_rows": int(len(led))}
        print(f"{s}: pbp={len(pbp)} ledger={len(led)}", flush=True)
    (OUT / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1999, 2025)
    run(list(range(lo, hi + 1)))
