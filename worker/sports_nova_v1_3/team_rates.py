"""Causal per-team sack-rate lookup, derived directly from
NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet -- the same file
scripts/sports_nova_v3_team_passing_volume_calibration_v15.py already
reads for REAL_SACKS. Uses the identical causal-cutoff discipline as
scripts/sports_nova_v3_player_joint_walkforward_v1.py (`_key < key`,
`season >= season - lookback`): only games strictly before the game being
simulated, within an up-to-5-season window, ever contribute.

LEAGUE_SACK_RATE_FALLBACK is training-derived with the 60-game
SPORTS_NOVA_V1_1 evaluation cohort's games excluded (same discipline as
the EXPERIMENT_001/002 constants) -- 0.0602941865..., a lower-bound
estimate (sacks / (pass_attempts + sacks); real scrambles are not
separable from designed rushes in this data source and are excluded from
both numerator and denominator, same limitation V15 already documents).
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_DRIVE_PATH = _ROOT / "data" / "sports_nova_v3" / "validation_inputs" / "NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet"

LEAGUE_SACK_RATE_FALLBACK = 0.060294186503005616
MIN_PASS_ATTEMPTS_FOR_TEAM_ESTIMATE = 200  # roughly one season's worth; below this, use the league fallback


@lru_cache(maxsize=1)
def _drive() -> pd.DataFrame:
    df = pd.read_parquet(_DRIVE_PATH, columns=["game_id", "season", "week", "offense_team", "pass_attempts", "sacks"])
    df["_key"] = df.season * 100 + df.week
    return df


def causal_sack_rate(team_id: str, season: int, week: int, *, lookback_seasons: int = 5) -> float:
    df = _drive()
    key = season * 100 + week
    window = df[(df.offense_team == team_id) & (df._key < key) & (df.season >= season - lookback_seasons)]
    pass_att = float(window.pass_attempts.sum())
    sacks = float(window.sacks.sum())
    if pass_att < MIN_PASS_ATTEMPTS_FOR_TEAM_ESTIMATE:
        return LEAGUE_SACK_RATE_FALLBACK
    return sacks / (pass_att + sacks)
