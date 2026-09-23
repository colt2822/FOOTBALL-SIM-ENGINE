"""Shared paths and frozen constants for SPORTS-NOVA V2."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "data" / "sports_nova_v1_preserved"
V2_DATA = ROOT / "data" / "sports_nova_v2"
PBP_AGG = V2_DATA / "pbp_agg"
ARTIFACTS = V2_DATA / "artifacts"

CAUSAL_PARQUET = V1_DIR / "NFL_PLAYER_GAME_CAUSAL_2025_V1.parquet"
ALIAS_MAP = V1_DIR / "NFL_TEAM_ALIAS_MAP_V1.json"

TARGET = "passing_yards"
V1_FEATURES = [
    "trailing4_pass_yds",
    "season_td_pass_yds",
    "career_prior_pass_yds",
    "trailing4_attempts",
]

# V1 eligibility, reproduced verbatim.
MIN_PRIOR_GAMES = 4
OOS_START_SEASON = 2006
RIDGE_ALPHA = 10.0

# Chronology discipline: hyperparameters and model family are chosen on DEV
# only; TEST is scored once.
DEV_SEASONS = list(range(2006, 2020))
TEST_SEASONS = list(range(2020, 2026))

PROB_THRESHOLD = 250.0
THRESH_LIST_MAIN = [150, 175, 200, 225, 250, 275, 300, 325]
CURVE_LO, CURVE_HI, CURVE_STEP = 0.0, 450.0, 0.5
SEED = 20260908

# V1 published metrics, used as the reproduction gate.
V1_PUBLISHED = {
    "TRAIN_N_FINAL": 14666,
    "OOS_N": 11491,
    "MODEL_MAE": 66.67783156655865,
    "MODEL_RMSE": 83.91562418620893,
    "BRIER": 0.20884424358546963,
    "LOGLOSS": 0.6046717558971826,
    "ECE": 0.016487923470846287,
}
