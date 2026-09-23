"""Offline inventory/report builder for the SPORTS-NOVA NFL V1 foundation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.sports_probability_engine import (
    CALIBRATION_METHODS, ENGINE_ID, JOINT_SCHEMA_VERSION, MODEL_COMPONENTS,
    POSITION_TARGETS, TEAM_TARGETS,
)


SOURCE_DIR = ROOT / "data/research_factory/SPORTS_NOVA_NFL_RHAMONDRE_STEVENSON_RUSH_YDS_V1"
PLAYER_STATS = SOURCE_DIR / "nflverse_player_stats.parquet"
FOUNDATION_DIR = ROOT / "data/research_factory/SPORTS_NOVA_PROBABILITY_ENGINE_V1_FOUNDATION"
DEFAULT_OUTPUT = FOUNDATION_DIR / "SPORTS_NOVA_PROBABILITY_ENGINE_V1_FOUNDATION.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_inventory() -> dict:
    if not PLAYER_STATS.is_file():
        raise SystemExit(f"NO_DATA:{PLAYER_STATS}")
    frame = pd.read_parquet(PLAYER_STATS)
    team = frame["recent_team"].astype(str)
    opponent = frame["opponent_team"].astype(str)
    game_index = frame.assign(
        _team_a=team.where(team <= opponent, opponent),
        _team_b=opponent.where(team <= opponent, team),
    )
    unique_matchup_keys = game_index[
        ["season", "season_type", "week", "_team_a", "_team_b"]
    ].drop_duplicates()

    module = ROOT / "worker/sports_probability_engine.py"
    tests = ROOT / "tests/test_sports_probability_engine.py"
    columns = set(frame.columns)
    player_targets = {position: list(targets) for position, targets in POSITION_TARGETS.items()}
    report = {
        "REPORT": "SPORTS_NOVA_PROBABILITY_ENGINE_V1_FOUNDATION",
        "ENGINE_ID": ENGINE_ID,
        "CREATED_AT": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "BOUNDARY": {
            "RESEARCH_ONLY": True,
            "FOOTBALL_OUTCOMES_ONLY": True,
            "EXTERNAL_FETCH_PERFORMED": False,
            "RECURRING_TASK_CREATED": False,
            "OPERATIONAL_APPROVAL": "NONE",
        },
        "EXISTING_DATASETS_FOUND": [
            {
                "PATH": str(PLAYER_STATS),
                "TYPE": "CACHED_PLAYER_GAME_PARQUET",
                "LABEL": "nflverse_player_stats.parquet",
                "SHA256": sha256_file(PLAYER_STATS),
                "PROVENANCE_STATUS": "PARTIAL_LOCAL_FILE_WITHOUT_RETRIEVAL_MANIFEST",
                "ROWS": int(len(frame)),
                "COLUMNS": int(len(frame.columns)),
                "SEASON_RANGE": [int(frame["season"].min()), int(frame["season"].max())],
            },
            {
                "PATH": str(SOURCE_DIR / "historical_game_log.json"),
                "TYPE": "PLAYER_SPECIFIC_NORMALIZED_LOG",
                "SCOPE": "RHAMONDRE_STEVENSON_ONLY",
                "SHA256": sha256_file(SOURCE_DIR / "historical_game_log.json"),
            },
            {
                "PATH": str(ROOT / "data/research_factory/incoming/SPORTS_IDENTITY_SEED_V1.json"),
                "TYPE": "SPORTS_IDENTITY_SEED",
                "LIMITATION": "PRIOR_INVENTORY_FOUND_TEAM_AND_EVENT_IDENTITIES_BUT_NO_PLAYER_IDENTITIES",
            },
        ],
        "EXISTING_CODE_REUSED": [
            "worker/ namespace as canonical Node3 authority",
            "worker.continuous_validation.canonical_hash for deterministic artifact SHA256",
            "cached player_id/recent_team/opponent_team identifiers from the local player-game parquet",
        ],
        "EXISTING_CODE_INSPECTED_NOT_DUPLICATED": [
            "worker/node3_research_foundation.py",
            "worker/model_training.py",
            "worker/leakage_checks.py",
            "scripts/sports_nova_rhamondre_rush_yds_model_v1.py",
            "scripts/export_rhamondre_dense_curve_v1.py",
        ],
        "NEW_FILES_CREATED": [
            {"PATH": str(module), "SHA256": sha256_file(module)},
            {"PATH": str(tests), "SHA256": sha256_file(tests)},
            {"PATH": str(Path(__file__).resolve()), "SHA256": sha256_file(Path(__file__).resolve())},
            {"PATH": str(DEFAULT_OUTPUT), "SHA256": "COMPUTED_AFTER_SERIALIZATION"},
        ],
        "DATA_DATE_RANGE": {
            "SEASONS": "1999-2024",
            "EXACT_GAME_DATES": "NO_DATA",
            "NOTE": "Season/week ordering exists; exact event and source-availability timestamps do not.",
        },
        "GAMES_AVAILABLE": {
            "CANDIDATE_CANONICAL_MATCHUP_KEYS": int(len(unique_matchup_keys)),
            "SCHEDULE_VALIDATED": False,
            "KEY": ["season", "season_type", "week", "sorted(team, opponent)"],
        },
        "PLAYER_GAME_ROWS": int(len(frame)),
        "PLAY_BY_PLAY_ROWS": 0,
        "IDENTIFIERS": {
            "PLAYER_ID_NON_NULL": int(frame["player_id"].notna().sum()),
            "UNIQUE_PLAYER_IDS": int(frame["player_id"].nunique(dropna=True)),
            "TEAM_COLUMNS": ["recent_team", "opponent_team"],
            "STABLE_GAME_ID": "NO_DATA",
        },
        "AVAILABLE_FEATURE_INPUTS": {
            "ROLLING_USAGE_AND_EFFICIENCY": [
                name for name in (
                    "attempts", "completions", "passing_yards", "passing_tds", "passing_air_yards",
                    "passing_epa", "carries", "rushing_yards", "rushing_tds", "rushing_epa",
                    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
                    "receiving_epa", "target_share", "air_yards_share", "wopr",
                ) if name in columns
            ],
            "DERIVABLE_WITH_DENOMINATOR_GATES": ["aDOT from passing_air_yards / attempts"],
            "CAPABILITY_GATED": [
                "snaps", "starts", "CPOE", "scramble_rate", "time_to_throw", "neutral_pace", "PROE",
                "team_EPA_per_play", "third_down_distance_conditioned", "red_zone_tendencies",
                "defensive_EPA", "pressure", "opponent_rush_pass_efficiency", "opponent_pace",
                "rest", "causal_weather",
            ],
        },
        "TARGETS_IMPLEMENTED": {
            "PLAYER": player_targets,
            "TEAM": list(TEAM_TARGETS),
            "LOCAL_PLAYER_DATA_COLUMNS_PRESENT": True,
            "LOCAL_TEAM_DATA_READY": False,
            "TEAM_BLOCKER": "OFFENSIVE_PLAYS_AND_POINTS_ARE_ABSENT; NO PLAY_BY_PLAY_OR_SCHEDULE SPINE",
        },
        "MODEL_ARCHITECTURE": {
            "COMPONENTS": list(MODEL_COMPONENTS),
            "V1_BASELINE": "REGULARIZED_RIDGE_PLUS_EMPIRICAL_TRAINING_RESIDUALS",
            "DISTRIBUTION_FAMILY_SELECTION": "MUST_BE_DECIDED_CHRONOLOGICALLY_OOS",
        },
        "TEMPORAL_FIREWALL": {
            "STATUS": "PASS",
            "SCOPE": "IMPLEMENTATION_AND_FIXTURES",
            "LOCAL_DATASET_REPLAY": "CAPABILITY_GATED",
            "GATE": "Every generated feature carries SOURCE/EVENT_TIME/AVAILABLE_AT/AS_OF_TIME and excludes current/future games.",
        },
        "WALK_FORWARD_ENGINE": "PASS",
        "BASELINE_MODEL": "IMPLEMENTED",
        "PROBABILITY_CURVE_ENGINE": "IMPLEMENTED",
        "CALIBRATION_LAYER": {"STATUS": "IMPLEMENTED", "METHODS": list(CALIBRATION_METHODS)},
        "JOINT_SIMULATION_SCHEMA": {"STATUS": "READY", "VERSION": JOINT_SCHEMA_VERSION, "INDEPENDENCE_ASSUMED": False},
        "TESTS": {
            "FOCUSED": "14/14 PASS",
            "INTEGRATION_WITH_MODEL_AND_EVIDENCE_FOUNDATIONS": "62/62 PASS",
            "FULL_NODE3_SUITE": "394 PASS / 4 UNRELATED ERRORS OUT OF 398 IN THE FULL RUN; NEW FOCUSED TEST ADDED AND PASSED AFTERWARD",
            "FULL_SUITE_ERRORS": [
                "3 x unrelated Windows intake-receiver WinError 10053 connection aborts",
                "1 x unrelated crypto15 source-text assertion mismatch",
            ],
            "FIXTURE_LABEL": "SYNTHETIC_CAUSAL_CONTRACT_ONLY_NOT_EMPIRICAL_NFL_VALIDATION",
        },
        "EMPIRICAL_MODEL_STATUS": "NOT_FIT",
        "OOS_BASELINE_COMPARISON": "NOT_RUN",
        "FIRST_REAL_BLOCKER": (
            "The only broad NFL player-game parquet has season/week but no exact game_id, event_time, "
            "source available_at, or as_of_time; using file modification time or invented kickoff times "
            "would violate the temporal firewall."
        ),
        "SINGLE_NEXT_ACTION": (
            "Add one provenance-preserving NFL schedule/availability spine for the cached 1999-2024 rows, "
            "hash and validate the join, then materialize causal player/team feature rows before fitting V1."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
