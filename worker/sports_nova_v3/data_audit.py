"""Phase 01 source audit and checkpoint-safe validation protocol.

The audit deliberately reports availability evidence as absent when the local
artifact only has a status label. It never infers a publication lag from a
filesystem timestamp or kickoff time.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data" / "sports_nova_v1_preserved"
V2 = ROOT / "data" / "sports_nova_v2"
OUT = ROOT / "data" / "sports_nova_v3"
MARKET_TERMS = ("odds", "moneyline", "spread", "vig", "vegas", "sportsbook")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _schema(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(path).names)
    except Exception as exc:  # pragma: no cover - environment-dependent
        return [f"SCHEMA_UNAVAILABLE:{type(exc).__name__}"]

def _inventory(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "exists": path.is_file()}
    if not path.is_file():
        return item
    item.update({"bytes": path.stat().st_size, "sha256": sha256_file(path),
                 "schema": _schema(path) if path.suffix == ".parquet" else None})
    schema = item.get("schema") or []
    item["market_schema_terms"] = [c for c in schema if any(t in c.lower() for t in MARKET_TERMS)]
    return item

def audit_sources(root: Path = ROOT) -> dict[str, Any]:
    required = [
        V1 / "NFL_PLAYER_GAME_CAUSAL_2025_V1.parquet",
        V1 / "NFL_CANONICAL_SCHEDULE_SPINE_V1.parquet",
        V1 / "NFL_TEAM_ALIAS_MAP_V1.json",
        V2 / "pbp_agg" / "ingest_manifest.json",
        V2 / "artifacts" / "SPORTS_NOVA_V2_DATA_GAPS.json",
    ]
    ledgers = sorted((V2 / "pbp_agg").glob("team_game_ledger_*.parquet"))
    files = [_inventory(p) for p in required + ledgers]
    player = next((x for x in files if x["path"].endswith("NFL_PLAYER_GAME_CAUSAL_2025_V1.parquet")), {})
    schedule = next((x for x in files if x["path"].endswith("NFL_CANONICAL_SCHEDULE_SPINE_V1.parquet")), {})
    player_schema = set(player.get("schema") or [])
    schedule_schema = set(schedule.get("schema") or [])
    return {
        "artifact_id": "SPORTS_NOVA_V3_DATA_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_version": "SPORTS_NOVA_V3_ARCH_1",
        "python": platform.python_version(),
        "sources": files,
        "coverage": {
            "player_panel_event_time_status": "event_time_status" in player_schema,
            "player_panel_source_available_at": "m_source_available_at" in player_schema or "source_available_at" in player_schema,
            "player_panel_only_availability_status": "m_source_available_at_status" in player_schema and "source_available_at" not in player_schema,
            "schedule_source_available_at": "source_available_at" in schedule_schema,
            "schedule_source_available_at_status": "source_available_at_status" in schedule_schema,
            "team_game_ledger_count": len(ledgers),
            "drive_block_observations": False,
            "authoritative_final_score_label_in_ledger": False,
            "timestamped_roster_or_inactives": False,
        },
        "lineage": {
            "player_panel": "V1 preserved canonical join; kickoff ordering is not source-availability proof",
            "team_ledgers": "V2 ingest_pbp team-game aggregates; no drive/block-level observation archive verified",
            "market_data": "not used by V3 model construction; schema terms are audited",
        },
        "data_gaps": {
            "REQUIRED": [
                {"id": "R1", "status": "BLOCKS_STRICT_EMPIRICAL_CERTIFICATION", "reason": "historical exact revision availability timestamps absent"},
                {"id": "R2", "status": "NOT_VERIFIED", "reason": "local data is team-game aggregate, not authoritative drive/block ledger"},
                {"id": "R3", "status": "NOT_VERIFIED", "reason": "no causal pregame roster/inactive archive"},
                {"id": "R4", "status": "NOT_MEASURED", "reason": "prospective holdout and calibration support not established"},
            ],
            "NICE_TO_HAVE": ["timestamped injuries/inactives/depth charts", "snap counts", "forecast revisions", "longer role histories"],
            "PREMIUM_FUTURE": ["routes and coverage assignments", "tracking/separation", "OL/DL grades"],
        },
        "status": "PARTIAL_RESEARCH_GRADE",
    }

def validation_protocol() -> dict[str, Any]:
    return {
        "protocol_id": "SPORTS_NOVA_V3_WALK_FORWARD_PROTOCOL_1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_test_order": "TRAIN < DEV/CALIBRATION < OUTER_TEST; grouped by complete game",
        "minimum_support": {"marginal_event": 200, "joint_event": 200, "calibration": 400, "correlation_pair": 200},
        "metrics": ["MAE", "RMSE", "Brier", "LogLoss", "ECE", "CRPS", "interval_coverage", "joint_brier", "correlation_error"],
        "intervals": [0.5, 0.8, 0.9, 0.95],
        "ablation_ids": ["NO_SCRIPT", "NO_MATCHUP", "NO_RECENT_FORM", "NO_SHRINKAGE", "NO_PERSONNEL", "PERMUTED_DEPENDENCE", "NO_EPISTEMIC"],
        "holdout": "PROSPECTIVE_CAPTURE_REQUIRED; no current local prospective holdout asserted",
        "status": "FROZEN_FOR_IMPLEMENTATION; EMPIRICAL_CERTIFICATION_PARTIAL",
    }

def write_phase01(out: Path = OUT) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    audit_path, protocol_path = out / "data_audit.json", out / "validation_protocol.json"
    audit_path.write_text(json.dumps(audit_sources(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol_path.write_text(json.dumps(validation_protocol(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit_path, protocol_path

if __name__ == "__main__":
    paths = write_phase01()
    print(json.dumps({"written": [str(p) for p in paths], "status": "PARTIAL_RESEARCH_GRADE"}))
