"""Node4-readable, dependency-aware joint probability exports.

This module is deliberately separate from the frozen V2 artifacts.  A ticket
is evaluated from per-path simulation outcomes; the marginal product is kept
only as a diagnostic.  Missing player outcomes or unsupported markets produce
``UNRESOLVED_DEPENDENCY`` and never an independence fallback.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

JOINT_SCHEMA_VERSION = "sports_nova_v2_joint_export.v1"
SAMPLE_SCHEMA_VERSION = "sports_nova_v2_simulation_samples.v1"

_MARKET_TARGETS = {
    "QB_PASS_YARDS": "QB",
    "LEAD_WR_REC_YARDS": "WR_TE",
    "LEAD_RB_RUSH_YARDS": "RB",
    "OPPOSING_QB_PASS_YARDS": "QB",
}
_FORBIDDEN_TERMS = ("odds", "sportsbook", "moneyline", "spread", "vig", "vegas")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_market_data(value: Any) -> None:
    text = json.dumps(value, sort_keys=True, default=str).lower()
    bad = [term for term in _FORBIDDEN_TERMS if term in text]
    if bad:
        raise ValueError(f"market data is forbidden in Node3 joint export: {bad}")


def _normalise_leg(leg: Mapping[str, Any]) -> dict[str, Any]:
    required = ("player", "market", "side", "line")
    missing = [key for key in required if key not in leg]
    if missing:
        raise ValueError(f"ticket leg missing required fields: {missing}")
    market = str(leg["market"])
    side = str(leg["side"]).lower()
    if side not in {"over", "under"}:
        raise ValueError(f"unsupported side: {side}")
    line = float(leg["line"])
    if not math.isfinite(line):
        raise ValueError("ticket line must be finite")
    return {"player": str(leg["player"]), "market": market,
            "side": side, "line": line}


def _ticket_legs(ticket: Mapping[str, Any]) -> list[dict[str, Any]]:
    _reject_market_data(ticket)
    raw = ticket.get("legs")
    if raw is None:
        raw = [value for key, value in ticket.items()
               if str(key).upper().startswith("LEG_")]
    if not raw:
        raise ValueError("ticket must contain at least one leg")
    return [_normalise_leg(x) for x in raw]


def _draw_for_leg(simulation: Mapping[str, Any], leg: Mapping[str, Any]) -> np.ndarray | None:
    market = leg["market"]
    if market not in _MARKET_TARGETS:
        return None
    draws = simulation.get("draws", {})
    player = leg["player"]
    if player not in draws:
        return None
    arr = np.asarray(draws[player], dtype=float)
    if arr.ndim != 1 or len(arr) != int(simulation.get("n_paths", len(arr))):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _bounds(p: float, n: int) -> tuple[float, float, float]:
    """Approximate binomial 90% interval, clipped to [0, 1]."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return (max(0.0, p - 1.645 * se), p, min(1.0, p + 1.645 * se))


def export_joint_probability(
    ticket: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    model_id: str,
    model_version: str,
    model_sha256: str,
    simulation_id: str,
    simulation_seed: int,
) -> dict[str, Any]:
    """Return one stable Node4 ticket result from correlated path outcomes."""
    legs = _ticket_legs(ticket)
    n_paths = int(simulation.get("n_paths", 0))
    if n_paths <= 0:
        raise ValueError("simulation must contain positive n_paths")

    hit_masks: list[np.ndarray] = []
    marginals: list[dict[str, Any]] = []
    unresolved = []
    for leg in legs:
        draw = _draw_for_leg(simulation, leg)
        if draw is None:
            unresolved.append(leg)
            continue
        hit = draw > leg["line"] if leg["side"] == "over" else draw < leg["line"]
        hit_masks.append(hit)
        marginals.append({**leg, "probability": float(hit.mean()),
                          "model_status": simulation.get("marginal_model_status", {}).get(
                              leg["market"], "UNVALIDATED_PLACEHOLDER")})

    if unresolved:
        return {
            "SCHEMA_VERSION": JOINT_SCHEMA_VERSION,
            "GAME_ID": ticket.get("game_id", ticket.get("GAME_ID")),
            "MODEL_ID": model_id, "MODEL_VERSION": model_version,
            "MODEL_SHA256": model_sha256,
            "SIMULATION_ID": simulation_id,
            "SIMULATION_SEED": int(simulation_seed),
            "SIMULATION_PATH_COUNT": n_paths,
            "LEG_MARGINAL_PROBABILITIES": marginals,
            "MARGINAL_PRODUCT_PROBABILITY": None,
            "JOINT_MODEL_PROBABILITY": None,
            "DEPENDENCY_DELTA": None,
            "JOINT_STANDARD_ERROR": None,
            "JOINT_P05": None, "JOINT_P50": None, "JOINT_P95": None,
            "DEPENDENCY_STATUS": "UNRESOLVED_DEPENDENCY",
            "UNRESOLVED_LEGS": unresolved,
            "MARGINAL_MODEL_STATUS": simulation.get("marginal_model_status", {}),
        }

    mask = np.logical_and.reduce(hit_masks)
    marginal_product = float(np.prod([x["probability"] for x in marginals]))
    joint = float(mask.mean())
    se = math.sqrt(max(joint * (1.0 - joint), 0.0) / n_paths)
    p05, p50, p95 = _bounds(joint, n_paths)
    return {
        "SCHEMA_VERSION": JOINT_SCHEMA_VERSION,
        "GAME_ID": ticket.get("game_id", ticket.get("GAME_ID")),
        "LEGS": legs,
        "MODEL_ID": model_id, "MODEL_VERSION": model_version,
        "MODEL_SHA256": model_sha256,
        "SIMULATION_ID": simulation_id,
        "SIMULATION_SEED": int(simulation_seed),
        "SIMULATION_PATH_COUNT": n_paths,
        "LEG_MARGINAL_PROBABILITIES": marginals,
        "MARGINAL_PRODUCT_PROBABILITY": marginal_product,
        "JOINT_MODEL_PROBABILITY": joint,
        "DEPENDENCY_DELTA": joint - marginal_product,
        "JOINT_STANDARD_ERROR": se,
        "JOINT_P05": p05, "JOINT_P50": p50, "JOINT_P95": p95,
        "DEPENDENCY_STATUS": "SIMULATED_JOINT",
        "MARGINAL_MODEL_STATUS": simulation.get("marginal_model_status", {}),
    }


def export_simulation_samples(
    simulation: Mapping[str, Any],
    destination: str | Path,
    *,
    model_id: str,
    model_version: str,
    model_sha256: str,
    simulation_id: str,
    compressed: bool = True,
    marginal_model_status: Mapping[str, str] | None = None,
) -> Path:
    """Write one metadata record plus one JSON record per simulation path."""
    n_paths = int(simulation.get("n_paths", 0))
    draws = {str(k): np.asarray(v) for k, v in simulation.get("draws", {}).items()}
    if n_paths <= 0 or not draws or any(len(v) != n_paths for v in draws.values()):
        raise ValueError("simulation draws are incomplete")
    _reject_market_data({"model_id": model_id, "model_version": model_version})
    status = dict(marginal_model_status or simulation.get("marginal_model_status", {}))
    market_state_map = dict(simulation.get("market_state_map", {}))
    path = Path(destination)
    opener = gzip.open if compressed or path.suffix == ".gz" else open
    metadata = {
        "SCHEMA_VERSION": SAMPLE_SCHEMA_VERSION,
        "MODEL_ID": model_id, "MODEL_VERSION": model_version,
        "MODEL_SHA256": model_sha256,
        "SIMULATION_ID": simulation_id,
        "SIMULATION_SEED": int(simulation.get("seed", 0)),
        "SIMULATION_PATH_COUNT": n_paths,
        "MARGINAL_MODEL_STATUS": status,
        "MARKET_STATE_MAP": market_state_map,
        "STATE_FIELDS": ["QB_PASS_YARDS", "LEAD_WR_REC_YARDS",
                         "LEAD_RB_RUSH_YARDS", "OPPOSING_QB_PASS_YARDS",
                         "TEAM_SCORE", "GAME_STATE"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(metadata, sort_keys=True) + "\n")
        for i in range(n_paths):
            row: dict[str, Any] = {"path_index": i}
            row.update({pid: float(values[i]) for pid, values in draws.items()})
            for market, pid in market_state_map.items():
                if pid in draws:
                    row[market] = float(draws[pid][i])
            for key in ("team_plays", "team_dropbacks", "team_score", "game_state"):
                if key in simulation:
                    value = simulation[key]
                    if isinstance(value, Mapping):
                        row[key] = {k: float(np.asarray(v)[i]) for k, v in value.items()}
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": JOINT_SCHEMA_VERSION,
        "type": "object",
        "required": ["GAME_ID", "MODEL_ID", "MODEL_VERSION", "MODEL_SHA256",
                      "SIMULATION_ID", "SIMULATION_SEED", "SIMULATION_PATH_COUNT",
                      "LEG_MARGINAL_PROBABILITIES", "MARGINAL_PRODUCT_PROBABILITY",
                      "JOINT_MODEL_PROBABILITY", "DEPENDENCY_DELTA",
                      "JOINT_STANDARD_ERROR", "JOINT_P05", "JOINT_P50", "JOINT_P95",
                      "DEPENDENCY_STATUS"],
        "properties": {
            "DEPENDENCY_STATUS": {"enum": ["SIMULATED_JOINT", "EMPIRICAL_JOINT",
                                             "UNRESOLVED_DEPENDENCY"]},
            "MARGINAL_PRODUCT_PROBABILITY": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "JOINT_MODEL_PROBABILITY": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "JOINT_P05": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "JOINT_P50": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "JOINT_P95": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
        "additionalProperties": True,
    }


def sample_schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SAMPLE_SCHEMA_VERSION,
        "description": "JSONL: metadata record followed by one record per seeded path",
        "metadata_required": ["SCHEMA_VERSION", "MODEL_ID", "MODEL_VERSION",
                              "MODEL_SHA256", "SIMULATION_ID", "SIMULATION_SEED",
                              "SIMULATION_PATH_COUNT", "MARGINAL_MODEL_STATUS"],
        "path_required": ["path_index"],
        "state_fields": ["QB_PASS_YARDS", "LEAD_WR_REC_YARDS",
                          "LEAD_RB_RUSH_YARDS", "OPPOSING_QB_PASS_YARDS",
                          "TEAM_SCORE", "GAME_STATE"],
    }
