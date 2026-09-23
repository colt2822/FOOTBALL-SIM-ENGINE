"""Causal pregame adapter.

The adapter accepts an explicit mapping from a canonical evidence store. It
does not infer teams, players, statuses, availability timestamps, or shares.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
import json

from .schemas import PregameState

_MARKET_TERMS = ("odds", "spread", "moneyline", "sportsbook", "vegas", "vig")

class PregameUnavailable(ValueError):
    """Required causal evidence is absent or ambiguous."""

def _reject_market(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(term in key_text for term in _MARKET_TERMS):
                raise PregameUnavailable(f"market-derived field forbidden at {path}.{key}")
            _reject_market(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _reject_market(item, f"{path}[{i}]")

def validate_evidence_revision(evidence: Mapping[str, Any], *, as_of: datetime) -> dict[str, Any]:
    if "available_at" not in evidence or "raw_sha256" not in evidence:
        raise PregameUnavailable("exact immutable evidence revision and availability are required")
    available = evidence["available_at"]
    if available.tzinfo is None or available.utcoffset() is None or available > as_of:
        raise PregameUnavailable("evidence is unavailable at requested as_of")
    return dict(evidence)

def resolve_game_roster(feature_store: Any, *, game_id: str, as_of: datetime) -> Mapping[str, Any]:
    """Resolve an explicit game record; unresolved or ambiguous records fail closed."""
    if isinstance(feature_store, Mapping):
        record = feature_store.get(game_id, feature_store.get("game"))
    elif hasattr(feature_store, "get_game"):
        record = feature_store.get_game(game_id=game_id, as_of=as_of)
    else:
        record = None
    if not isinstance(record, Mapping):
        raise PregameUnavailable(f"no canonical game record for {game_id}")
    if str(record.get("game_id", game_id)) != game_id:
        raise PregameUnavailable("game identity mismatch")
    return record

def build_pregame_state(*, game_id: str, as_of, feature_store, identity_registry) -> PregameState:
    """Build a fully validated immutable state from explicit canonical records.

    `identity_registry` is an authority parameter: the record must already
    contain its registry hash and canonical IDs. This function does not create
    a second identity authority or use a display-name fallback.
    """
    record = resolve_game_roster(feature_store, game_id=game_id, as_of=as_of)
    _reject_market(record)
    payload = dict(record)
    payload["game_id"] = game_id
    payload["as_of"] = as_of
    if identity_registry is not None:
        expected = getattr(identity_registry, "registry_sha256", None)
        if expected is None and isinstance(identity_registry, Mapping):
            expected = identity_registry.get("registry_sha256")
        if expected is not None and payload.get("identity_registry_sha256") != expected:
            raise PregameUnavailable("identity registry hash mismatch")
    try:
        return PregameState.model_validate(payload)
    except Exception as exc:
        raise PregameUnavailable(str(exc)) from exc
