"""Stable, immutable contracts between the six SPORTS-NOVA modules."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


def _utc_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class M1OutputSchema(_Frozen):
    """Football-only simulation output; no market or strategy fields."""

    game_id: str = Field(min_length=1)
    simulation_version: str = Field(min_length=1)
    simulation_count: int = Field(gt=0)
    team_distributions: dict[str, Any]
    player_distributions: dict[str, Any]
    joint_distributions: dict[str, Any]
    metadata: dict[str, Any]
    input_cutoff_ts: datetime
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def causal_cutoff(self):
        _utc_timestamp(self.input_cutoff_ts)
        return self


class M2OutputSchema(_Frozen):
    """NOVA fair values derived from one sealed M1 output."""

    game_id: str = Field(min_length=1)
    nova_book_version: str = Field(min_length=1)
    fair_prices: dict[str, float]
    fair_lines: dict[str, float]
    probabilities: dict[str, float]
    source_sim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def probabilities_are_valid(self):
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("NOVA probabilities must be in [0, 1]")
        return self


class M3OutputSchema(_Frozen):
    """One normalized external-market observation."""

    game_id: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    market: str = Field(min_length=1)
    timestamp: datetime
    price: float
    line: float | None = None
    vig: float | None = Field(default=None, ge=0)
    liquidity: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def timestamp_is_causal(self):
        _utc_timestamp(self.timestamp)
        return self


class M4OutputSchema(_Frozen):
    """Mechanical comparison of an M2 value and an M3 observation."""

    game_id: str = Field(min_length=1)
    market: str = Field(min_length=1)
    nova_value: float
    external_value: float
    delta: float
    timestamp: datetime

    @model_validator(mode="after")
    def delta_is_mechanical(self):
        _utc_timestamp(self.timestamp)
        if abs(self.delta - (self.nova_value - self.external_value)) > 1e-12:
            raise ValueError("delta must equal nova_value - external_value")
        return self


class M5OutputSchema(_Frozen):
    """Research candidate; it is not an execution instruction."""

    strategy_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    inputs_used: tuple[str, ...]
    signal: str = Field(min_length=1)
    edge_estimate: float | None = None
    execution_requirement: dict[str, Any]


class M6OutputSchema(_Frozen):
    """Capital/risk plan emitted only after an M5 candidate is approved."""

    stake: float = Field(ge=0)
    price_limit: float | None = None
    venue: str = Field(min_length=1)
    risk: dict[str, Any]
    correlation_exposure: dict[str, Any]
    execution_status: str = Field(min_length=1)
    audit: dict[str, Any]
