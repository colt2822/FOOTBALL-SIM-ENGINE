"""Strict, deeply immutable boundary types; no arbitrary feature dictionaries."""
from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ARCHITECTURE_VERSION, MODEL_VERSION

class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

class Evidence(Frozen):
    source_id: str = Field(min_length=1)  # Registry identifier, never credentials/URL.
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_end: datetime
    available_at: datetime
    retrieved_at: datetime
    published_at: datetime | None = None
    availability_basis: Literal["VERIFIED_ARCHIVE", "LIVE_RECEIPT"]

    @model_validator(mode="after")
    def timestamps(self):
        for t in (self.event_end, self.available_at, self.retrieved_at, self.published_at):
            if t is not None and (t.tzinfo is None or t.utcoffset() is None):
                raise ValueError("timezone-aware timestamps required")
        if self.available_at > self.retrieved_at:
            raise ValueError("availability cannot follow retrieval")
        if self.published_at is not None and self.published_at > self.available_at:
            raise ValueError("publication follows claimed availability")
        if self.availability_basis == "LIVE_RECEIPT" and self.available_at != self.retrieved_at:
            raise ValueError("live availability must equal first verified receipt")
        return self

FeatureName = Literal["pace_seconds", "pass_rate", "rush_rate", "proe",
    "opponent_pass_efficiency", "opponent_rush_efficiency", "recent_form",
    "catch_rate", "yards_per_reception", "yards_per_carry", "sack_rate",
    "scramble_rate", "red_zone_rate", "wind_forecast", "temperature_forecast",
    "pass_td_rate", "rush_td_rate",
    "games_since_last_team_game"]

class Feature(Frozen):
    name: FeatureName
    value: float | None
    unit: str = Field(min_length=1)
    evidence: Evidence

    @model_validator(mode="after")
    def finite_value(self):
        if self.value is not None and not isfinite(float(self.value)):
            raise ValueError("feature value must be finite or null")
        if any(term in self.name.lower() for term in ("odds", "spread", "line", "vegas", "moneyline")):
            raise ValueError("market feature is forbidden")
        return self

class Uncertainty(Frozen):
    distribution_variance: float | None = Field(default=None, ge=0)
    effective_sample_size: float = Field(ge=0)
    role_concentration: float | None = Field(default=None, gt=0)
    personnel_unknown: bool
    model_variance: float | None = Field(default=None, ge=0)

class PlayerState(Frozen):
    player_id: str = Field(min_length=1)
    team_id: str = Field(min_length=1)
    position: Literal["QB", "RB", "WR", "TE", "OTHER"]
    availability: Literal["AVAILABLE", "OUT", "UNKNOWN"]
    identity_evidence: Evidence
    availability_evidence: Evidence | None = None
    features: tuple[Feature, ...] = ()
    uncertainty: Uncertainty

    @model_validator(mode="after")
    def availability_requires_evidence(self):
        if self.availability != "UNKNOWN" and self.availability_evidence is None:
            raise ValueError("known personnel status requires evidence")
        return self

class OpportunityShares(Frozen):
    player_ids: tuple[str, ...]
    shares: tuple[float, ...]
    residual_share: float = Field(ge=0, le=1)
    evidence: Evidence

    @model_validator(mode="after")
    def coherent(self):
        if len(self.player_ids) != len(self.shares) or len(set(self.player_ids)) != len(self.player_ids):
            raise ValueError("unique aligned share identities required")
        if any(not p for p in self.player_ids) or any(not isfinite(x) or not 0 <= x <= 1 for x in self.shares):
            raise ValueError("invalid share")
        if abs(sum(self.shares) + self.residual_share - 1) > 1e-10:
            raise ValueError("shares including residual must sum to one")
        return self

class TeamState(Frozen):
    team_id: str = Field(min_length=1)
    identity_evidence: Evidence
    features: tuple[Feature, ...]
    target_shares: OpportunityShares
    carry_shares: OpportunityShares

    @model_validator(mode="after")
    def unique_features(self):
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("duplicate team feature")
        return self

class PregameState(Frozen):
    architecture_version: Literal["SPORTS_NOVA_V3_ARCH_1"] = ARCHITECTURE_VERSION
    game_id: str = Field(min_length=1)
    sport: Literal["NFL"] = "NFL"
    kickoff: datetime
    as_of: datetime
    ruleset_id: str = Field(min_length=1)
    identity_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    game_evidence: Evidence
    home: TeamState
    away: TeamState
    players: tuple[PlayerState, ...]
    environment: tuple[Feature, ...] = ()

    @model_validator(mode="after")
    def causal_identity(self):
        if any(t.tzinfo is None or t.utcoffset() is None for t in (self.as_of, self.kickoff)):
            raise ValueError("timezone required")
        if self.as_of >= self.kickoff:
            raise ValueError("as_of must precede kickoff")
        if self.home.team_id == self.away.team_id:
            raise ValueError("distinct teams required")
        ids = [p.player_id for p in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate player identity")
        for p in self.players:
            if p.team_id not in (self.home.team_id, self.away.team_id):
                raise ValueError("player outside game")
        for team in (self.home, self.away):
            eligible = {p.player_id for p in self.players if p.team_id == team.team_id}
            for shares in (team.target_shares, team.carry_shares):
                if not set(shares.player_ids) <= eligible:
                    raise ValueError("share identity outside team")
                for pid, share in zip(shares.player_ids, shares.shares):
                    if share > 0 and any(p.player_id == pid and p.availability == "OUT" for p in self.players):
                        raise ValueError("OUT player has positive opportunity share")
        def walk(obj):
            if isinstance(obj, Evidence):
                if obj.available_at > self.as_of or obj.event_end > self.as_of:
                    raise ValueError("future or unavailable evidence")
            if isinstance(obj, Frozen):
                for key in type(obj).model_fields:
                    walk(getattr(obj, key))
            elif isinstance(obj, tuple):
                for value in obj:
                    walk(value)
        walk(self)
        for team in (self.home, self.away):
            if team.identity_evidence.event_end > self.as_of:
                raise ValueError("team identity is not causal at as_of")
        return self

class PlayerOutcome(Frozen):
    player_id: str
    team_id: str
    pass_attempts: int = Field(ge=0)
    pass_yards: int  # Yardage may legitimately be negative; counts may not.
    rush_attempts: int = Field(ge=0)
    rush_yards: int
    targets: int = Field(ge=0)
    receptions: int = Field(ge=0)
    receiving_yards: int
    pass_tds: int = Field(ge=0)
    rush_tds: int = Field(ge=0)
    receiving_tds: int = Field(ge=0)
    other_tds: int = Field(ge=0)

    @model_validator(mode="after")
    def counts(self):
        if self.receptions > self.targets or self.receiving_tds > self.receptions:
            raise ValueError("receiving count hierarchy violated")
        if self.pass_tds > self.pass_attempts or self.rush_tds > self.rush_attempts:
            raise ValueError("TD exceeds opportunity count")
        for count, yards in ((self.pass_attempts, self.pass_yards),
                (self.rush_attempts, self.rush_yards), (self.receptions, self.receiving_yards)):
            if count == 0 and yards != 0:
                raise ValueError("yardage without opportunity")
        return self

class FrozenModelRef(Frozen):
    version: Literal["sports_nova_v3.drive_block.1"] = MODEL_VERSION
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_available_through: datetime
    calibration_available_through: datetime

    @model_validator(mode="after")
    def cutoffs(self):
        if self.training_available_through.tzinfo is None or self.calibration_available_through.tzinfo is None:
            raise ValueError("model cutoffs require timezone")
        if self.calibration_available_through > self.training_available_through:
            raise ValueError("calibration cutoff cannot exceed training cutoff")
        return self

class ProbabilityResult(Frozen):
    probability: float | None = Field(ge=0, le=1)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    status: Literal["OK", "NO_CONDITION_SUPPORT"]

    @model_validator(mode="after")
    def coherent(self):
        if self.numerator > self.denominator:
            raise ValueError("numerator exceeds denominator")
        if self.denominator == 0:
            if self.probability is not None or self.status != "NO_CONDITION_SUPPORT":
                raise ValueError("empty condition must be unresolved")
        elif self.status != "OK" or self.probability is None or abs(self.probability - self.numerator / self.denominator) > 1e-12:
            raise ValueError("probability inconsistent with counts")
        return self
