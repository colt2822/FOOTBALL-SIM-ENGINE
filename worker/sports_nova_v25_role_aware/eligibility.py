"""Deterministic pregame game-eligibility and role resolution for V25.  Pure functions: no IO, no market data, no outcomes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .config import (
    DOUBTFUL_AVAILABILITY_WEIGHT, ELIGIBLE_ROSTER_STATUS, NON_SKILL_ROLE, QUESTIONABLE_AVAILABILITY_WEIGHT,
    REASON_PRECEDENCE, ROSTER_STATUS_REASON, SKILL_ROLES)


@dataclass(frozen=True)
class RosterSnapshot:
    """Everything eligibility/role may read, all available pre-kickoff.

    roster:      gsis_id -> (team, status) from the weekly roster valid for the game week
    game_status: gsis_id -> OUT | DOUBTFUL | QUESTIONABLE from the pre-kickoff injury report for this game
    positions:   gsis_id -> (roster position, depth_chart_position) from the same roster_weekly file (optional; falls back to PlayerState.position)
    """
    roster: Mapping[str, tuple[str, str]]
    game_status: Mapping[str, str]
    roster_week: int
    source_sha256: Mapping[str, str] = field(default_factory=dict)
    positions: Mapping[str, tuple[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerEligibility:
    player_id: str
    team_id: str
    eligible: bool
    reason: str | None
    all_reasons: tuple[str, ...]
    availability_weight: float
    role: str
    detail: str


def resolve_role(player, snapshot: RosterSnapshot) -> str:
    pos, depth = snapshot.positions.get(player.player_id, (None, None))
    for cand in (pos, depth, player.position):
        if cand in SKILL_ROLES:
            return cand
    return NON_SKILL_ROLE


def classify_player(player, snapshot: RosterSnapshot) -> PlayerEligibility:
    roster_team, status = snapshot.roster.get(player.player_id, (None, None))
    reasons = []
    if roster_team is None:
        reasons.append("OFF_ROSTER")
    elif roster_team != player.team_id:
        reasons.append("OTHER_TEAM")
    elif status != ELIGIBLE_ROSTER_STATUS:
        reasons.append(ROSTER_STATUS_REASON.get(status, "OTHER_EXPLICIT_INELIGIBLE"))
    game_status = snapshot.game_status.get(player.player_id)
    if player.availability == "OUT" or game_status == "OUT":
        reasons.append("OUT")
    primary = next((r for r in REASON_PRECEDENCE if r in reasons), None)
    weight = 1.0
    if not reasons:
        weight = {"DOUBTFUL": DOUBTFUL_AVAILABILITY_WEIGHT, "QUESTIONABLE": QUESTIONABLE_AVAILABILITY_WEIGHT}.get(game_status, 1.0)
    detail = f"roster={roster_team}/{status} state_team={player.team_id} injury={game_status}"
    return PlayerEligibility(player.player_id, player.team_id, not reasons, primary,
                             tuple(r for r in REASON_PRECEDENCE if r in reasons), weight, resolve_role(player, snapshot), detail)


def game_eligibility(state, snapshot: RosterSnapshot) -> dict[str, PlayerEligibility]:
    return {p.player_id: classify_player(p, snapshot) for p in state.players}
