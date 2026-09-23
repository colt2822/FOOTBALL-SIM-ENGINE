"""Joint scoring transition and terminal score ledger."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from .game_state import GameState

ScoreKind = Literal["TD", "FG", "SAFETY", "PUNT", "TURNOVER", "END_CLOCK"]

@dataclass(frozen=True)
class ScoringEvent:
    team_id: str | None
    kind: ScoreKind
    points: int
    player_id: str | None = None
    passer_id: str | None = None
    receiver_id: str | None = None

    def __post_init__(self):
        expected = {"TD": 6, "FG": 3, "SAFETY": 2, "PUNT": 0, "TURNOVER": 0, "END_CLOCK": 0}[self.kind]
        if self.points != expected:
            raise ValueError(f"{self.kind} must carry {expected} points")
        if self.kind == "TD" and self.team_id is None:
            raise ValueError("touchdown requires team")
        if self.passer_id is not None and self.receiver_id is None:
            raise ValueError("pass TD credit requires receiver")

@dataclass(frozen=True)
class BlockTransition:
    state: GameState
    event: ScoringEvent
    seconds_elapsed: int
    ended_drive: bool

def score_event(team_id: str, kind: ScoreKind, *, player_id: str | None = None,
                passer_id: str | None = None, receiver_id: str | None = None) -> ScoringEvent:
    return ScoringEvent(team_id=team_id, kind=kind,
        points={"TD": 6, "FG": 3, "SAFETY": 2, "PUNT": 0, "TURNOVER": 0, "END_CLOCK": 0}[kind],
        player_id=player_id, passer_id=passer_id, receiver_id=receiver_id)

def transition_block(state: GameState, *, offense_team: str, seconds_elapsed: int,
                     field_position: int, event: ScoringEvent | None = None,
                     next_possession: str | None = None,
                     home_team_id: str = "home", away_team_id: str = "away") -> BlockTransition:
    event = event or ScoringEvent(None, "END_CLOCK", 0)
    if event.team_id is not None and event.team_id != offense_team and event.kind != "SAFETY":
        raise ValueError("block event team must be offense, except defensive safety")
    home_score, away_score = state.home_score, state.away_score
    scoring_team = event.team_id
    if event.points:
        if scoring_team == offense_team:
            if offense_team == home_team_id:
                home_score += event.points
            elif offense_team == away_team_id:
                away_score += event.points
            else:
                raise ValueError("offense team is outside game")
        elif event.kind == "SAFETY":
            if offense_team == home_team_id:
                away_score += event.points
            elif offense_team == away_team_id:
                home_score += event.points
            else:
                raise ValueError("offense team is outside game")
    next_state = state.advance(seconds=seconds_elapsed, possession=next_possession,
                               field_position=field_position,
                               home_score=home_score, away_score=away_score)
    return BlockTransition(next_state, event, max(0, int(seconds_elapsed)),
                           event.kind in {"TD", "FG", "SAFETY", "PUNT", "TURNOVER", "END_CLOCK"})

def apply_points(state: GameState, *, home_points: int = 0, away_points: int = 0,
                 seconds_elapsed: int = 0, next_possession: str | None = None) -> GameState:
    if min(home_points, away_points) < 0:
        raise ValueError("negative points")
    return state.advance(seconds=seconds_elapsed, possession=next_possession,
                         home_score=state.home_score + home_points,
                         away_score=state.away_score + away_points)

def winner(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "HOME"
    if away_score > home_score:
        return "AWAY"
    return "TIE"
