from __future__ import annotations
import numpy as np
from .environment import EnvironmentDraw
from .game_state import GameState
from .schemas import PregameState

def _proe(team, default=0.0):
    for item in team.features:
        if item.name == "proe" and item.value is not None:
            return float(item.value)
    return default

def script_pass_rate(pregame: PregameState, state: GameState, env: EnvironmentDraw) -> float:
    base = env.home_pass_rate if state.possession == pregame.home.team_id else env.away_pass_rate
    team = pregame.home if state.possession == pregame.home.team_id else pregame.away
    differential = state.home_score - state.away_score
    if state.possession == pregame.away.team_id:
        differential = -differential
    # Positive differential means the offense is leading and generally reduces pass rate.
    value = base + .015 * (-np.sign(differential)) + .0015 * np.clip(-differential, -21, 21) + .04 * _proe(team)
    return float(np.clip(value, .25, .85))
