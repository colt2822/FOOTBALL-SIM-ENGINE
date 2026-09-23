from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .schemas import PregameState

@dataclass(frozen=True)
class EnvironmentDraw:
    pace_seconds: float
    home_efficiency: float
    away_efficiency: float
    home_pass_rate: float
    away_pass_rate: float
    epistemic_id: int

def _feature(obj, name: str, default: float) -> float:
    for item in getattr(obj, "features", ()):
        if item.name == name and item.value is not None:
            return float(item.value)
    return default

def draw_environment(pregame: PregameState, rng: np.random.Generator, *, epistemic_id: int = 0) -> EnvironmentDraw:
    pace_home = _feature(pregame.home, "pace_seconds", 27.0)
    pace_away = _feature(pregame.away, "pace_seconds", 27.0)
    pace = float(np.clip(rng.normal((pace_home + pace_away) / 2.0, 1.5), 18.0, 36.0))
    home_pass = np.clip(_feature(pregame.home, "pass_rate", .58) + rng.normal(0, .025), .25, .8)
    away_pass = np.clip(_feature(pregame.away, "pass_rate", .58) + rng.normal(0, .025), .25, .8)
    return EnvironmentDraw(pace, float(rng.normal(_feature(pregame.home, "recent_form", 0.), .35)),
                           float(rng.normal(_feature(pregame.away, "recent_form", 0.), .35)),
                           float(home_pass), float(away_pass), int(epistemic_id))
