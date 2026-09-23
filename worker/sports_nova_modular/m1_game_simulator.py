"""M1 adapter over the existing frozen V21 game simulator.

This module intentionally imports only football-state and simulator code.  It
does not import M2, M3, M4, M5, M6, sportsbook code, or market interfaces.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import numpy as np

from worker.sports_nova_v3.schemas import PregameState
from worker.sports_nova_v21.config import MODEL_VERSION
from worker.sports_nova_v21.simulator import SimulationBatch, simulate_game

from .contracts import M1OutputSchema
from .firewall import assert_market_free


_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {f"p{int(level * 100):02d}": float(np.quantile(array, level)) for level in _QUANTILES}


def _distribution_map(batch: SimulationBatch, *, player: bool) -> dict[str, Any]:
    ids = batch.player_ids if player else batch.team_ids
    stats = batch.player_stats if player else batch.team_stats
    return {
        identifier: {stat: _summary(np.asarray(values)[:, index]) for stat, values in stats.items()}
        for index, identifier in enumerate(ids)
    }


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_m1_output(batch: SimulationBatch, *, input_cutoff_ts: datetime) -> M1OutputSchema:
    """Convert raw simulator paths into the stable M1 boundary schema."""

    team_distributions = _distribution_map(batch, player=False)
    player_distributions = _distribution_map(batch, player=True)
    home, away = batch.team_ids
    home_score = np.asarray(batch.team_stats["score"])[:, 0]
    away_score = np.asarray(batch.team_stats["score"])[:, 1]
    joint_distributions = {
        "winner_probability": {
            outcome: float(np.mean(batch.winner == outcome)) for outcome in ("HOME", "AWAY", "TIE")
        },
        "margin": _summary(home_score - away_score),
        "total": _summary(home_score + away_score),
        "teams": {"home": home, "away": away},
    }
    payload = {
        "game_id": batch.game_id,
        "simulation_version": batch.model_version,
        "simulation_count": batch.n_sims,
        "team_distributions": team_distributions,
        "player_distributions": player_distributions,
        "joint_distributions": joint_distributions,
        "metadata": {
            "seed": batch.seed,
            "state_hash": batch.state_hash,
            "status": batch.status,
            "runtime": dict(batch.runtime),
        },
        "input_cutoff_ts": input_cutoff_ts,
    }
    return M1OutputSchema(**payload, hash=_canonical_hash(payload))


def simulate_m1(
    pregame_state: PregameState,
    n_sims: int,
    seed: int,
    model_version: str = MODEL_VERSION,
) -> SimulationBatch:
    """Run M1 with no external market, strategy, or execution input."""

    if not isinstance(pregame_state, PregameState):
        raise TypeError("validated PregameState required")
    assert_market_free(pregame_state.model_dump(mode="json"))
    return simulate_game(pregame_state, n_sims, seed, model_version)
