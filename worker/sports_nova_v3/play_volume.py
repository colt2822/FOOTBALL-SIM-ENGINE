from __future__ import annotations
import numpy as np
from .distributions import sample_negative_binomial
from .environment import EnvironmentDraw
from .game_state import GameState

def draw_block_volume(state: GameState, env: EnvironmentDraw, rng: np.random.Generator,
                      *, dispersion: float = 8.0) -> int:
    remaining_blocks = max(1, 24 - state.block_index)
    # A block is a drive-level unit in this engine.  The prior expression used
    # a per-minute pace factor as though it were plays per block, collapsing
    # the game to roughly 50 plays.  Convert the pace signal to a drive-sized
    # play expectation; the clock cap still bounds terminal drives.
    expected = np.clip((60.0 / env.pace_seconds) * 2.7, 2.0, 12.0)
    # A terminal block is bounded by the remaining clock and cannot be negative.
    clock_cap = max(0, int(np.ceil(state.seconds_remaining / max(env.pace_seconds, 1.0))))
    return sample_negative_binomial(expected, dispersion, rng, maximum=max(0, min(12, clock_cap)))
