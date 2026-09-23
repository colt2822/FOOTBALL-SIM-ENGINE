from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .distributions import sample_beta_binomial

@dataclass(frozen=True)
class BlockSelection:
    plays: int
    dropbacks: int
    pass_attempts: int
    sacks: int
    scrambles: int
    designed_rushes: int
    rush_attempts: int

def select_plays(plays: int, pass_rate: float, rng: np.random.Generator,
                 *, sack_rate: float = .065, scramble_rate: float = .07) -> BlockSelection:
    plays = max(0, int(plays))
    dropbacks = sample_beta_binomial(plays, max(pass_rate * 45, 1e-3), max((1-pass_rate) * 45, 1e-3), rng)
    sacks = int(rng.binomial(dropbacks, np.clip(sack_rate, 0, .3)))
    non_sack = dropbacks - sacks
    scrambles = int(rng.binomial(non_sack, np.clip(scramble_rate, 0, .4)))
    pass_attempts = non_sack - scrambles
    designed_rushes = plays - dropbacks
    return BlockSelection(plays, dropbacks, pass_attempts, sacks, scrambles,
                          designed_rushes, designed_rushes + scrambles)
