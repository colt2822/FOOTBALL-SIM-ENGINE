from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .distributions import DistributionParameters, sample_beta_binomial, sample_compound_signed

@dataclass(frozen=True)
class EfficiencyResult:
    receptions: int
    receiving_yards: int
    rushing_yards: int
    pass_tds: int
    rush_tds: int
    receiving_tds: int

def draw_efficiency(targets: int, carries: int, params: DistributionParameters,
                    rng: np.random.Generator, *, catch_alpha: float | None = None,
                    catch_beta: float | None = None) -> tuple[int, int, int]:
    receptions = sample_beta_binomial(targets, catch_alpha or params.catch_alpha,
                                      catch_beta or params.catch_beta, rng)
    receiving_yards = sample_compound_signed(receptions, params.receiving_yards_mean,
                                             params.receiving_yards_sd, rng)
    rushing_yards = sample_compound_signed(carries, params.rush_yards_mean,
                                           params.rush_yards_sd, rng)
    return receptions, receiving_yards, rushing_yards
