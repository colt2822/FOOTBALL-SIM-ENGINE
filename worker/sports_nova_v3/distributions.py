"""Causal distribution primitives for the V3 drive/block engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping
import math
import numpy as np

FAMILIES = (
    ("block_plays", "clock_bounded_negative_binomial"),
    ("pass_rush_split", "beta_binomial_dropbacks_then_multinomial_disposition"),
    ("targets_carries", "dirichlet_multinomial_with_explicit_residual"),
    ("receptions", "beta_binomial_given_targets"),
    ("yards", "compound_signed_empirical_per_opportunity_shrunk_by_role"),
    ("touchdowns_points", "joint_block_transition_categorical_then_scorer_allocation"),
)

@dataclass(frozen=True)
class DistributionParameters:
    version: str
    training_cutoff: datetime | None
    volume_dispersion: float
    pass_concentration: float
    catch_alpha: float
    catch_beta: float
    td_rate: float
    fg_rate: float
    pass_yards_mean: float
    pass_yards_sd: float
    rush_yards_mean: float
    rush_yards_sd: float
    receiving_yards_mean: float
    receiving_yards_sd: float
    status: str = "UNVALIDATED_DEFAULT"
    source_rows: int = 0

def _finite(rows: Iterable[Mapping[str, Any]], key: str, cutoff=None) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if cutoff is not None:
            for stamp_key in ("available_at", "source_available_at", "event_end"):
                stamp = row.get(stamp_key)
                if stamp is not None and stamp > cutoff:
                    break
            else:
                try:
                    value = float(value)
                    if math.isfinite(value):
                        out.append(value)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                value = float(value)
                if math.isfinite(value):
                    out.append(value)
            except (TypeError, ValueError):
                pass
    return out

def _mean(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default

def _sd(values: list[float], default: float) -> float:
    return max(float(np.std(values, ddof=1)) if len(values) > 1 else default, 1e-6)

def fit_distributions(training_rows: Iterable[Mapping[str, Any]], cutoff: datetime | None) -> DistributionParameters:
    """Fit only rows proven available by ``cutoff``; empty input is explicit fallback."""
    rows = list(training_rows or [])
    pass_rates = _finite(rows, "pass_rate", cutoff)
    catches = _finite(rows, "catch_rate", cutoff)
    pass_yards = _finite(rows, "pass_yards_per_attempt", cutoff)
    rush_yards = _finite(rows, "rush_yards_per_attempt", cutoff)
    rec_yards = _finite(rows, "receiving_yards_per_reception", cutoff)
    n = len(rows)
    mean_catch = min(max(_mean(catches, .64), .05), .95)
    concentration = max(2.0, min(500.0, float(len(catches) or 50)))
    return DistributionParameters(
        version="sports_nova_v3.distributions.1", training_cutoff=cutoff,
        volume_dispersion=8.0, pass_concentration=50.0,
        catch_alpha=mean_catch * concentration, catch_beta=(1 - mean_catch) * concentration,
        td_rate=min(max(_mean(_finite(rows, "td_rate", cutoff), .045), .001), .2),
        fg_rate=min(max(_mean(_finite(rows, "fg_rate", cutoff), .08), 0.0), .5),
        pass_yards_mean=_mean(pass_yards, 7.2), pass_yards_sd=_sd(pass_yards, 9.0),
        rush_yards_mean=_mean(rush_yards, 4.2), rush_yards_sd=_sd(rush_yards, 5.5),
        receiving_yards_mean=_mean(rec_yards, 11.0), receiving_yards_sd=_sd(rec_yards, 10.0),
        status="FIT_FROM_CAUSAL_ROWS" if n else "UNVALIDATED_DEFAULT", source_rows=n)

def sample_negative_binomial(mean: float, dispersion: float, rng: np.random.Generator,
                            *, maximum: int | None = None) -> int:
    mean = max(float(mean), 0.0)
    dispersion = max(float(dispersion), 1e-6)
    if mean == 0:
        return 0
    p = dispersion / (dispersion + mean)
    value = int(rng.negative_binomial(dispersion, p))
    return min(value, maximum) if maximum is not None else value

def sample_beta_binomial(n: int, alpha: float, beta: float, rng: np.random.Generator) -> int:
    if n <= 0:
        return 0
    p = float(rng.beta(max(alpha, 1e-6), max(beta, 1e-6)))
    return int(rng.binomial(int(n), p))

def sample_dirichlet_multinomial(n: int, shares: np.ndarray, concentration: float,
                                 rng: np.random.Generator) -> np.ndarray:
    shares = np.asarray(shares, dtype=float)
    if n < 0 or shares.ndim != 1 or len(shares) == 0:
        raise ValueError("invalid allocation inputs")
    shares = np.clip(shares, 0.0, None)
    if shares.sum() <= 0:
        shares = np.ones(len(shares)) / len(shares)
    else:
        shares = shares / shares.sum()
    p = rng.dirichlet(np.maximum(shares * max(concentration, 1e-6), 1e-6))
    return rng.multinomial(int(n), p)

def sample_compound_signed(count: int, mean: float, sd: float, rng: np.random.Generator) -> int:
    if count <= 0:
        return 0
    draws = rng.normal(float(mean), max(float(sd), 1e-6), size=int(count))
    return int(np.rint(draws).sum())
