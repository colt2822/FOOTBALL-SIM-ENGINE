from typing import Literal
import math
import numpy as np
from .schemas import Frozen, ProbabilityResult

class Predicate(Frozen):
    scope: Literal["player", "team", "game"]
    entity_id: str
    stat: Literal["pass_attempts", "pass_yards", "rush_attempts", "rush_yards",
        "targets", "receptions", "receiving_yards", "pass_tds", "rush_tds",
        "receiving_tds", "scored_tds", "score", "win", "total"]
    op: Literal[">", "<"]
    threshold: float

    @property
    def key(self) -> str:
        return f"{self.scope}:{self.entity_id}:{self.stat}:{self.op}:{self.threshold}"

def probability_from_masks(events: tuple[tuple[bool, ...], ...],
                           given: tuple[tuple[bool, ...], ...] = ()) -> ProbabilityResult:
    """Exact sample-space contract for P(A), P(AND), and P(AND | AND)."""
    if not events or not events[0]:
        raise ValueError("nonempty aligned events required")
    n = len(events[0])
    if any(len(mask) != n or any(type(v) is not bool for v in mask) for mask in events + given):
        raise ValueError("unaligned or nonboolean mask")
    eligible = [i for i in range(n) if all(mask[i] for mask in given)]
    hits = sum(all(mask[i] for mask in events) for i in eligible)
    return ProbabilityResult(probability=hits / len(eligible) if eligible else None,
        numerator=hits, denominator=len(eligible), status="OK" if eligible else "NO_CONDITION_SUPPORT")

def query(samples, predicates: tuple[Predicate, ...], given: tuple[Predicate, ...] = ()):
    """Evaluate all legs on one aligned SimulationBatch sample axis."""
    if not predicates:
        raise ValueError("at least one event predicate is required")
    event_masks = tuple(evaluate_predicate(samples, p) for p in predicates)
    given_masks = tuple(evaluate_predicate(samples, p) for p in given)
    return probability_from_masks(tuple(tuple(bool(x) for x in mask) for mask in event_masks),
        tuple(tuple(bool(x) for x in mask) for mask in given_masks))

def _values(samples, predicate: Predicate) -> np.ndarray:
    if not math.isfinite(float(predicate.threshold)):
        raise ValueError("predicate threshold must be finite")
    stat = predicate.stat
    if predicate.scope == "player":
        if stat in {"score", "win", "total"}:
            raise ValueError("invalid player predicate stat")
        if stat == "scored_tds":
            return (samples.player(predicate.entity_id, "rush_tds") +
                    samples.player(predicate.entity_id, "receiving_tds"))
        if stat not in samples.player_stats:
            raise ValueError(f"unsupported player stat: {stat}")
        return samples.player(predicate.entity_id, stat)
    if predicate.scope == "team":
        if predicate.entity_id not in samples.team_ids:
            raise ValueError("unknown team identity")
        if stat == "win":
            index = samples.team_ids.index(predicate.entity_id)
            label = "HOME" if index == 0 else "AWAY"
            return (samples.winner == label).astype(float)
        if stat in {"total", "score"}:
            return samples.team(predicate.entity_id, "score")
        if stat not in samples.team_stats:
            raise ValueError(f"unsupported team stat: {stat}")
        return samples.team(predicate.entity_id, stat)
    if predicate.scope == "game":
        if predicate.entity_id != samples.game_id:
            raise ValueError("unknown game identity")
        if stat == "total":
            return samples.team_stats["score"].sum(axis=1)
        if stat == "score":
            return samples.team_stats["score"].sum(axis=1)
        if stat == "win":
            return (samples.winner != "TIE").astype(float)
        raise ValueError("unsupported game stat")
    raise ValueError("unsupported predicate scope")

def evaluate_predicate(samples, predicate: Predicate) -> np.ndarray:
    values = _values(samples, predicate)
    if predicate.op == ">":
        return values > predicate.threshold
    if predicate.op == "<":
        return values < predicate.threshold
    raise ValueError("unsupported predicate operator")

def joint_probability(samples, predicates: tuple[Predicate, ...], given: tuple[Predicate, ...] = ()) -> ProbabilityResult:
    return query(samples, predicates, given)
