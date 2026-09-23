"""M5 strategy-research boundary; no capital execution occurs here."""

from __future__ import annotations

from typing import Any, Iterable

from .contracts import M5OutputSchema


def make_candidate(
    *,
    strategy_id: str,
    candidate_id: str,
    inputs_used: Iterable[str],
    signal: str,
    edge_estimate: float | None,
    execution_requirement: dict[str, Any],
) -> M5OutputSchema:
    return M5OutputSchema(
        strategy_id=strategy_id,
        candidate_id=candidate_id,
        inputs_used=tuple(inputs_used),
        signal=signal,
        edge_estimate=edge_estimate,
        execution_requirement=dict(execution_requirement),
    )
