"""M6 capital/risk boundary. This adapter does not place orders."""

from __future__ import annotations

from typing import Any

from .contracts import M5OutputSchema, M6OutputSchema


def approve_execution_plan(
    candidate: M5OutputSchema,
    *,
    stake: float,
    price_limit: float | None,
    venue: str,
    risk: dict[str, Any],
    correlation_exposure: dict[str, Any],
    execution_status: str = "PLAN_ONLY",
) -> M6OutputSchema:
    """Create an auditable plan from an M5 candidate without executing it."""

    return M6OutputSchema(
        stake=stake,
        price_limit=price_limit,
        venue=venue,
        risk=dict(risk),
        correlation_exposure=dict(correlation_exposure),
        execution_status=execution_status,
        audit={"candidate_id": candidate.candidate_id, "strategy_id": candidate.strategy_id},
    )
