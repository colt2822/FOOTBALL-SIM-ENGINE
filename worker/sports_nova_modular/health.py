"""Simulator-only health report; market metrics are intentionally impossible."""

from __future__ import annotations

from typing import Any


def simulator_health_report(
    *,
    simulator_version: str,
    structural_test_status: str,
    temporal_firewall: str,
    data_health: dict[str, Any],
    game_metrics: dict[str, Any],
    team_metrics: dict[str, Any],
    player_metrics: dict[str, Any],
    distribution_metrics: dict[str, Any],
    top_failure_modes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the allowed M1 health surface and fail if market metrics leak in."""

    report = {
        "SIMULATOR_VERSION": simulator_version,
        "STRUCTURAL_TEST_STATUS": structural_test_status,
        "TEMPORAL_FIREWALL": temporal_firewall,
        "DATA_HEALTH": data_health,
        "GAME_METRICS": game_metrics,
        "TEAM_METRICS": team_metrics,
        "PLAYER_METRICS": player_metrics,
        "DISTRIBUTION_METRICS": distribution_metrics,
        "TOP_SIMULATOR_FAILURE_MODES": top_failure_modes,
    }
    forbidden = ("pnl", "ev", "closing-line", "sportsbook", "profitability")
    serialized = repr(report).lower()
    if any(term in serialized for term in forbidden):
        raise ValueError("market metric leaked into M1 health report")
    return report
