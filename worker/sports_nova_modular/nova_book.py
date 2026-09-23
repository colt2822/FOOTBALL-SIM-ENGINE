"""M2 NOVA-book boundary; it consumes M1 and nothing external."""

from __future__ import annotations

from typing import Any

from .contracts import M1OutputSchema, M2OutputSchema


def build_nova_book(
    simulation: M1OutputSchema,
    *,
    nova_book_version: str,
    fair_prices: dict[str, float],
    fair_lines: dict[str, float],
    probabilities: dict[str, float],
) -> M2OutputSchema:
    """Seal fair-value representation derived from one M1 simulation hash.

    The pricing/calibration implementation remains caller-owned and frozen;
    this boundary prevents it from receiving M3 observations accidentally.
    """

    return M2OutputSchema(
        game_id=simulation.game_id,
        nova_book_version=nova_book_version,
        fair_prices=dict(fair_prices),
        fair_lines=dict(fair_lines),
        probabilities=dict(probabilities),
        source_sim_hash=simulation.hash,
    )
