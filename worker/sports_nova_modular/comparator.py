"""M4 comparison boundary. It never mutates or retunes M1/M2."""

from __future__ import annotations

from .contracts import M2OutputSchema, M3OutputSchema, M4OutputSchema


def compare_nova_to_market(nova: M2OutputSchema, market: M3OutputSchema) -> M4OutputSchema:
    if nova.game_id != market.game_id:
        raise ValueError("M2 and M3 game_id must match")
    if market.market in nova.fair_prices:
        nova_value = nova.fair_prices[market.market]
    elif market.market in nova.fair_lines:
        nova_value = nova.fair_lines[market.market]
    else:
        raise KeyError(f"M2 has no fair value for market {market.market!r}")
    return M4OutputSchema(
        game_id=market.game_id,
        market=market.market,
        nova_value=float(nova_value),
        external_value=float(market.price),
        delta=float(nova_value) - float(market.price),
        timestamp=market.timestamp,
    )
