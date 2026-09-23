"""M3 external market normalization, isolated from M1 and M2."""

from __future__ import annotations

from datetime import datetime

from .contracts import M3OutputSchema


def normalize_market_observation(
    *,
    game_id: str,
    venue: str,
    market: str,
    timestamp: datetime,
    price: float,
    line: float | None = None,
    vig: float | None = None,
    liquidity: float | None = None,
) -> M3OutputSchema:
    return M3OutputSchema(
        game_id=game_id,
        venue=venue,
        market=market,
        timestamp=timestamp,
        price=price,
        line=line,
        vig=vig,
        liquidity=liquidity,
    )
