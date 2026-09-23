"""Fail-closed M1 input firewall.

The existing V3 schemas already reject market-derived features.  This second
boundary is deliberately generic so a future simulator entrypoint cannot
silently accept a raw mapping containing a price, line, odds, or strategy key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class M1MarketInputError(ValueError):
    """Raised when market or strategy data is presented to M1."""


_FORBIDDEN_EXACT = {"market", "price", "line", "odds", "pnl", "ev"}
_FORBIDDEN_PARTS = (
    "sportsbook",
    "kalshi",
    "moneyline",
    "predictionmarket",
    "marketprice",
    "marketprobability",
    "closingline",
    "bettingline",
    "execution",
    "strategy",
    "trading",
)


def _token(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def assert_market_free(payload: Any, *, path: str = "$") -> None:
    """Reject prohibited keys recursively; values are never keyword-scanned."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            token = _token(key)
            if token in _FORBIDDEN_EXACT or any(part in token for part in _FORBIDDEN_PARTS):
                raise M1MarketInputError(f"M1_FORBIDDEN_INPUT:{path}.{key}")
            assert_market_free(value, path=f"{path}.{key}")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            assert_market_free(value, path=f"{path}[{index}]")


def assert_m1_module_imports(module_names: set[str]) -> None:
    """Guard the runtime M1 adapter from importing downstream modules."""

    forbidden = {
        "sports_market_interface",
        "sports_research",
        "sports_nova_modular.market_book",
        "sports_nova_modular.comparator",
        "sports_nova_modular.trading_lab",
        "sports_nova_modular.capital_execution",
    }
    hit = sorted(module_names & forbidden)
    if hit:
        raise M1MarketInputError(f"M1_DOWNSTREAM_IMPORTS:{','.join(hit)}")
