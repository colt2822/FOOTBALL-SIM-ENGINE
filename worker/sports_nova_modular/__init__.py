"""Explicit SPORTS-NOVA module boundaries.

This package is an adapter layer around the frozen SPORTS-NOVA implementations.
It does not replace or retune the simulator.  M1 is intentionally importable
without importing any market, comparator, trading, or execution module.
"""

from .contracts import (
    M1OutputSchema,
    M2OutputSchema,
    M3OutputSchema,
    M4OutputSchema,
    M5OutputSchema,
    M6OutputSchema,
)
from .m1_game_simulator import build_m1_output, simulate_m1

__all__ = [
    "M1OutputSchema",
    "M2OutputSchema",
    "M3OutputSchema",
    "M4OutputSchema",
    "M5OutputSchema",
    "M6OutputSchema",
    "build_m1_output",
    "simulate_m1",
]
