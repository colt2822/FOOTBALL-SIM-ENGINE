# SPORTS-NOVA modular separation reset

## Status

```text
MODULAR_RESET_STATUS=PASS
CURRENT_SIMULATOR_CHAMPION=V21
V22_CLASSIFICATION=NOVA_BOOK_CALIBRATION_COMPONENT
```

M1 is now an explicit football-only boundary over the existing frozen V21
simulator. The simulator internals, learned parameters, historical artifacts,
and version hashes were not rewritten.

## Module status

```text
M1_SIMULATOR_STATUS=PASS
M2_NOVA_BOOK_STATUS=PASS
M3_MARKET_BOOK_STATUS=PASS
M4_COMPARATOR_STATUS=PASS
M5_TRADING_LAB_STATUS=PASS
M6_CAPITAL_STATUS=PASS
```

The new typed contracts and one-way adapters are in
`worker/sports_nova_modular/`:

- `m1_game_simulator.py` wraps V21 and emits `M1OutputSchema`.
- `nova_book.py` consumes only sealed M1 output.
- `market_book.py` normalizes external observations independently.
- `comparator.py` compares M2 to M3 without mutation or retuning.
- `trading_lab.py` emits research candidates only.
- `capital_execution.py` emits plan/audit records and does not place orders.
- `firewall.py` rejects market, strategy, trading, and execution keys at M1.
- `audit.py` performs the static import-boundary audit.

## Coupling audit

```text
SIMULATOR_MARKET_DEPENDENCIES_BEFORE=0 runtime imports/inputs
SIMULATOR_MARKET_DEPENDENCIES_AFTER=0 runtime imports/inputs
M1_MARKET_INPUTS=0
M1_STRATEGY_DEPENDENCIES=0
M1_EXECUTION_DEPENDENCIES=0
TEMPORAL_FIREWALL_STATUS=PASS
```

No M1 runtime violation was found in `worker/sports_nova_v3/`,
`worker/sports_nova_v21/`, or `worker/sports_probability_engine.py`. Existing
market-related strings in those paths are fail-closed rejection guards or
documentation, not model inputs.

The architecture debt was explicit separation, not a hidden M1 market import:

- `worker/sports_market_interface.py` remains a legacy M3/M4 surface
  (`ModelOutputRecord`, `MarketInputRecord`, `JoinedEdgeRecord`, `compute_edge`).
- `worker/sports_research.py` remains a legacy M3/M4/M5 research surface
  (`sportsbook_consensus`, `remove_vig`, `compute_clv`,
  `sportsbook_vs_prediction_market`, `cross_venue_response_lag`).

Neither legacy surface is imported by M1. The new package makes the intended
ownership and interfaces machine-checkable.

## Tests and smoke evidence

```text
MODULAR_BOUNDARY_TESTS=4 passed
FOCUSED_REGRESSION_SLICE=39 passed
LONG_SIMULATOR_REGRESSIONS=2 passed in 144.91s
M1_LIVE_SMOKE=PASS, 8 deterministic football-only paths
```

The M1 smoke emitted a valid immutable output with game id `TEST_GAME`,
simulation count `8`, and a content hash. No external market payload was
provided.

## Simulator-only health

The available V21 Week-1 football artifact reports 16 scored games:

```text
SCORE_MAE=9.964984375
TEAM_SCORE_BIAS=-5.330765625
BRIER_HOME_WIN=0.2459138125
MEAN_CRPS_HOME=7.09867140625
MEAN_CRPS_AWAY=7.198599515625
PIT_MEAN=0.660109375
P10-P90_COVERAGE=0.9375
```

Top simulator-only failure modes remain diagnostic, not promotion claims:

1. Team-score level bias (V21 Week-1 bias `-5.3308`).
2. Total-score underproduction (model mean `38.7760` versus actual mean `49.4375`).
3. PIT location bias (`0.6601`).
4. Historical V3 player/joint diagnostic QB zero predictions (`74.5%`; repair gate pending).
5. Historical V3 player/joint diagnostic QB underproduction (QB MAE/RMSE
   `182.19/208.06`; diagnostic only).

No market, EV, PnL, closing-line, or profitability metric is part of the M1
health surface.

## Blocker and next action

```text
REAL_BLOCKER=SPORTS_NOVA_V3_ENGINE_REPAIR_VALIDATION_V1.json is still pending;
player/joint empirical certification remains closed.
NEXT_SINGLE_ACTION=Run the authorized narrow engine-repair validation on the
exact hash-verified causal cohort, then audit the emitted validation artifact.
```

This reset does not create V23, reopen sealed OOS, retune thresholds, or
promote the pending player/joint diagnostic.
