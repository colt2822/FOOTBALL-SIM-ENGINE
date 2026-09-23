"""SPORTS_NOVA_V21 challenger namespace -- SPORTS_V21_EXTRA_POINT_FIX.

Tests exactly ONE change against V20 (the current structural champion,
itself a strict extension of frozen V18 + V19): every simulated touchdown
now draws an independent extra-point (PAT) attempt before being credited,
instead of crediting a bare 6 points with no kick at all.

Found via SPORTS_NOVA_V1_1 P4 (variance/tails audit,
SPORTS_NOVA_V1_1_P4_VARIANCE_TAILS_AUDIT.json): every simulated score in
the 2026 Week1 replay draws (V18/V19/V20 alike, all sharing this scoring
mechanism unchanged) is exactly divisible by 3 --
`worker/sports_nova_v3/simulator.py::_run_one` only ever adds 6 (TD) or 3
(FG) to `block_points`, with no PAT, no 2-point conversion, no safety.
Confirmed structural (not a fitted-parameter or small-sample artifact): the
model's own MODEL_TOTAL (mean 33.9 across 2026 Week1) sits ~11 points below
the pre-kickoff MARKET total line (mean 45.1, an independent estimate that
never saw the outcome), not just below the actual results (mean 49.4).

Fix scope (deliberately narrow -- one interpretable change): after each
touchdown (pass or rush), draw `rng.binomial(td_count, PAT_MAKE_RATE)` made
extra points and credit 1 point per make. `PAT_MAKE_RATE` is a fixed,
widely-published NFL league constant (post-2015-rule-change placement,
~94% make rate), NOT fit to this project's own causal panel or to the 2026
Week1 outcomes being evaluated -- adding it needs no outcome data, it is
derived from the rules of football the same way `fg_rate`'s existing flat
default already is. Two-point conversions are explicitly OUT of scope for
this pass (a much smaller, coaching-decision-dependent effect, ~1-2 points/
game vs PAT's ~5-6); left as a known, disclosed residual for a future
increment rather than bundled here.

Every other mechanism (pace/block-volume, allocation, efficiency, TD rates,
catch rate, QB identity/roster construction) is imported UNCHANGED from
worker.sports_nova_v19 / worker.sports_nova_v20 -- only `_run_one` and
`simulate_game` are forked (they own the scoring-accumulation lines), same
minimal-diff pattern as v1_1/v1_2/v1_3/v19.
"""
