"""SPORTS_NOVA_V1_3 challenger namespace.

Independent of worker/sports_nova_v1_1 (pace, REJECTED) and
worker/sports_nova_v1_2 (YPR shrinkage, REJECTED) -- one change per
experiment. Tests: worker/sports_nova_v3's scripts/sports_nova_v3_
player_joint_walkforward_v1.py hardcodes `sack_rate=.065` and
`scramble_rate=.07` as LITERAL CONSTANTS for every team in the league --
not derived from any real per-team data at all (unlike catch_rate/
yards_per_reception, which at least read a real, if unshrunk, per-player
ratio). Real per-team sack rate on training data (NFL_V3_DRIVE_BLOCK_
CANONICAL_V1.parquet, eval cohort excluded) ranges from 0.044 (IND) to
0.069 (SEA) -- a real, causally available, currently-unused signal.

Only simulator.py is forked, plus a new team_rates.py helper that computes
a causal (strictly-prior-games-only), training-derived per-team sack rate
directly from the drive-block panel using the SAME game_id-based causal
cutoff discipline (season/week strictly before the game being simulated)
as every other feature in this codebase. scramble_rate is left UNCHANGED
at .07: real scrambles are not separable from designed rushes in this
data source (a limitation already documented in
scripts/sports_nova_v3_team_passing_volume_calibration_v15.py's module
docstring), so there is no real signal available to substitute -- this
experiment does not invent one.

No identity-gate wiring: offline DEV/VALIDATION calibration only, same
caveat as the other v1_* namespaces.
"""
