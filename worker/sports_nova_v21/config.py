from worker.sports_nova_v3.config import RNG_ALGORITHM

MODEL_VERSION = "sports_nova_v21.drive_block.extra_point_fix.1"

# UNVALIDATED_DEFAULT, same category and same caveat as the existing flat
# `fg_rate=.08` in worker/sports_nova_v3/distributions.py: recalled as a
# widely-cited NFL league constant (post-2015 rule change moved PAT
# placement to the 15-yard line, make rate ~94% since), but NOT verified
# against this project's own causal panel -- checked and confirmed the
# panel (NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet) carries no PAT/kicking
# columns at all, so there is no in-project data to fit or check this
# against. It is not fit to 2026 Week1 outcomes either way. Sensitivity:
# at ~2.8 team-TDs/game, the gap between 0.90 and 0.98 is ~0.22 expected
# points/team-game -- well inside this engine's ~10-point score MAE, so
# this constant's imprecision does not materially change V21's result.
PAT_MAKE_RATE = 0.94
