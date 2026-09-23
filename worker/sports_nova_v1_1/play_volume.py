from __future__ import annotations
import numpy as np
from worker.sports_nova_v3.distributions import sample_negative_binomial
from worker.sports_nova_v3.environment import EnvironmentDraw
from worker.sports_nova_v3.game_state import GameState

# SPORTS_NOVA_V1_1_HURRY_UP_PACE: worker/sports_nova_v3/environment.py draws a
# single pace_seconds value ONCE per simulation and every block uses it
# unchanged regardless of score/clock state. Real teams do not play at a
# constant tempo: NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet (1999-2025, the 60
# SPORTS_NOVA_V15 evaluation games excluded from this derivation) shows a
# trailing offense with <=300s remaining averages 15.4s/play, and <=60s
# remaining averages 8.6s/play, against a >900s-remaining league baseline of
# 24.6s/play -- a >2x tempo change the static single-draw pace cannot
# represent. SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_CALIBRATION.json's
# DOMINANT_STAGE is SIMULATION_DRIFT: the sim's own mean blocks, run through
# the analytic play-volume formula, would already match REAL_PASS_ATT (gap
# -0.30) -- but the actual stochastic sim mean pass attempts falls 1.23
# below that, i.e. the simulation itself produces fewer plays late in games
# than a constant-pace model implies. This multiplier is the mechanism most
# directly responsible for that: it does not touch pass_rate, sack_rate, or
# any allocation/identity logic, only how many plays a late-game block gets.
#
# Ratios below are TRAIN_MEAN_SEC_PER_PLAY(bucket) / BASELINE(24.613),
# BASELINE = mean sec_per_play for plays with >900s remaining, computed on
# the full drive-block dataset with the SPORTS_NOVA_V3_PHASE08_OOS_GAME_
# MANIFEST_V1.json 60-game evaluation cohort excluded (no eval-cohort
# tuning). Anchored at bucket midpoints, np.interp elsewhere (clamped).
_BASELINE_SEC_PER_PLAY = 24.613427463699175

_TRAILING_T = np.array([30, 90, 150, 240, 450, 750, 2250], dtype=float)
_TRAILING_RATIO = np.array([
    8.623345 / _BASELINE_SEC_PER_PLAY,
    12.421322 / _BASELINE_SEC_PER_PLAY,
    11.836303 / _BASELINE_SEC_PER_PLAY,
    15.420483 / _BASELINE_SEC_PER_PLAY,
    21.199404 / _BASELINE_SEC_PER_PLAY,
    24.131509 / _BASELINE_SEC_PER_PLAY,
    24.205148 / _BASELINE_SEC_PER_PLAY,
], dtype=float)

_NOT_TRAILING_T = np.array([30, 90, 150, 240, 450, 750, 2250], dtype=float)
_NOT_TRAILING_RATIO = np.array([
    14.076467 / _BASELINE_SEC_PER_PLAY,
    22.972360 / _BASELINE_SEC_PER_PLAY,
    19.388614 / _BASELINE_SEC_PER_PLAY,
    22.190409 / _BASELINE_SEC_PER_PLAY,
    28.071590 / _BASELINE_SEC_PER_PLAY,
    28.213717 / _BASELINE_SEC_PER_PLAY,
    24.920402 / _BASELINE_SEC_PER_PLAY,
], dtype=float)


def pace_multiplier(seconds_remaining: float, trailing: bool) -> float:
    """Data-derived tempo ratio (<1 = faster/more plays), function of
    real-clock time and whether the offense currently trails. Purely a
    function of already-simulated in-game state (never a future outcome),
    same causal footing as game_script.script_pass_rate's own use of
    differential + time."""
    t = np.clip(float(seconds_remaining), 0.0, 3600.0)
    if trailing:
        ratio = np.interp(t, _TRAILING_T, _TRAILING_RATIO)
    else:
        ratio = np.interp(t, _NOT_TRAILING_T, _NOT_TRAILING_RATIO)
    return float(np.clip(ratio, 0.25, 1.3))


def effective_pace_seconds(env: EnvironmentDraw, seconds_remaining: float, score_diff_for_offense: float) -> float:
    """The single source of truth for this block's tempo. MUST be reused for
    both how many plays get drawn (draw_block_volume) and how much clock a
    play consumes (worker/sports_nova_v1_1/simulator.py's `elapsed`
    calculation) -- computing the multiplier twice, or applying it to only
    one side, would draw hurry-up-sized play counts while still charging
    normal-tempo clock, which contradicts the mechanism (a first version of
    this function did exactly that; caught before shipping, see
    SPORTS_NOVA_V1_1_CHANGELOG.md)."""
    return env.pace_seconds * pace_multiplier(seconds_remaining, score_diff_for_offense < 0)


def draw_block_volume(state: GameState, env: EnvironmentDraw, rng: np.random.Generator,
                      *, dispersion: float = 8.0, effective_pace: float) -> int:
    remaining_blocks = max(1, 24 - state.block_index)
    expected = np.clip((60.0 / effective_pace) * 2.7, 2.0, 12.0)
    clock_cap = max(0, int(np.ceil(state.seconds_remaining / max(effective_pace, 1.0))))
    return sample_negative_binomial(expected, dispersion, rng, maximum=max(0, min(12, clock_cap)))
