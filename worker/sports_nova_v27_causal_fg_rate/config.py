"""V27 frozen constants.  Fixed BEFORE any V27 simulation output exists."""
import hashlib
import json

from worker.sports_nova_v26_active_skill_state.config import MODEL_VERSION as V26_MODEL_VERSION  # noqa: F401  (re-exported, unchanged)
from worker.sports_nova_v25_role_aware.config import MODEL_VERSION as V25_MODEL_VERSION  # noqa: F401
from worker.sports_nova_v23.config import MODEL_VERSION as V23_MODEL_VERSION, RNG_ALGORITHM  # noqa: F401

MODEL_VERSION = "sports_nova_v27.causal_fg_rate.1"

# Estimator A -- pinned.  These are the values recorded in FG_CAUSAL_ABLATION_V1_PREREG.json (written before any ablation simulation).
ESTIMATOR_RELPATH = "scripts/sports_nova_fg_causal_estimator_v1.py"
ESTIMATOR_SHA256 = "cc3a9f20a6761f6f18202d3c8a47af54f207834c0085d6936dbe7957041bd569"
DRIVE_RELPATH = "data/sports_nova_v3/validation_inputs/NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet"
DRIVE_SHA256 = "47b71f2b40866e0d1dcba2191ff0fd14a601777548b357076a1179e57017e50d"

# The unfitted default fit_distributions([], as_of) returns; used ONLY by the regression harness (test T1), never as a fallback.
V23_DEFAULT_FG_RATE = 0.08

CAUSAL_FG_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V27_CAUSAL_FG_POLICY", "VERSION": MODEL_VERSION,
    "ESTIMAND": "P(made FG | block scored no TD AND plays>=3) == params.fg_rate at worker/sports_nova_v23/simulator.py:114 (block_points==0 and plays>=3 and rng.random()<fg_rate)",
    "ESTIMATOR": "estimator A: made FIELD_GOAL / blocks with touchdowns==0 AND plays>=3 (possession_result normalized upper-case, spaces->underscores)",
    "WINDOW": "blocks with season*100+week < simulated key AND season >= max(1999, simulated_season-5): five prior season labels plus current-season-to-date where available",
    "SEASON_WEEK_SOURCE": "game_id 'SEASON_WEEK_AWAY_HOME' (never as_of: historical as_of is a surrogate kickoff)",
    "NOT_APPLIED": ["all-history window", "REG-only filter", "points_scored==3 numerator", "postseason exclusion", "smoothing", "window optimisation", "denominator change"],
    "SUBSTITUTION": "params = fit_distributions([], as_of); params = dataclasses.replace(params, fg_rate=causal_rate); unchanged V23 _run_one",
    "FAIL_CLOSED": "estimator file/hash mismatch, drive table hash mismatch, empty window, leakage (key>=cutoff or own game in window), non-finite or out-of-range rate -> CausalFGError; never 0.08",
    "CAVEAT": "row-level publication timestamps are unavailable (table: event NOT_EXPOSED, publication RELEASE_LEVEL_ONLY_NOT_ROW_LEVEL); causality is week-granular event order. "
              "Bounded effect: dropping the most recent 1/4/17 weeks moves the rate by at most 0.0008/0.0019/0.0034 over the 60 pilot cutoffs.",
    "UNCHANGED": ["V26 state completion", "V25 role-aware redistribution", "V25/V26 roster eligibility", "QB selection", "play volume", "completion model", "TD model",
                  "turnover model", "scoring-event ordering", "random seeds / RNG contract", "market isolation", "M2"],
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


CAUSAL_FG_POLICY_HASH = hashlib.sha256(canonical_json(CAUSAL_FG_POLICY)).hexdigest()
