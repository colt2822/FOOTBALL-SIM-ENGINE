"""V24 frozen policy constants.  Every value here is fixed BEFORE any V24 simulation output exists.

Derivations (all from pre-2026 data or already-existing code, never from slate outputs):
  * DOUBTFUL weight: nflverse injury reports 2019-2025 (REG, QB/RB/WR/TE/FB) joined to the causal player-game panel:
    3 of 367 Doubtful players recorded >=1 pass attempt / rush attempt / target (0/38 QB, 0/112 RB, 0/72 TE);
    Wilson 95% upper bound 2.4%, rounded up to 0.025.  Out players: 1/2302 (join sanity).  Questionable: 0.535 (left unchanged).
  * Concentration flags: the thresholds already used by scripts/sports_nova_sunday_m1_baseline_v1.py (OPPORTUNITY_CONCENTRATION).
"""
import hashlib
import json

from worker.sports_nova_v23.config import RNG_ALGORITHM, PAT_MAKE_RATE  # noqa: F401  (re-exported, unchanged)

MODEL_VERSION = "sports_nova_v24.roster_eligibility_fix.1"

REASON_CODES = ("OUT", "IR", "OFF_ROSTER", "OTHER_TEAM", "INACTIVE", "OTHER_EXPLICIT_INELIGIBLE")
# primary reason = first match in this order (structural status before the injury-report status); all matches are also recorded
REASON_PRECEDENCE = ("OTHER_TEAM", "OFF_ROSTER", "IR", "INACTIVE", "OTHER_EXPLICIT_INELIGIBLE", "OUT")
# nflverse roster_weekly `status`; anything not listed and not ACT fails closed to OTHER_EXPLICIT_INELIGIBLE
ROSTER_STATUS_REASON = {"RES": "IR", "INA": "INACTIVE", "DEV": "INACTIVE", "CUT": "OFF_ROSTER", "RET": "OFF_ROSTER",
                        "EXE": "OTHER_EXPLICIT_INELIGIBLE"}
ELIGIBLE_ROSTER_STATUS = "ACT"

DOUBTFUL_AVAILABILITY_WEIGHT = 0.025
QUESTIONABLE_AVAILABILITY_WEIGHT = 1.0          # unchanged existing M1 handling (UNKNOWN, full allocation)

CONCENTRATION_THRESHOLD = {"carry": 0.60, "target": 0.30}
THIN_HISTORY_MAX_GAMES = 8
THIN_HISTORY_MIN_POST_SHARE = 0.10
SHARE_TOLERANCE = 1e-10

ELIGIBILITY_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V24_ELIGIBILITY_POLICY", "VERSION": MODEL_VERSION,
    "ELIGIBLE_IF": "roster_weekly team == the team the M1 state assigns the player to AND roster status == ACT AND not OUT",
    "ROSTER_SOURCE": "nflverse roster_weekly_2026, week == game week (latest week <= game week), from the immutable panel sources",
    "OUT_SOURCE": "PlayerState.availability == OUT (existing panel flip) or the panel's pre-kickoff injury GAME_STATUS == OUT",
    "REASON_CODES": list(REASON_CODES), "REASON_PRECEDENCE": list(REASON_PRECEDENCE),
    "ROSTER_STATUS_REASON": ROSTER_STATUS_REASON, "UNKNOWN_STATUS_REASON": "OTHER_EXPLICIT_INELIGIBLE",
    "NOT_ON_ROSTER_REASON": "OFF_ROSTER", "DIFFERENT_TEAM_REASON": "OTHER_TEAM",
    "OUT_ONLY_EXCLUSIONS": "left to the existing V23 availability gate (state untouched) so existing OUT behaviour is preserved bit-for-bit",
    "FORBIDDEN_INFERENCE": ["historical usage", "market data", "sportsbook lines", "future outcomes", "depth chart replacement"],
}
DOUBTFUL_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V24_DOUBTFUL_POLICY", "VERSION": MODEL_VERSION,
    "RULE": "EXPECTED_AVAILABILITY_WEIGHT: an eligible Doubtful player's historical share is multiplied by DOUBTFUL_AVAILABILITY_WEIGHT before pro-rata renormalization",
    "DOUBTFUL_AVAILABILITY_WEIGHT": DOUBTFUL_AVAILABILITY_WEIGHT,
    "QUESTIONABLE_AVAILABILITY_WEIGHT": QUESTIONABLE_AVAILABILITY_WEIGHT,
    "EXISTING_SYSTEM": "no numeric Doubtful availability probability exists in the codebase; existing categorical precedent treats Doubtful as unavailable "
                       "(QB_RULE_V1 in sports_nova_sunday_panel/build.py, INJURY_OUT_STATUSES in sports_nova_v3_v18_current_identity_refresh.py); "
                       "the panel adapter maps Doubtful->UNKNOWN for skill positions (the defect)",
    "EVIDENCE": {"SAMPLE": "nflverse injuries 2019-2025 REG QB/RB/WR/TE/FB x causal player-game panel", "DOUBTFUL_N": 367, "DOUBTFUL_WITH_OPPORTUNITY": 3,
                 "RATE": 3 / 367, "WILSON95_UPPER": 0.02375, "OUT_N": 2302, "OUT_WITH_OPPORTUNITY": 1,
                 "METRIC": ">=1 pass attempt, rush attempt or target in the player-game panel (SNAPS is unpopulated)"},
    "LIMITATION": "deterministic expected-share weight, not a per-simulation Bernoulli: means are right, zero-inflation of a Doubtful player's own distribution is not modelled",
    "QUESTIONABLE": "unchanged: existing M1 behaviour (availability UNKNOWN, full allocation); observed play rate 0.535 does not justify a change inside this patch",
}
REDISTRIBUTION_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V24_REDISTRIBUTION_POLICY", "VERSION": MODEL_VERSION,
    "RULE": "zero the ineligible player's historical share and move that mass to residual_share; the unchanged V23 allocation drops the residual bucket and "
            "renormalizes over the surviving same-team players, i.e. pro-rata to their existing frozen historical shares",
    "RECIPIENTS": "game-eligible same-team players with positive frozen historical share only; no new players, no depth-chart or talent model",
    "MASS": "sum(shares)+residual_share == 1 preserved; effective draw distribution sums to 1 before and after",
    "ZERO_SURVIVOR_FALLBACK": "existing V3/V23 degenerate path (no listed player -> opportunities unattributed at team level); counted and reported, expected 0",
    "CAPS": "none: no share is capped; extreme redistribution is flagged only",
    "CONCENTRATION_THRESHOLD": CONCENTRATION_THRESHOLD, "CONCENTRATION_THRESHOLD_SOURCE": "existing OPPORTUNITY_CONCENTRATION flag in sports_nova_sunday_m1_baseline_v1.py",
    "THIN_HISTORY": {"MAX_GAMES": THIN_HISTORY_MAX_GAMES, "MIN_POST_SHARE": THIN_HISTORY_MIN_POST_SHARE, "RULE": "diagnostic flag only; share increased"},
    "CONTEXT_ONLY": "2019-2025 team-season max single-player share p99: carry 0.838, target 0.328",
    "UNCHANGED": ["play volume", "play selection", "efficiency", "scoring", "QB selection (_qb_shares)", "RNG contract", "calibration"],
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(canonical_json(policy)).hexdigest()


ELIGIBILITY_POLICY_HASH = policy_hash(ELIGIBILITY_POLICY)
REDISTRIBUTION_POLICY_HASH = policy_hash(REDISTRIBUTION_POLICY)
DOUBTFUL_POLICY_HASH = policy_hash(DOUBTFUL_POLICY)
