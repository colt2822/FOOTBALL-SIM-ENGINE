"""V25 frozen policy constants.  Fixed BEFORE any V25 simulation output exists."""
import hashlib
import json

from worker.sports_nova_v23.config import RNG_ALGORITHM, PAT_MAKE_RATE  # noqa: F401  (re-exported, unchanged)

MODEL_VERSION = "sports_nova_v25.role_aware_redistribution.1"

REASON_CODES = ("OUT", "IR", "OFF_ROSTER", "OTHER_TEAM", "INACTIVE", "OTHER_EXPLICIT_INELIGIBLE")
REASON_PRECEDENCE = ("OTHER_TEAM", "OFF_ROSTER", "IR", "INACTIVE", "OTHER_EXPLICIT_INELIGIBLE", "OUT")
ROSTER_STATUS_REASON = {"RES": "IR", "INA": "INACTIVE", "DEV": "INACTIVE", "CUT": "OFF_ROSTER", "RET": "OFF_ROSTER",
                        "EXE": "OTHER_EXPLICIT_INELIGIBLE"}
ELIGIBLE_ROSTER_STATUS = "ACT"
DOUBTFUL_AVAILABILITY_WEIGHT = 0.025            # unchanged from V24 (frozen evidence: 3/367)
QUESTIONABLE_AVAILABILITY_WEIGHT = 1.0

SKILL_ROLES = ("QB", "RB", "FB", "WR", "TE")
RECIPIENT_ROLES = ("RB", "FB", "WR", "TE")      # may receive redistributed carry OR target mass
HELD_ROLE_QB = "QB"
NON_SKILL_ROLE = "NON_SKILL"                    # K/P/LS/OL/DL/LB/DB and anything unresolvable: held, never a recipient
ROLE_SOURCE_ORDER = ("roster_weekly.position", "roster_weekly.depth_chart_position", "PlayerState.position")

CONCENTRATION_THRESHOLD = {"carry": 0.60, "target": 0.30}      # unchanged V24 flags
THIN_HISTORY_MAX_GAMES = 8
THIN_HISTORY_MIN_POST_SHARE = 0.10
SHARE_TOLERANCE = 1e-10

ELIGIBILITY_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V25_ELIGIBILITY_POLICY", "VERSION": MODEL_VERSION,
    "ELIGIBLE_IF": "roster_weekly team == the team the M1 state assigns the player to AND roster status == ACT AND not OUT",
    "REASON_CODES": list(REASON_CODES), "REASON_PRECEDENCE": list(REASON_PRECEDENCE), "ROSTER_STATUS_REASON": ROSTER_STATUS_REASON,
    "UNKNOWN_STATUS_REASON": "OTHER_EXPLICIT_INELIGIBLE", "NOT_ON_ROSTER_REASON": "OFF_ROSTER", "DIFFERENT_TEAM_REASON": "OTHER_TEAM",
    "OUT_HANDLING": "OUT players are removed mass like any other ineligible player (V24 left them to V23's gate, whose renormalization also fed the QB)",
    "FORBIDDEN_INFERENCE": ["historical usage as a role oracle", "market data", "sportsbook lines", "future outcomes", "depth chart replacement"],
}
DOUBTFUL_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V25_DOUBTFUL_POLICY", "VERSION": MODEL_VERSION,
    "RULE": "unchanged from V24: eligible Doubtful player's share is multiplied by DOUBTFUL_AVAILABILITY_WEIGHT; the removed mass is then redistributed by the V25 role rule",
    "DOUBTFUL_AVAILABILITY_WEIGHT": DOUBTFUL_AVAILABILITY_WEIGHT, "QUESTIONABLE_AVAILABILITY_WEIGHT": QUESTIONABLE_AVAILABILITY_WEIGHT,
    "EVIDENCE": "V24_PREREG/DOUBTFUL_EVIDENCE.json (nflverse injuries 2019-2025, 3/367 Doubtful with opportunity, Wilson95 upper 2.4%)",
}
ROLE_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V25_ROLE_POLICY", "VERSION": MODEL_VERSION,
    "ROLE_SOURCE_ORDER": list(ROLE_SOURCE_ORDER), "SKILL_ROLES": list(SKILL_ROLES),
    "RULE": "role = first of ROLE_SOURCE_ORDER that is in SKILL_ROLES, else NON_SKILL; roster_weekly is the immutable panel source (pre-kickoff)",
    "RECIPIENT_ROLES": list(RECIPIENT_ROLES),
    "HELD_ROLES": [HELD_ROLE_QB, NON_SKILL_ROLE],
    "FORBIDDEN": ["QB as recipient of RB/WR/FB/TE mass", "non-skill role as recipient", "carry-role logic in the target pool", "player-name overrides", "caps"],
}
REDISTRIBUTION_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V25_REDISTRIBUTION_POLICY", "VERSION": MODEL_VERSION,
    "SCOPE": "carry pool and target pool of each team, independently; same rule, no cross-pool information",
    "STEP_1_NORMALIZE": "p_i = frozen_share_i / sum(all positive frozen shares of the team pool) -- the full pool including OUT/ineligible players "
                        "(residual bucket dropped exactly as V23's allocation drops it)",
    "STEP_2_HELD": "eligible held-role players (QB, NON_SKILL): post_i = p_i * availability_weight_i, i.e. exactly the share V23 gives them when nobody is removed; ineligible held players -> 0",
    "STEP_3_RECIPIENTS": "eligible RB/FB/WR/TE: base_i = p_i * availability_weight_i, then scaled by one common factor so that sum(post) over recipients = 1 - sum(held post); "
                         "i.e. all removed mass is redistributed pro-rata to eligible same-team same-pool recipients only",
    "STEP_4_HAND_OFF": "shares written with residual_share=0 and sum=1; the unchanged V23 allocation then draws (renormalization is the identity)",
    "UNCHANGED_POOL": "a team/kind pool with no removed mass (no ineligible/OUT positive share, no availability weight != 1) is returned untouched, bit-identical to V23",
    "ZERO_RECIPIENT_FALLBACK": "removed mass > 0 but no eligible recipient with positive share: raise NoEligibleRecipientsError; never hand the mass to a held role",
    "MASS": "sum(post)=1 within tolerance; held mass unchanged; removed mass == redistributed mass",
    "CAPS": "none", "OVERRIDES": "none",
    "SCRAMBLES": "QB scrambling is V23's separate selection.scrambles path and is untouched",
    "NOT_FIXED": "V23's inherited QB double count (designed-carry share incl. scrambles + scrambles credited again)",
    "CONCENTRATION_THRESHOLD": CONCENTRATION_THRESHOLD,
    "UNCHANGED": ["play volume", "play selection", "efficiency", "scoring", "QB selection (_qb_shares)", "RNG contract", "calibration", "V23", "Doubtful weight", "CIN Burrow fallback"],
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(canonical_json(policy)).hexdigest()


ELIGIBILITY_POLICY_HASH = policy_hash(ELIGIBILITY_POLICY)
DOUBTFUL_POLICY_HASH = policy_hash(DOUBTFUL_POLICY)
ROLE_POLICY_HASH = policy_hash(ROLE_POLICY)
REDISTRIBUTION_POLICY_HASH = policy_hash(REDISTRIBUTION_POLICY)
