"""V26 frozen policy constants.  Fixed BEFORE any V26 simulation output exists.

V26 changes ONE thing: active RB/FB/WR/TE players that the M1 state builder omitted are added to the state before V25's (unchanged) redistribution.
"""
import hashlib
import json

from worker.sports_nova_v25_role_aware.config import RECIPIENT_ROLES  # noqa: F401  (re-exported, unchanged)

MODEL_VERSION = "sports_nova_v26.active_skill_state_completion.1"

# Byte-identity of the V25 package V26 depends on (T4).  These are the hashes recorded in V25_PREREG at V25's freeze.
V25_PACKAGE_FILE_SHA256 = {
    "worker/sports_nova_v25_role_aware/__init__.py": "22515efc127a7dfb354b3a5e8e31cd8993a859070d1a686701f452b8d291f8ad",
    "worker/sports_nova_v25_role_aware/config.py": "4394058a2556e9c67ed4edad819d811af77ed4c7bcfd902865cae8524898685c",
    "worker/sports_nova_v25_role_aware/eligibility.py": "cc415581a92230fe0ddfa5fff371bb04850b194d6e0f009f8aa0f012e32c223a",
    "worker/sports_nova_v25_role_aware/redistribution.py": "d5ba0855b18cd85500c140a28a6be90e8eb775f483d65674dfc89ef3d51218b0",
    "worker/sports_nova_v25_role_aware/simulator.py": "d221299699fc3f19f3424b66d54915a2f3ee02dc4821f39a2748a73a174d67a9",
}
V25_HASH = "8f21ba304d330bea44a051dabcb066c3b7007aa6e9aa44f37c17fb8e0de73a04"

# missing-reason taxonomy (Phase 2)
REASON_CODES = ("IDENTITY_JOIN", "HISTORY_FILTER", "ROSTER_FILTER", "POSITION_FILTER", "STATE_BUILD_OMISSION", "NO_HISTORY", "OTHER")
ACTION_ADDED = "ADDED"
ACTION_ADDED_ZERO_SHARE = "ADDED_ZERO_SHARE_PORTABILITY_POLICY_REQUIRED"
ACTION_BLOCKED_COLD_START = "BLOCKED_COLD_START_POLICY_REQUIRED"
ACTION_BLOCKED_DUPLICATE = "BLOCKED_DUPLICATE_IDENTITY_OTHER_TEAM_IN_STATE"

SHARE_BASIS_OFFICIAL = "tid_only"          # the frozen make_state join: a player's volume counts only where TEAM == the team it is being allocated to
SHARE_BASIS_COUNTERFACTUAL = "portable"    # DIAGNOSTIC ONLY (never simulated, never emitted): a player's volume follows him across teams
DEFAULT_FEATURES = {"catch_rate": 0.64, "yards_per_reception": 10.0, "yards_per_carry": 4.2}   # make_state's own zero-volume defaults
TOP_N_SKILL_POOL = 40                       # make_state's `.head(40)` truncation of the non-QB pool (evidence only; V26 does not change it)

COMPLETION_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V26_COMPLETION_POLICY", "VERSION": MODEL_VERSION,
    "DEFECT": "make_state builds the player pool from historical panel rows keyed (TEAM == tid AND latest_team == tid), truncated to the top-40 non-QB rows by "
              "(targets, carries, pass_att); the current roster is never consulted, so ACTIVE RB/FB/WR/TE players are absent when they (a) have no panel rows "
              "(NO_HISTORY), (b) last played for another team (IDENTITY_JOIN), or (c) fall below the top-40 volume cut (HISTORY_FILTER)",
    "ACTIVE_SKILL_SET": "roster_weekly team == the team AND status == ACT AND role in RB/FB/WR/TE (roster position, then depth_chart_position) AND not OUT "
                        "(game-status OUT or panel M1_AVAILABILITY OUT); Questionable/Doubtful stay in the set and keep V25's existing availability weights",
    "ADD_RULE": "a missing set member with panel history is added to PlayerState through the SAME feature formulas and the SAME share formula as make_state "
                "(share = volume on TEAM==tid rows / current-roster team total); recomputed existing shares must reproduce the state's own shares exactly or the completion aborts",
    "COLD_START": "no cold-start/rookie prior exists in the V3/V19/V23 lineage (only sports_nova_v2/tests/test_v2.py::test_rookie_prior_falls_back_to_league, a different lineage). "
                  "A member with no panel rows is therefore BLOCKED (not added, no weight invented) and reported COLD_START_POLICY_REQUIRED",
    "CROSS_TEAM": "a member whose only volume is on another team gets the share the frozen tid-only machinery gives it (0), so it is present in state but carries no workload; "
                  "giving it portable volume is a new cross-team policy and is reported PORTABILITY_POLICY_REQUIRED with the at-stake mass quantified, never simulated",
    "DUPLICATE_IDENTITY": "PregameState forbids duplicate player identities: a member already in the OTHER team's state cannot be added; blocked and reported (its share on this team would be 0)",
    "UNCHANGED": ["V25 redistribution (byte-identical)", "QB selection (_qb_shares)", "QB scramble/rush logic", "play volume", "scoring", "total points", "variance",
                  "M2 calibration", "M3/M4 mirror", "market isolation", "settlement", "caps (none)", "player-name overrides (none)"],
    "NO_OVERRIDES": True, "CAPS": "none",
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(canonical_json(policy)).hexdigest()


COMPLETION_POLICY_HASH = policy_hash(COMPLETION_POLICY)
