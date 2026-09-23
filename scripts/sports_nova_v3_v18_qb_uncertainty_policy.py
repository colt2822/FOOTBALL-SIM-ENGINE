"""SPORTS_NOVA_V18_QB_UNCERTAINTY_POLICY_AND_WEEKLY_REFRESH -- frozen policy.

Small, shared module (no simulator/model code touched) that other prospective
scripts import to answer two questions the same way every time:

  1. Is the identity-resolution artifact fresh enough to predict from at all?
     (STALE_IDENTITY_GATE)
  2. For a given team, does its current identity resolution license a primary
     point prediction, or must it be excluded?
     (PRIMARY_COHORT_IDENTITY_RULE / UNCERTAIN_QB_POLICY)

Decision recorded here, not re-litigated per script: EXCLUDE_FROM_PRIMARY,
not PREDECLARED_SCENARIO_MIXTURE, for UNCERTAIN teams.

Why EXCLUDE over MIXTURE (checked against the frozen simulator before
deciding, not assumed): worker/sports_nova_v3/simulator.py's `_qb_shares`
already blends existing roster candidates by causal pass_rate *
effective_sample_size, sharpened by a power-2.5 exponent -- so ordinary
within-roster QB competition (e.g. a backup emerging) is already represented
in the simulated distribution before `_primary_qb` ever picks a name to
report. That existing mechanism does NOT cover the case actually driving
today's UNCERTAIN teams: `_qb_shares`' candidate pool (`ids`) is built from
the live panel's TEAM-filtered history, so a player with zero recorded games
for the CURRENT team (e.g. a mid-offseason trade) is structurally invisible
to it, however real his depth-chart/roster standing is. Building a
provably-causal mixture that adds such a player as a scenario would mean
injecting a roster member the frozen `make_state`/`_qb_shares` code never
sees today -- a change to what the frozen pipeline models, not just how its
output is reported, and "weights must come from causal pregame evidence"
deserves more validation than fits inside a same-night policy freeze. EXCLUDE
is the choice that costs nothing extra to satisfy every invariant (no model
change, no silent guess, no outcome-dependent choice -- it is made before any
of the three current UNCERTAIN teams' next game has kicked off).

This is a REPORTING-layer policy: it decides whether a per-team prediction is
written to predictions_live/, never how simulate_game() computes anything.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_DIR = ROOT / "data" / "sports_nova_v3" / "prospective" / "identity"
RESOLUTION_PATH = IDENTITY_DIR / "SPORTS_NOVA_V18_QB_IDENTITY_RESOLUTION_2026.json"
POLICY_PATH = IDENTITY_DIR.parent / "SPORTS_NOVA_V18_QB_UNCERTAINTY_POLICY.json"

# Chosen conservatively: the depth-chart source (validated in the identity-
# refresh mission) scrapes roughly daily and the injury report cycle is
# weekly-but-game-week-aligned; 24h caps staleness to "at most one missed
# daily depth-chart snapshot" without forcing an hourly rerun cadence this
# project has no infrastructure to guarantee.
FRESHNESS_MAX_AGE_HOURS = 24

PRIMARY_COHORT_IDENTITY_RULE = (
    "STATUS == 'CONFIRMED' (causal panel box-score leader and current "
    "timestamped depth-chart QB1 agree, and the depth-chart QB1 is not "
    "report_status Out/Doubtful/IR) -> use the frozen _primary_qb pick "
    "unchanged as the primary prediction. Empirical backing: 2025 backcheck "
    "on exactly this agreement condition measured 6.45% mismatch against "
    "the real box score (32/496) -- read as ~93-94%, not 100%."
)

UNCERTAIN_QB_POLICY = "EXCLUDE_FROM_PRIMARY"

MIXTURE_POLICY_IF_USED = None  # not adopted this mission; see module docstring for why


def load_resolution() -> dict:
    if not RESOLUTION_PATH.exists():
        raise SystemExit(f"BLOCKED_NO_IDENTITY_ARTIFACT: {RESOLUTION_PATH} does not exist; "
                          f"run sports_nova_v3_v18_current_identity_refresh.py first")
    return json.loads(RESOLUTION_PATH.read_text())


def enforce_fresh_identity(resolution: dict | None = None, now: datetime | None = None) -> dict:
    """Hard-blocks (SystemExit) if the identity artifact is older than
    FRESHNESS_MAX_AGE_HOURS. Returns the loaded resolution dict on success."""
    resolution = resolution if resolution is not None else load_resolution()
    now = now or datetime.now(timezone.utc)
    as_of_raw = resolution.get("AS_OF_UTC")
    if not as_of_raw:
        raise SystemExit("BLOCKED_STALE_IDENTITY_ARTIFACT: resolution artifact has no AS_OF_UTC")
    as_of = datetime.fromisoformat(str(as_of_raw))
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    age_hours = (now - as_of).total_seconds() / 3600.0
    if age_hours > FRESHNESS_MAX_AGE_HOURS:
        raise SystemExit(
            f"BLOCKED_STALE_IDENTITY_ARTIFACT: identity resolution is {age_hours:.1f}h old "
            f"(AS_OF_UTC={as_of_raw}), exceeds FRESHNESS_MAX_AGE_HOURS={FRESHNESS_MAX_AGE_HOURS}. "
            f"Re-run sports_nova_v3_v18_current_identity_refresh.py before predicting.")
    return resolution


def classify_team(team: str, resolution: dict) -> tuple[str, dict]:
    """Returns (COHORT, team_record). COHORT is 'PRIMARY' or
    'EXCLUDED_UNCERTAIN_IDENTITY'. Never returns a QB pick itself -- that
    stays the frozen simulator's job for PRIMARY teams."""
    rec = resolution.get("TEAMS", {}).get(team)
    if rec is None:
        return "EXCLUDED_UNCERTAIN_IDENTITY", {"TEAM": team, "STATUS": "NO_RESOLUTION_RECORD"}
    if rec.get("STATUS") == "CONFIRMED":
        return "PRIMARY", rec
    return "EXCLUDED_UNCERTAIN_IDENTITY", rec


def write_policy_artifact(extra: dict) -> dict:
    payload = {
        "MISSION": "SPORTS_V18_QB_UNCERTAINTY_POLICY_AND_WEEKLY_REFRESH",
        "FROZEN_AT_UTC": datetime.now(timezone.utc).isoformat(),
        "PRIMARY_COHORT_IDENTITY_RULE": PRIMARY_COHORT_IDENTITY_RULE,
        "UNCERTAIN_QB_POLICY": UNCERTAIN_QB_POLICY,
        "MIXTURE_POLICY_IF_USED": MIXTURE_POLICY_IF_USED,
        "STALE_IDENTITY_GATE": {
            "FRESHNESS_MAX_AGE_HOURS": FRESHNESS_MAX_AGE_HOURS,
            "ENFORCED_BY": "sports_nova_v3_v18_qb_uncertainty_policy.enforce_fresh_identity(), "
                           "called at the top of every prospective capture batch before any "
                           "team is classified or predicted",
        },
    }
    payload.update(extra)
    POLICY_PATH.write_text(json.dumps(payload, indent=2, default=str))
    return payload
