"""V28 frozen constants.  Fixed BEFORE any V28 simulation output exists."""
import hashlib
import json

from worker.sports_nova_v23.config import RNG_ALGORITHM  # noqa: F401
from worker.sports_nova_v23.config import MODEL_VERSION as V23_MODEL_VERSION  # noqa: F401
from worker.sports_nova_v27_causal_fg_rate.config import (  # noqa: F401
    DRIVE_RELPATH, DRIVE_SHA256, ESTIMATOR_SHA256, MODEL_VERSION as V27_MODEL_VERSION)

BASE_VERSION = "sports_nova_v28.game_state_scoring"
from typing import NamedTuple

# variant number -> flags.  kappa scales the NOISE part of the per-game efficiency draw (env.*_efficiency = recent_form + kappa * N(0, .35)); recent_form (the team signal) is untouched.
# team_mult: TD odds x (team finishing-propensity intensity / league intensity), preserving the V23 team-specific pass_td_rate / rush_td_rate signal.
# clock_scale: multiplies a block's elapsed seconds so simulated possessions/team-game match the drive table (block-vs-drive definition gap: 11.594 vs 11.033).
CLOCK_SCALE = 11.594010416666666 / 11.033372711163615     # V27 sim blocks per team-game (sealed pilot arrays) / history drives per team-game (2020-2025 drive table): both known before V28.6


class Variant(NamedTuple):
    k: int
    ot: bool
    align: bool
    kappa: float
    team_mult: bool
    clock_scale: float
    eff_shrink: bool = False


_FLAGS = {1: (False, False, 1.0, False, 1.0), 2: (True, False, 1.0, False, 1.0), 3: (False, True, 1.0, False, 1.0), 4: (True, True, 1.0, False, 1.0),
          5: (False, False, 0.0, False, 1.0), 6: (True, False, 0.0, False, 1.0), 7: (False, False, 0.0, True, 1.0), 8: (False, False, 0.0, True, CLOCK_SCALE),
          9: (True, False, 0.0, True, CLOCK_SCALE), 10: (True, False, 0.0, True, 1.0), 11: (False, False, 0.0, True, CLOCK_SCALE, True), 12: (True, False, 0.0, True, CLOCK_SCALE, True)}
VARIANTS = {k: v[:3] for k, v in _FLAGS.items()}          # (ot, align, kappa) view kept for the older harness code
VERSIONS = {k: f"{BASE_VERSION}.{k}" for k in _FLAGS}
MODEL_VERSION = VERSIONS[1]                     # the candidate namespace start named by the brief

SCORE_TAG = 0x28            # separate SeedSequence stream for every scoring draw (TD / scorer / PAT / FG / OT coin)
OT_SECONDS_REG = 600        # regular-season overtime: 10 minutes
OT_SECONDS_POST = 90000     # postseason: untimed (played to a winner)
OT_MAX_BLOCKS = 200
PAT_NOTE = "PAT_MAKE_RATE imported unchanged from worker.sports_nova_v21.config"

PLAYER_RELPATH = "data/sports_nova_v3/validation_inputs/NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
PLAYER_SHA256 = "cb8e0ee8df6819961d154236dcbefe0d1399d05d50b1bf857a9624df9ab5388c"
TEAM_MULT_CLIP = (0.5, 2.0)

# ---- hazard specification (frozen before the first V28 run) ----
HAZARD_TARGET = "touchdowns > 0 on a historical drive block (offensive TD; the table's max is 1 per block)"
HAZARD_WINDOW = "identical to estimator A: blocks with season*100+week < simulated key AND season >= max(1999, simulated_season-5)"
YARD_KNOTS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
SUPPORT_PCT = (0.1, 99.9)   # yardage winsorization bounds = window percentiles (the historical support boundary)
DIFF_CLIP = 21
PLAYS_CLIP = 15
ALIGN_GRID = 2001           # quantile grid for the (POST_HOC) yardage-marginal alignment

HAZARD_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V28_HAZARD_POLICY", "VERSION": BASE_VERSION,
    "TARGET": HAZARD_TARGET, "WINDOW": HAZARD_WINDOW, "MODEL": "unpenalized logistic regression (sklearn lbfgs), design below",
    "DESIGN": {"yards": "clip(yards, window p0.1, window p99.9)/100 plus hinge max(y-k,0)/100 for k in YARD_KNOTS",
               "plays": "clip(plays,0,15)/10 plus indicators plays==1, plays==2, plays==3",
               "score": "offense-perspective pre-block differential clipped to +-21, /10, plus its positive part",
               "clock": "indicators seconds-in-half<120, seconds-in-half<300, second-half-not-started (seconds_remaining<=1800 is 'second half')"},
    "YARD_KNOTS": list(YARD_KNOTS), "SUPPORT_PCT": list(SUPPORT_PCT), "DIFF_CLIP": DIFF_CLIP, "PLAYS_CLIP": PLAYS_CLIP,
    "NEVER_USED": ["final outcomes of the simulated game", "market prices", "same-week or later blocks", "player season totals after the cutoff"],
    "FG": "V27 causal estimator A, flat P(made FG | no TD AND plays>=3), unchanged",
}
TD_ALLOCATION_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V28_TD_ALLOCATION_POLICY", "VERSION": BASE_VERSION,
    "RULE": "one categorical draw over the block's realized scoring opportunities; weight = receptions_i*clip(pass_td_rate,.001,.2) for receivers, "
            "carries_j*clip(rush_td_rate,.0005,.15) for carriers, QB scrambles*clip(rush_td_rate,.0005,.15) added to the QB's rush weight; pass TD credits receiver + QB pass_tds, "
            "rush TD credits the carrier only; PAT drawn once per TD via PAT_MAKE_RATE; TD with zero total weight is converted to no-TD (counted)",
    "NO_NAME_EXCEPTIONS": True,
}
OVERTIME_POLICY = {
    "SCHEMA": "SPORTS_NOVA_V28_OVERTIME_POLICY", "VERSION": BASE_VERSION,
    "RULES": "NFL modified sudden death (both teams get one possession; then first score wins); regular season 10-minute period then TIE; postseason played to a winner (no final ties); "
             "coin toss 50/50 on the scoring stream; game-ending TD gets no PAT; final score includes overtime points",
    "POSTSEASON_RULE": "week > 17 for seasons <= 2020, week > 18 for seasons >= 2021, from game_id (documented approximation; the 2026 slate is regular season)",
    "MARKET_MAPPING": "UNRESOLVED: no local Kalshi NFL rules_primary text exists; totals/team-totals/winner are exported both including and excluding overtime",
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


HAZARD_POLICY_HASH = hashlib.sha256(canonical_json(HAZARD_POLICY)).hexdigest()
TD_ALLOCATION_POLICY_HASH = hashlib.sha256(canonical_json(TD_ALLOCATION_POLICY)).hexdigest()
OVERTIME_POLICY_HASH = hashlib.sha256(canonical_json(OVERTIME_POLICY)).hexdigest()
EXTRA_TEAM_STATS = ("reg_score", "ot_score", "went_ot", "tds", "fgs", "td_pass", "td_rush", "td_qb_rush", "td_unattributable")


def variant_of(model_version: str) -> Variant:
    for k, v in VERSIONS.items():
        if v == model_version:
            return Variant(k, *_FLAGS[k])
    raise ValueError("unfrozen simulation version")


def is_postseason(season: int, week: int) -> bool:
    return week > (17 if season <= 2020 else 18)
