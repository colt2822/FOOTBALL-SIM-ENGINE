"""SPORTS_NOVA_V20 -- isolated `make_state` fork: roster-assignment (Case B)
injection layer.

SN3_V20_ROSTER_ASSIGNMENT_FIX
------------------------------
scripts/sports_nova_v3_player_joint_walkforward_v1.py::make_state builds each
team's roster from `prior` (completed player-game rows) via TWO stacked
filters: `prior.TEAM == tid` AND `PLAYER_ID.map(latest_team) == tid` (see
that module's docstring in `make_state`). A player who has been traded,
signed, or drafted onto a team but has not yet completed a game FOR THAT
TEAM in the causal panel satisfies neither filter and can never appear on
that team's roster at all -- not a `_qb_shares` eligibility problem (V19's
scope), a `make_state`-roster-construction problem (this fork's scope).
Measured on the 2026 Week-1 replay (SPORTS_NOVA_V20_CASE_B_TRIAGE.json):
of V19's 6 remaining QB-identity mismatches, exactly 3 (MIA, LV, NYJ) are
this defect with a CONFIRMED, CORRECT depth-chart resolution already
available; the other 3 (ATL, MIN, SEA) have a wrong upstream depth-chart
signal and are NOT reachable by any roster-construction fix (see report).

Fix: reuse the SAME already-DEV-validated CONFIRMED depth-chart resolution
V19 already computes (worker.sports_nova_v19.simulator._qb_shares' override
input) as a roster-completion signal. If a CONFIRMED resolution names a QB
for (game_id, team) who is NOT present on that team's constructed roster,
inject a single PlayerState for him -- using his own most-recent team's
game history for features when he has any (a new-team player, not a
neutral-priors player), or minimal defaults when he has none (a rookie).
If that same player is ALSO sitting on the OTHER team's roster (a traded
player can still appear on his OLD team, via the mirror image of this exact
defect -- e.g. 2025 Week 1 PIT@NYJ, where Aaron Rodgers (PIT-bound) and
Justin Fields (NYJ-bound) each swapped teams and each initially resolved
onto the OTHER team's stale roster), that stale appearance is removed: a
CONFIRMED resolution is a more authoritative, more recent current-team
signal for that specific player than the historical last-known-team lookup,
and PregameState requires globally-unique player identity across both
rosters. Removal only ever touches a player who is himself one of this
game's injected candidates -- never a bystander.

Strict-extension invariant (mirrors V19's own inert-when-absent property):
injection fires ONLY when the resolved player is absent from the roster
`make_state` already built. If he is already present (V19's Case A), this
fork changes nothing -- byte-for-byte identical PregameState to the
unmodified make_state. Validated on DEV (2025 Week 1, n=32 team-slots,
scripts/sports_nova_v20_dev_triage.py): 8/32 slots are injection-eligible
(6 ROWS_OTHER_TEAM_ONLY + 2 NO_ROWS_ANYWHERE), all 8 recover the correct
actual starter, 0 false injections, and the other 24 slots are provably
untouched (gate never fires on PRESENT_IN_ROSTER).

Known, pre-declared limitation (not a new failure mode): if the CONFIRMED
resolution itself is wrong (ATL, MIN, SEA on the 2026 Week-1 test -- an
upstream nflverse depth-chart data-quality issue, not a code defect; see
SPORTS_NOVA_V20_CASE_B_TRIAGE.json), injection will faithfully complete the
roster with the WRONG player, so `_qb_shares`' override (which only checks
presence, not correctness) will pick him. This is not a regression: V19's
override already has this exact property for Case-A slots with a wrong
resolution. It converts "wrong pick, unreachable" into "different wrong
pick, now reachable" -- still a mismatch either way, never new damage to a
previously-correct slot, because the gate is presence, and any previously-
correct slot's resolved player is (by definition) already the one they
picked, hence already present.

Everything except the following injection block is character-for-character
identical to scripts/sports_nova_v3_player_joint_walkforward_v1.py::make_state
(imported and delegated to via the parent module `wf3` below rather than
copy-pasted, so there is exactly one diff surface).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "sports_nova_v3_player_joint_walkforward_v1_V20_BASE",
    ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py")
wf3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf3)

from worker.sports_nova_v3.schemas import PlayerState, Uncertainty
from worker.sports_nova_v3.pregame_state import build_pregame_state
from worker.sports_nova_v3.schemas import PregameState as _PregameStateForAssembly


def _resolve_injection_candidate(prior, resolutions, game_id, tid, roster_ids, ev):
    """Return a PlayerState to inject for `tid`, or None if no injection applies.

    Gate: CONFIRMED resolution for (game_id, tid) AND resolved player is
    absent from `roster_ids` (the roster `make_state` already built for
    this team from THIS team's own filtered rows). Absence is the only
    gate -- correctness of the resolution is not and cannot be checked here
    (that is exactly the information make_state does not have pregame).
    """
    if not resolutions:
        return None
    rec = resolutions.get((game_id, tid))
    if rec is None or rec.get("STATUS") != "CONFIRMED":
        return None
    rid = rec.get("QB_ID")
    if not rid or rid in roster_ids:
        return None

    own_rows = prior[prior.PLAYER_ID == rid]
    if own_rows.empty:
        # NO_ROWS_ANYWHERE: no causal history at all (rookie / undrafted).
        # Minimal, non-informative defaults; harmless because `_qb_shares`'
        # override makes him the sole candidate whenever this fires, so his
        # feature values do not compete against anyone.
        return PlayerState(player_id=str(rid), team_id=tid, position="QB",
            availability="UNKNOWN", identity_evidence=ev, features=(),
            uncertainty=Uncertainty(effective_sample_size=1.0, personnel_unknown=True))

    # ROWS_OTHER_TEAM_ONLY: use his own real history, wherever it is, with
    # the SAME recency window (last 4 games) make_state already uses for
    # every other QB, so this player is not treated more or less generously
    # than an incumbent would be.
    recent = own_rows.sort_values(["SEASON", "WEEK"]).tail(4)
    recent_n = max(1, len(recent))
    recent_rate = float(recent.PASS_ATTEMPTS.sum()) / recent_n
    fs = (wf3.feat("pass_rate", recent_rate, ev),
          wf3.feat("games_since_last_team_game", 0.0, ev))
    return PlayerState(player_id=str(rid), team_id=tid, position="QB",
        availability="UNKNOWN", identity_evidence=ev, features=fs,
        uncertainty=Uncertainty(effective_sample_size=float(recent_n), personnel_unknown=True))


def make_state_v20(g, prior, current_time, positions, resolutions=None):
    """Delegates to the unmodified V3 `make_state` for everything except one
    additive step: completing a team's QB roster from a CONFIRMED, causal,
    pre-kickoff identity resolution when the resolved player is absent.

    `resolutions`: dict[(game_id, team_id) -> resolution record], same shape
    and same CONFIRMED/CONFIDENCE/temporal-provenance contract as
    worker.sports_nova_v19.simulator.set_identity_resolutions already
    enforces (this function does not re-validate that contract; the caller
    is expected to have built it via that same validated path).
    """
    season, week, away, home = wf3.game_parts(g)
    cutoff = current_time - wf3.timedelta(seconds=1)
    last = cutoff - wf3.timedelta(seconds=1)
    ev = wf3.evidence(last, wf3.PLAYER_SHA)

    base_state = wf3.make_state(g, prior, current_time, positions)
    if not resolutions:
        return base_state

    injected = []
    for tid in (home, away):
        roster_ids = {p.player_id for p in base_state.players if p.team_id == tid}
        candidate = _resolve_injection_candidate(prior, resolutions, g, tid, roster_ids, ev)
        if candidate is not None:
            injected.append(candidate)

    if not injected:
        return base_state

    # A traded/signed player can still be sitting on his OLD team's roster
    # (that team's own `latest_team` lookup is stale for exactly this player
    # -- the same mechanism, just observed from the other side of the trade).
    # A CONFIRMED resolution is a more authoritative, more recent
    # current-team signal for THIS player than that historical lookup, so
    # his stale other-team appearance is dropped, not left duplicated
    # (PregameState requires globally-unique player identity, and a player
    # cannot correctly be on both rosters in the same game). This never
    # touches any player who is not one of the injected candidates.
    injected_ids = {c.player_id for c in injected}
    kept_base_players = tuple(p for p in base_state.players if p.player_id not in injected_ids)
    all_players = kept_base_players + tuple(injected)

    # A removed player's team may have already credited him a nonzero
    # target/carry share (a mobile QB can have real historical RUSH_ATTEMPTS
    # share on his old team). Fold that mass into residual_share rather than
    # dropping it, so `sum(shares) + residual_share == 1` still holds -- his
    # old team's true accounted-for opportunity total does not change, it is
    # simply no longer attributed to a named player who is not on that
    # roster this game.
    def _strip(opp_shares):
        keep = [(pid, sh) for pid, sh in zip(opp_shares.player_ids, opp_shares.shares)
                if pid not in injected_ids]
        if len(keep) == len(opp_shares.player_ids):
            return opp_shares  # nothing to remove -- share/residual mass unaffected
        removed_mass = sum(sh for pid, sh in zip(opp_shares.player_ids, opp_shares.shares)
                            if pid in injected_ids)
        return opp_shares.model_copy(update={
            "player_ids": tuple(pid for pid, _ in keep),
            "shares": tuple(sh for _, sh in keep),
            "residual_share": opp_shares.residual_share + removed_mass,
        })

    home_state = base_state.home.model_copy(update={
        "target_shares": _strip(base_state.home.target_shares),
        "carry_shares": _strip(base_state.home.carry_shares)})
    away_state = base_state.away.model_copy(update={
        "target_shares": _strip(base_state.away.target_shares),
        "carry_shares": _strip(base_state.away.carry_shares)})

    payload = base_state.model_dump()
    payload["players"] = tuple(p.model_dump() for p in all_players)
    payload["home"] = home_state.model_dump()
    payload["away"] = away_state.model_dump()
    return build_pregame_state(game_id=g, as_of=cutoff,
        feature_store={"game": payload}, identity_registry={"registry_sha256": "0" * 64})
