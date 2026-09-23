"""V24 opportunity repair: move ineligible players' frozen historical share into `residual_share`.

The unchanged V23 allocation (worker/sports_nova_v23/allocation.py) drops the residual bucket and renormalizes over the surviving
same-team players, so the redistribution is pro-rata to their existing frozen historical shares -- no new draw, no new weights,
same RNG contract.  This module edits ONLY OpportunityShares in the PregameState and reports what that did.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from worker.sports_nova_v3.allocation import _shares
from worker.sports_nova_v19.simulator import _qb_shares, _state_hash
from .config import (
    CONCENTRATION_THRESHOLD, SHARE_TOLERANCE, THIN_HISTORY_MAX_GAMES, THIN_HISTORY_MIN_POST_SHARE)
from .eligibility import PlayerEligibility, RosterSnapshot, game_eligibility


@dataclass
class RepairAudit:
    eligibility: dict[str, PlayerEligibility]
    rows: list[dict] = field(default_factory=list)          # one per (team, kind, player) with pre>0 or post>0
    summaries: list[dict] = field(default_factory=list)     # one per (team, kind)
    zero_survivor_fallbacks: list[str] = field(default_factory=list)
    changed: bool = False
    raw_state_hash: str = ""
    repaired_state_hash: str = ""
    qb_inputs_unchanged: bool = True

    def excluded(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for e in self.eligibility.values():
            if not e.eligible:
                out.setdefault(e.reason, []).append({"PLAYER_ID": e.player_id, "TEAM": e.team_id, "ALL_REASONS": list(e.all_reasons)})
        return out


def _normalized(state, team_id: str, kind: str) -> dict[str, float]:
    ids, values, _ = _shares(state, team_id, kind)          # the exact vector V23's allocation draws from
    total = float(values.sum())
    return {pid: float(v) / total for pid, v in zip(ids, values)} if total > 0 else {}


def _flags(kind: str, post: float, delta: float, sample: float) -> list[str]:
    flags = []
    if post > CONCENTRATION_THRESHOLD[kind]:
        flags.append("CONCENTRATION_ABOVE_THRESHOLD")
    if sample < THIN_HISTORY_MAX_GAMES and post >= THIN_HISTORY_MIN_POST_SHARE and delta > SHARE_TOLERANCE:
        flags.append("THIN_HISTORY_ROLE_GROWTH")
    return flags


def repair_pregame_state(state, snapshot: RosterSnapshot):
    """Return (repaired_state, audit).  The input object itself is returned when nothing changes."""
    elig = game_eligibility(state, snapshot)
    player_by_id = {p.player_id: p for p in state.players}
    audit = RepairAudit(eligibility=elig, raw_state_hash=_state_hash(state))
    team_updates = {}
    for team in (state.home, state.away):
        updates = {}
        for kind in ("target", "carry"):
            src = getattr(team, f"{kind}_shares")
            new_values, changed = [], False
            for pid, share in zip(src.player_ids, src.shares):
                e = elig[pid]
                # OUT-only exclusions stay with the existing V23 availability gate (state untouched)
                if share > 0 and not e.eligible and player_by_id[pid].availability != "OUT":
                    new_values.append(0.0)
                    changed = True
                elif share > 0 and e.eligible and e.availability_weight != 1.0:
                    new_values.append(share * e.availability_weight)
                    changed = True
                else:
                    new_values.append(share)
            if changed:
                updates[f"{kind}_shares"] = src.model_copy(update={
                    "shares": tuple(new_values), "residual_share": max(0.0, 1.0 - sum(new_values))})
        if updates:
            team_updates[team.team_id] = team.model_copy(update=updates)
    repaired = state
    if team_updates:
        repaired = state.model_copy(update={"home": team_updates.get(state.home.team_id, state.home),
                                            "away": team_updates.get(state.away.team_id, state.away)})
        audit.changed = True
    audit.repaired_state_hash = _state_hash(repaired)

    for team in (repaired.home, repaired.away):
        for kind in ("target", "carry"):
            pre, post = _normalized(state, team.team_id, kind), _normalized(repaired, team.team_id, kind)
            src_before = getattr(state.home if state.home.team_id == team.team_id else state.away, f"{kind}_shares")
            src_after = getattr(team, f"{kind}_shares")
            if pre and not post:
                audit.zero_survivor_fallbacks.append(f"{team.team_id}:{kind}")
            for pid in sorted(set(pre) | set(post)):
                e, p = elig[pid], player_by_id[pid]
                sample = float(p.uncertainty.effective_sample_size)
                b, a = pre.get(pid, 0.0), post.get(pid, 0.0)
                audit.rows.append({
                    "GAME_ID": state.game_id, "TEAM": team.team_id, "KIND": kind, "PLAYER_ID": pid, "POSITION": p.position,
                    "PLAYER_HISTORY_SAMPLE": sample, "PRE_REPAIR_SHARE": b, "POST_REPAIR_SHARE": a, "SHARE_DELTA": a - b,
                    "ELIGIBLE": e.eligible, "REASON_CODE": e.reason, "ALL_REASONS": list(e.all_reasons),
                    "AVAILABILITY_WEIGHT": e.availability_weight, "UNCERTAINTY_FLAG": _flags(kind, a, a - b, sample)})
            audit.summaries.append({
                "TEAM": team.team_id, "KIND": kind,
                "SUM_SHARE_PLUS_RESIDUAL_BEFORE": float(sum(src_before.shares) + src_before.residual_share),
                "SUM_SHARE_PLUS_RESIDUAL_AFTER": float(sum(src_after.shares) + src_after.residual_share),
                "EFFECTIVE_SUM_BEFORE": float(sum(pre.values())), "EFFECTIVE_SUM_AFTER": float(sum(post.values())),
                "REMOVED_MASS": float(sum(max(0.0, pre.get(k, 0.0) - post.get(k, 0.0)) for k in set(pre) | set(post))),
                "REDISTRIBUTED_MASS": float(sum(max(0.0, post.get(k, 0.0) - pre.get(k, 0.0)) for k in set(pre) | set(post))),
                "N_SURVIVORS": len(post), "N_PRE": len(pre)})
    team_ids = (state.home.team_id, state.away.team_id)
    audit.qb_inputs_unchanged = _qb_snapshot(state, team_ids) == _qb_snapshot(repaired, team_ids)
    return repaired, audit


def _qb_snapshot(s, team_ids):
    return [(tuple(map(str, ids)), tuple(map(float, shares))) for ids, shares in (_qb_shares(s, t) for t in team_ids)]


def assert_repair_invariants(state, repaired, audit: RepairAudit) -> None:
    for s in audit.summaries:
        if abs(s["SUM_SHARE_PLUS_RESIDUAL_AFTER"] - 1.0) > SHARE_TOLERANCE:
            raise AssertionError(f"share+residual mass not conserved: {s}")
        if s["N_SURVIVORS"] and abs(s["EFFECTIVE_SUM_AFTER"] - 1.0) > SHARE_TOLERANCE:
            raise AssertionError(f"effective mass not 1: {s}")
        if abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) > SHARE_TOLERANCE and s["N_SURVIVORS"]:
            raise AssertionError(f"removed mass != redistributed mass: {s}")
    for team in (repaired.home, repaired.away):
        allowed = {p.player_id for p in repaired.players if p.team_id == team.team_id}
        for src in (team.target_shares, team.carry_shares):
            for pid, share in zip(src.player_ids, src.shares):
                e = audit.eligibility[pid]
                if pid not in allowed:
                    raise AssertionError(f"wrong-team share: {pid}")
                if share > 0 and not e.eligible and next(p for p in repaired.players if p.player_id == pid).availability != "OUT":
                    raise AssertionError(f"ineligible positive share: {pid} {e.reason}")
    if not audit.qb_inputs_unchanged:
        raise AssertionError("QB selection inputs changed")
