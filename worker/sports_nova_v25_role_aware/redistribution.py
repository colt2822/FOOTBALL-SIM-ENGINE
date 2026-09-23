"""V25 role-aware redistribution of removed opportunity mass.

For each team x {carry, target} pool (same rule, no cross-pool information):
  p_i     = frozen share / sum(all positive frozen shares)            (full pool, OUT and ineligible players included)
  held    = QB and NON_SKILL roles: post = p_i * availability_weight if eligible else 0   (never receive removed mass)
  recip.  = eligible RB/FB/WR/TE: post = p_i * w_i * c, one common c so that sum(post) == 1
The unchanged V23 allocation then draws from these shares (residual_share = 0, sum = 1, so its renormalization is the identity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from worker.sports_nova_v3.allocation import _shares
from worker.sports_nova_v19.simulator import _qb_shares, _state_hash
from .config import (
    CONCENTRATION_THRESHOLD, RECIPIENT_ROLES, SHARE_TOLERANCE, THIN_HISTORY_MAX_GAMES, THIN_HISTORY_MIN_POST_SHARE)
from .eligibility import PlayerEligibility, RosterSnapshot, game_eligibility


class NoEligibleRecipientsError(RuntimeError):
    """Removed mass exists in a pool but no eligible RB/FB/WR/TE recipient holds a positive frozen share."""


@dataclass
class RepairAudit:
    eligibility: dict[str, PlayerEligibility]
    rows: list[dict] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)
    changed: bool = False
    raw_state_hash: str = ""
    repaired_state_hash: str = ""
    qb_inputs_unchanged: bool = True

    def excluded(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for e in self.eligibility.values():
            if not e.eligible:
                out.setdefault(e.reason, []).append({"PLAYER_ID": e.player_id, "TEAM": e.team_id, "ROLE": e.role, "ALL_REASONS": list(e.all_reasons)})
        return out


def _normalized(state, team_id: str, kind: str) -> dict[str, float]:
    ids, values, _ = _shares(state, team_id, kind)          # the exact vector V23's allocation draws from
    total = float(values.sum())
    return {pid: float(v) / total for pid, v in zip(ids, values)} if total > 0 else {}


def role_aware_values(player_ids, shares, elig: dict[str, PlayerEligibility]):
    """Pure function.  Returns (post_values aligned to player_ids, p aligned, removed_mass, changed).  Raises NoEligibleRecipientsError."""
    total = float(sum(s for s in shares if s > 0))
    if total <= 0:
        return tuple(float(s) for s in shares), tuple(0.0 for _ in shares), 0.0, False
    p = [float(s) / total if s > 0 else 0.0 for s in shares]
    changed = any(pi > 0 and (not elig[pid].eligible or elig[pid].availability_weight != 1.0) for pid, pi in zip(player_ids, p))
    if not changed:
        return tuple(float(s) for s in shares), tuple(p), 0.0, False
    held_post, base = {}, {}
    for pid, pi in zip(player_ids, p):
        if pi <= 0 or not elig[pid].eligible:
            continue
        w = elig[pid].availability_weight
        if elig[pid].role in RECIPIENT_ROLES:
            base[pid] = pi * w
        else:
            held_post[pid] = pi * w
    recipient_target = 1.0 - sum(held_post.values())
    base_sum = sum(base.values())
    if recipient_target > SHARE_TOLERANCE and base_sum <= 0.0:
        raise NoEligibleRecipientsError("removed mass has no eligible RB/FB/WR/TE recipient with a positive frozen share")
    c = recipient_target / base_sum if base_sum > 0 else 0.0
    post = tuple(held_post.get(pid, base.get(pid, 0.0) * c) for pid in player_ids)
    removed = 1.0 - sum(pi * elig[pid].availability_weight for pid, pi in zip(player_ids, p) if elig[pid].eligible)
    return post, tuple(p), removed, True


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
    team_updates, pool_info = {}, {}
    for team in (state.home, state.away):
        updates = {}
        for kind in ("target", "carry"):
            src = getattr(team, f"{kind}_shares")
            post, p, removed, changed = role_aware_values(src.player_ids, src.shares, elig)
            pool_info[(team.team_id, kind)] = (p, removed)
            if changed:
                updates[f"{kind}_shares"] = src.model_copy(update={"shares": tuple(post), "residual_share": 0.0})
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
            src_before = getattr(state.home if state.home.team_id == team.team_id else state.away, f"{kind}_shares")
            src_after = getattr(team, f"{kind}_shares")
            p_vec, removed = pool_info[(team.team_id, kind)]
            pfull = dict(zip(src_before.player_ids, p_vec))
            v23 = _normalized(state, team.team_id, kind)                       # what frozen V23 draws from today
            post = _normalized(repaired, team.team_id, kind)                   # what V25 hands to the V23 draw
            surv = {pid: pfull[pid] * elig[pid].availability_weight for pid in pfull if pfull[pid] > 0 and elig[pid].eligible}
            surv_sum = sum(surv.values())
            v24 = {pid: v / surv_sum for pid, v in surv.items()} if surv_sum > 0 else {}     # V24-equivalent pro-rata over all survivors
            for pid in sorted(set(v23) | set(post) | set(v24)):
                e, pl = elig[pid], player_by_id[pid]
                sample = float(pl.uncertainty.effective_sample_size)
                a, b = post.get(pid, 0.0), v23.get(pid, 0.0)
                audit.rows.append({
                    "GAME_ID": state.game_id, "TEAM": team.team_id, "KIND": kind, "PLAYER_ID": pid, "POSITION": pl.position, "ROLE": e.role,
                    "ROLE_CLASS": "RECIPIENT" if e.role in RECIPIENT_ROLES else "HELD", "PLAYER_HISTORY_SAMPLE": sample,
                    "PRE_FULL_POOL_SHARE": pfull.get(pid, 0.0), "V23_EFFECTIVE_SHARE": b, "V24_EQUIVALENT_SHARE": v24.get(pid, 0.0), "V25_POST_SHARE": a,
                    "DELTA_VS_V23": a - b, "DELTA_VS_V24": a - v24.get(pid, 0.0),
                    "ELIGIBLE": e.eligible, "REASON_CODE": e.reason, "ALL_REASONS": list(e.all_reasons), "AVAILABILITY_WEIGHT": e.availability_weight,
                    "UNCERTAINTY_FLAG": _flags(kind, a, a - pfull.get(pid, 0.0), sample)})
            held_pre = sum(pfull[pid] * elig[pid].availability_weight for pid in pfull if pfull[pid] > 0 and elig[pid].eligible and elig[pid].role not in RECIPIENT_ROLES)
            held_post = sum(post.get(pid, 0.0) for pid in post if elig[pid].role not in RECIPIENT_ROLES)
            audit.summaries.append({
                "TEAM": team.team_id, "KIND": kind,
                "PRE_REMOVAL_TOTAL": float(sum(pfull.values())), "POST_REDISTRIBUTION_TOTAL": float(sum(post.values())),
                "SUM_SHARE_PLUS_RESIDUAL_AFTER": float(sum(src_after.shares) + src_after.residual_share),
                "REMOVED_MASS": float(removed), "REDISTRIBUTED_MASS": float(sum(max(0.0, post.get(k, 0.0) - pfull.get(k, 0.0) * elig[k].availability_weight)
                                                                              for k in post if elig[k].role in RECIPIENT_ROLES)),
                "HELD_MASS_PRE": float(held_pre), "HELD_MASS_POST": float(held_post),
                "MASS_TO_HELD_ROLES": float(sum(max(0.0, post.get(k, 0.0) - pfull.get(k, 0.0) * elig[k].availability_weight) for k in post if elig[k].role not in RECIPIENT_ROLES)),
                "V23_QB_MASS": float(sum(v for k, v in v23.items() if elig[k].role == "QB")), "V24_QB_MASS": float(sum(v for k, v in v24.items() if elig[k].role == "QB")),
                "V25_QB_MASS": float(sum(v for k, v in post.items() if elig[k].role == "QB")),
                "N_RECIPIENTS": sum(1 for k in post if elig[k].role in RECIPIENT_ROLES and post[k] > 0), "CHANGED": bool(removed > SHARE_TOLERANCE or src_before is not src_after)})
    team_ids = (state.home.team_id, state.away.team_id)
    audit.qb_inputs_unchanged = _qb_snapshot(state, team_ids) == _qb_snapshot(repaired, team_ids)
    return repaired, audit


def _qb_snapshot(s, team_ids):
    return [(tuple(map(str, ids)), tuple(map(float, shares))) for ids, shares in (_qb_shares(s, t) for t in team_ids)]


def assert_repair_invariants(state, repaired, audit: RepairAudit) -> None:
    tol = 1e-9
    for s in audit.summaries:
        if abs(s["POST_REDISTRIBUTION_TOTAL"] - s["PRE_REMOVAL_TOTAL"]) > tol:
            raise AssertionError(f"pool mass not conserved (pre_removal_total != post_redistribution_total): {s}")
        if abs(s["SUM_SHARE_PLUS_RESIDUAL_AFTER"] - 1.0) > 1e-6:
            raise AssertionError(f"share+residual != 1: {s}")
        if abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) > tol:
            raise AssertionError(f"removed mass != redistributed mass: {s}")
        if s["MASS_TO_HELD_ROLES"] > tol:
            raise AssertionError(f"a held role (QB/non-skill) received redistributed mass: {s}")
        if abs(s["HELD_MASS_POST"] - s["HELD_MASS_PRE"]) > tol and s["CHANGED"]:
            raise AssertionError(f"held mass changed: {s}")
    for team in (repaired.home, repaired.away):
        allowed = {p.player_id for p in repaired.players if p.team_id == team.team_id}
        for src in (team.target_shares, team.carry_shares):
            for pid, share in zip(src.player_ids, src.shares):
                e = audit.eligibility[pid]
                if pid not in allowed:
                    raise AssertionError(f"wrong-team share: {pid}")
                if share > 0 and not e.eligible and repaired is not state:
                    raise AssertionError(f"ineligible positive share: {pid} {e.reason}")
    for r in audit.rows:
        if r["ROLE_CLASS"] == "HELD" and r["V25_POST_SHARE"] > r["PRE_FULL_POOL_SHARE"] + tol:
            raise AssertionError(f"held-role share increased: {r['PLAYER_ID']} {r['KIND']}")
    if not audit.qb_inputs_unchanged:
        raise AssertionError("QB selection inputs changed")
