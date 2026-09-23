"""V26 active-skill state completion.  Runs on the M1 PregameState BEFORE V25's redistribution.

Pure with respect to its inputs (state, roster snapshot, the prior panel slice make_state used): no IO, no market data, no outcomes, no player-name logic.
The feature and share formulas below are a line-for-line restatement of scripts/sports_nova_v3_player_joint_walkforward_v1.make_state (which is not edited);
`verify_machinery(...)` proves on every game that the restatement reproduces the state's OWN players and shares before anything is added.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from worker.sports_nova_v23.simulator import _state_hash
from worker.sports_nova_v3.schemas import Feature, OpportunityShares, PlayerState, PregameState, Uncertainty
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot
from .config import (
    ACTION_ADDED, ACTION_ADDED_ZERO_SHARE, ACTION_BLOCKED_COLD_START, ACTION_BLOCKED_DUPLICATE, DEFAULT_FEATURES, RECIPIENT_ROLES, SHARE_BASIS_COUNTERFACTUAL,
    SHARE_BASIS_OFFICIAL, TOP_N_SKILL_POOL)

SHARE_COLUMNS = {"target": "TARGETS", "carry": "RUSH_ATTEMPTS"}
_TOL = 1e-12


class MachineryMismatchError(RuntimeError):
    """The restated make_state share/feature formulas did not reproduce the state's own values: completion aborts (fail closed)."""


@dataclass
class CompletionResult:
    state: PregameState
    raw_state_hash: str
    completed_state_hash: str
    rows: list = field(default_factory=list)          # one dict per expected-but-absent player (Phase 2 detail)
    teams: dict = field(default_factory=dict)         # team -> counts
    changed: bool = False


def expected_active_skill(roster: RosterSnapshot, team_id: str, known_out=frozenset()) -> dict:
    """gsis_id -> role for every ACT RB/FB/WR/TE on `team_id` that is not OUT.  Uses only the pre-kickoff roster snapshot."""
    out = {}
    for pid, (t, status) in roster.roster.items():
        if t != team_id or status != "ACT" or pid in known_out or roster.game_status.get(pid) == "OUT":
            continue
        pos, depth = roster.positions.get(pid, (None, None))
        role = next((c for c in (pos, depth) if c in RECIPIENT_ROLES), None)
        if role is not None:
            out[pid] = role
    return out


def latest_team_map(prior: pd.DataFrame) -> dict:
    return (prior.sort_values(["SEASON", "WEEK"]).drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["TEAM"].to_dict())


def _mode_position(rows: pd.DataFrame) -> str:
    m = rows.POSITION.mode()
    pos = m.iat[0] if not m.empty else "OTHER"
    return pos if pos in ("QB", "RB", "WR", "TE") else "OTHER"


def _feats(rows: pd.DataFrame, ev) -> tuple:
    """make_state's three per-player features from that player's TEAM==tid rows (defaults when there is no volume)."""
    targets, rec = float(rows.TARGETS.sum()), float(rows.RECEPTIONS.sum())
    carries, rush_y, rec_y = float(rows.RUSH_ATTEMPTS.sum()), float(rows.RUSH_YARDS.sum()), float(rows.RECEIVING_YARDS.sum())

    def feat(name, value):
        return Feature(name=name, value=None if value is None or not np.isfinite(value) else float(value), unit="model_unit", evidence=ev)
    return (feat("catch_rate", (rec / targets) if targets else DEFAULT_FEATURES["catch_rate"]),
            feat("yards_per_reception", (rec_y / rec) if rec else DEFAULT_FEATURES["yards_per_reception"]),
            feat("yards_per_carry", (rush_y / carries) if carries else DEFAULT_FEATURES["yards_per_carry"]))


def _team_total(prior: pd.DataFrame, latest: dict, tid: str, col: str) -> float:
    q = prior[prior.TEAM == tid]
    mask = q.PLAYER_ID.map(latest).fillna("") == tid
    return max(0.0, float(q.loc[mask, col].sum()))


def _volume(prior_team_rows: pd.DataFrame, pid: str, col: str) -> float:
    return max(0.0, float(prior_team_rows[prior_team_rows.PLAYER_ID == pid][col].sum()))


def verify_machinery(state: PregameState, prior: pd.DataFrame, latest: dict | None = None) -> dict:
    """Prove the restated formulas reproduce the state's own shares (every player, both pools) and features (every non-QB player with tid rows)."""
    latest = latest if latest is not None else latest_team_map(prior)
    checked, max_share_err, feat_checked = 0, 0.0, 0
    for team in (state.home, state.away):
        tid = team.team_id
        q = prior[prior.TEAM == tid]
        for kind, col in SHARE_COLUMNS.items():
            src = team.target_shares if kind == "target" else team.carry_shares
            total = _team_total(prior, latest, tid, col)
            if total <= 0:
                continue
            for pid, s in zip(src.player_ids, src.shares):
                got = _volume(q, pid, col) / total
                max_share_err = max(max_share_err, abs(got - float(s)))
                checked += 1
        for p in state.players:
            if p.team_id != tid or p.position == "QB":
                continue
            rows = q[q.PLAYER_ID == p.player_id]
            if rows.empty:
                continue
            ev = p.identity_evidence
            mine = _feats(rows, ev)
            got = {f.name: f.value for f in mine}
            have = {f.name: f.value for f in p.features if f.name in got}
            for k, v in got.items():
                if v is None or have.get(k) is None or abs(v - have[k]) > 1e-9:
                    raise MachineryMismatchError(f"feature {k} for {p.player_id}: rebuilt {v} != state {have.get(k)}")
            if abs(float(len(rows)) - p.uncertainty.effective_sample_size) > 1e-9:
                raise MachineryMismatchError(f"effective_sample_size for {p.player_id}")
            feat_checked += 1
    if max_share_err > 1e-9:
        raise MachineryMismatchError(f"share formula does not reproduce the state's shares (max err {max_share_err})")
    return {"SHARES_CHECKED": checked, "MAX_SHARE_ERR": max_share_err, "FEATURE_ROWS_CHECKED": feat_checked}


def _classify_and_build(state, tid, pid, role, prior, latest, in_state_any, template_ev, rank):
    hist = prior[prior.PLAYER_ID == pid]
    qt = hist[hist.TEAM == tid]
    rec = {"TEAM": tid, "PLAYER_ID": pid, "ROLE": role, "HIST_ROWS_ALL_TEAMS": int(len(hist)), "HIST_ROWS_THIS_TEAM": int(len(qt)),
           "LATEST_TEAM_IN_HISTORY": latest.get(pid),
           "TARGETS_THIS_TEAM": float(qt.TARGETS.sum()), "CARRIES_THIS_TEAM": float(qt.RUSH_ATTEMPTS.sum()),
           "TARGETS_ALL_TEAMS": float(hist.TARGETS.sum()), "CARRIES_ALL_TEAMS": float(hist.RUSH_ATTEMPTS.sum())}
    if pid in in_state_any:
        rec.update({"REASON": "IDENTITY_JOIN", "SUB_REASON": "OTHER_TEAM_IN_STATE", "STATE_TEAM": in_state_any[pid].team_id, "ACTION": ACTION_BLOCKED_DUPLICATE})
        return rec, None
    if hist.empty:
        rec.update({"REASON": "NO_HISTORY", "SUB_REASON": "NO_PANEL_ROWS", "ACTION": ACTION_BLOCKED_COLD_START})
        return rec, None
    if latest.get(pid) != tid:
        rec.update({"REASON": "IDENTITY_JOIN", "SUB_REASON": f"LATEST_TEAM={latest.get(pid)}", "ACTION": ACTION_ADDED_ZERO_SHARE if len(qt) == 0 or (rec["TARGETS_THIS_TEAM"] + rec["CARRIES_THIS_TEAM"]) == 0 else ACTION_ADDED})
    else:
        truncated = rank is not None and rank[1] > TOP_N_SKILL_POOL
        rec.update({"REASON": "HISTORY_FILTER" if truncated else "STATE_BUILD_OMISSION",
                    "SUB_REASON": (f"TOP_{TOP_N_SKILL_POOL}_TRUNCATION(ahead={rank[0]},ahead_or_tied_incl_self={rank[1]})" if truncated else "UNEXPLAINED"),
                    "ACTION": ACTION_ADDED})
    src = qt if len(qt) else hist
    pl = PlayerState(player_id=pid, team_id=tid, position=_mode_position(src), availability="UNKNOWN", identity_evidence=template_ev,
                     features=_feats(qt, template_ev), uncertainty=Uncertainty(effective_sample_size=float(len(qt)), personnel_unknown=True))
    return rec, pl


def _pool_rank_keys(prior: pd.DataFrame, latest: dict, tid: str) -> pd.DataFrame:
    """make_state's truncation ordering inputs: per non-QB player with TEAM==tid & latest_team==tid rows -> (targets, carries, pass_att)."""
    q = prior[(prior.TEAM == tid) & (prior.PLAYER_ID.map(latest).fillna("") == tid)]
    st = q.groupby("PLAYER_ID", as_index=False).agg(targets=("TARGETS", "sum"), carries=("RUSH_ATTEMPTS", "sum"), pass_att=("PASS_ATTEMPTS", "sum"),
                                                     POSITION=("POSITION", lambda x: x.mode().iat[0] if not x.mode().empty else "OTHER"))
    return st[st.POSITION != "QB"]


def _rank_bounds(keys: pd.DataFrame, pid: str) -> tuple | None:
    """(players strictly ahead, players ahead-or-tied INCLUDING this one) under make_state's (targets, carries, pass_att) descending sort.
    Ties (e.g. many zero-volume players) are ordered arbitrarily by make_state's sort, so a player is provably cut by `.head(40)` iff even the tied
    block alone cannot fit inside the top 40, i.e. the second number > 40."""
    row = keys[keys.PLAYER_ID == pid]
    if row.empty:
        return None
    t, c, a = float(row.targets.iat[0]), float(row.carries.iat[0]), float(row.pass_att.iat[0])
    gt = (keys.targets > t) | ((keys.targets == t) & (keys.carries > c)) | ((keys.targets == t) & (keys.carries == c) & (keys.pass_att > a))
    eq = (keys.targets == t) & (keys.carries == c) & (keys.pass_att == a)
    return int(gt.sum()), int(gt.sum() + eq.sum())


def complete_state(state: PregameState, roster: RosterSnapshot, prior: pd.DataFrame, *, known_out=frozenset(),
                   share_basis: str = SHARE_BASIS_OFFICIAL, verify: bool = True) -> CompletionResult:
    """Add absent ACTIVE RB/FB/WR/TE players to the state.  Returns the input object itself when nothing is added (bit-identical hash)."""
    if share_basis not in (SHARE_BASIS_OFFICIAL, SHARE_BASIS_COUNTERFACTUAL):
        raise ValueError(share_basis)
    raw_hash = _state_hash(state)
    latest = latest_team_map(prior)
    if verify:
        verify_machinery(state, prior, latest)
    in_state_any = {p.player_id: p for p in state.players}
    add_players, rows, teams, new_team_states = [], [], {}, {}
    for team in (state.home, state.away):
        tid = team.team_id
        expected = expected_active_skill(roster, tid, known_out)
        present = {p.player_id for p in state.players if p.team_id == tid}
        missing = sorted(pid for pid in expected if pid not in present)
        team_players = [p for p in state.players if p.team_id == tid]
        template_ev = team_players[0].identity_evidence if team_players else team.identity_evidence
        keys = _pool_rank_keys(prior, latest, tid)
        added = []
        for pid in missing:
            rec, pl = _classify_and_build(state, tid, pid, expected[pid], prior, latest, in_state_any, template_ev,
                                          _rank_bounds(keys, pid))
            rows.append(rec)
            if pl is not None:
                added.append(pl)
        teams[tid] = {"ACTIVE_SKILL_EXPECTED": len(expected), "PRESENT_BEFORE": len(expected) - len(missing), "MISSING": len(missing), "ADDED": len(added),
                      "BLOCKED": len(missing) - len(added)}
        if not added:
            continue
        add_players += added
        q = prior[prior.TEAM == tid]
        updates = {}
        for kind, col in SHARE_COLUMNS.items():
            src = team.target_shares if kind == "target" else team.carry_shares
            total = _team_total(prior, latest, tid, col)
            if total <= 0:
                raise MachineryMismatchError(f"{tid} {kind}: empty pool cannot be completed")
            ids = list(src.player_ids)
            vols = [_volume(q, p, col) for p in ids]
            add_ids, add_vols = [p.player_id for p in added], []
            denom = total
            for p in added:
                v_tid = _volume(q, p.player_id, col)
                if share_basis == SHARE_BASIS_COUNTERFACTUAL:
                    v_all = _volume(prior, p.player_id, col)
                    in_mask = latest.get(p.player_id) == tid
                    denom += v_all - (v_tid if in_mask else 0.0)
                    add_vols.append(v_all)
                else:
                    add_vols.append(v_tid)
            shares = [v / denom for v in vols + add_vols]
            if share_basis == SHARE_BASIS_OFFICIAL:
                for old, new in zip(src.shares, shares[:len(ids)]):
                    if abs(old - new) > 1e-9:
                        raise MachineryMismatchError(f"{tid} {kind}: existing share changed {old} -> {new}")
            resid = 1.0 - float(sum(shares))
            if -1e-9 < resid < 0.0:
                resid = 0.0
            updates[f"{kind}_shares"] = OpportunityShares(player_ids=tuple(ids + add_ids), shares=tuple(map(float, shares)), residual_share=resid, evidence=src.evidence)
        new_team_states[tid] = team.model_copy(update=updates)
    if not add_players:
        return CompletionResult(state, raw_hash, raw_hash, rows, teams, False)
    update = {"home": new_team_states.get(state.home.team_id, state.home), "away": new_team_states.get(state.away.team_id, state.away),
              "players": tuple(state.players) + tuple(add_players)}
    # The adapter flips OUT players with an unvalidated model_copy, so the raw state itself violates PregameState's "OUT player has positive share" rule
    # (V23's allocation filters OUT at draw time).  Re-run every OTHER validator (identity uniqueness, share-team membership, evidence causality) on a copy
    # with OUT reset to UNKNOWN; the object handed on keeps the OUT flags exactly as the adapter set them.
    check = dict(update, players=tuple(p.model_copy(update={"availability": "UNKNOWN"}) if p.availability == "OUT" else p for p in update["players"]))
    type(state)(**{**{k: getattr(state, k) for k in type(state).model_fields}, **check})
    completed = state.model_copy(update=update)
    return CompletionResult(completed, raw_hash, _state_hash(completed), rows, teams, True)
