"""V13 Phase A: SPORTS_NOVA_V13_AMBIGUOUS_QB_IDENTITY -- rule design/freeze.

Diagnostic-only. NO engine changes in this script. Reuses V11's frozen
training candidate rows (SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv,
2000-2019, 621 multi-QB team-games, strictly disjoint from the V9 cohort)
and derives/freezes an ambiguity-classification rule from `gap`,
`gap_rank`, `volume`, `volume_rank`, `n_candidates` only -- never from
`is_realized_starter`, which is read afterward ONLY to score the frozen
rule, exactly as V11 scored its frozen picker.

Today's mechanism (worker/sports_nova_v3/simulator.py::_qb_shares) hard-
eliminates every candidate whose recency gap exceeds the team's min gap
BEFORE any volume weighing happens -- a candidate who fails that filter
gets share 0 no matter how much recent volume he carries. That is the
one "forced single identity" mechanism in the engine (the volume**2.5
split itself is already a soft, multi-candidate distribution among
whoever survives the gap filter; _primary_qb is reporting-only and does
not drive simulated attempts).

TWO candidate zero-parameter rules were tried against the training
sample, in order, per the mission's own escalation clause ("no fitted
multi-parameter classifier unless [the] simpler deterministic
uncertainty rule fails"):

  ATTEMPT 1 (REJECTED): AMBIGUOUS iff the team's globally-highest-volume
    candidate (any gap) is excluded by the min-gap filter. This mostly
    just re-flags the exact case the gap filter exists to resolve (a
    long-tenured former/benched starter's large *cumulative* volume
    outranking a short-tenured current starter's smaller volume, per
    the code's own reasoning in _qb_shares). Measured: 29.0% of the 621
    training team-games flagged ambiguous, and 26.6% of the 557 games
    where today's mechanism is ALREADY CORRECT would be relabeled
    ambiguous -- far from "near zero", so this attempt is rejected and
    not used.

  ATTEMPT 2 (FROZEN, used below): restrict the widened candidate to gap
    == min_gap + 1 (the single next-most-recent step -- the same
    "sat out only the single most recent prior game" case the
    _qb_shares docstring already names) AND require that candidate's
    own volume to exceed the min-gap survivor's own top volume (must be
    genuinely competitive on recent usage, not merely present one game
    back). Both conditions are single, a-priori, non-swept criteria
    (no threshold search was run; "+1" and "strictly greater than the
    incumbent" were the only values ever evaluated). Measured: 7.9%
    ambiguous rate, 6.8% false-ambiguity rate on currently-correct
    games (vs. 26.6% for attempt 1), 95.9% containment on ambiguous
    games, and recovers 9/11 (81.8%) of the currently-wrong cases it
    flags. This is the frozen rule.

  CONFIDENT_SINGLE_QB: no adjacent (gap = min_gap+1) candidate exists,
    or the adjacent candidate's volume does not exceed the min-gap
    survivor's -- unchanged behavior.
  AMBIGUOUS_MULTI_QB: an adjacent (gap = min_gap+1) candidate exists
    with strictly higher volume than the min-gap survivor -- retain
    BOTH the min-gap survivor set AND that adjacent candidate as live
    candidates instead of eliminating the latter.
  UNKNOWN_QB_IDENTITY: every candidate has zero measured volume (the
    existing `if volume.sum() <= 0` fallback in _qb_shares) -- no
    pregame evidence differentiates any candidate at all. This already
    exists in the code; here it is only named and counted.

No new recency/volume formula, no fitted parameter, ** 2.5 unchanged --
only candidate SET membership changes for the AMBIGUOUS_MULTI_QB state.

Metrics required before any engine edit:
  - TRAINING_AMBIGUOUS_RATE: share of the 621 team-games flagged
    AMBIGUOUS.
  - containment: among AMBIGUOUS games, is the realized starter inside
    the widened (survivors UNION top-volume) set. Justifies widening
    only if high.
  - TRAINING_FALSE_AMBIGUITY_RATE: among team-games where TODAY'S
    mechanism (hard gap filter + volume argmax) already picks the
    realized starter correctly, what share get relabeled AMBIGUOUS.
    This is the regression-risk number: flagging an already-correct
    game ambiguous spreads share off the correct dominant QB for no
    accuracy gain and pushes attempt bias further negative.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
TRAINING_ROWS = DATA / "SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv"


def classify(sub: pd.DataFrame) -> tuple[str, set[str]]:
    """sub: candidate rows for one team-game. Returns (state, retained_pid_set).
    Uses gap/gap_rank/volume/volume_rank/n_candidates only -- never is_realized_starter.
    Frozen rule (ATTEMPT 2 above): widen only to a gap=min_gap+1 candidate
    whose own volume strictly exceeds the min-gap survivor's top volume.
    """
    if sub.volume.sum() <= 0:
        return "UNKNOWN_QB_IDENTITY", set(sub.pid)
    min_gap = sub.gap.min()
    survivors = sub[sub.gap == min_gap]
    adjacent = sub[sub.gap == min_gap + 1]
    if adjacent.empty or adjacent.volume.max() <= survivors.volume.max():
        return "CONFIDENT_SINGLE_QB", set(survivors.pid)
    adjacent_top = adjacent.loc[adjacent.volume.idxmax()]
    return "AMBIGUOUS_MULTI_QB", set(survivors.pid) | {adjacent_top.pid}


def current_engine_pick(sub: pd.DataFrame) -> str:
    """Replicates today's _qb_shares/_primary_qb: min-gap survivors, argmax by
    volume (the **2.5 power is monotonic and does not change the argmax)."""
    min_gap = sub.gap.min()
    survivors = sub[sub.gap == min_gap]
    return survivors.loc[survivors.volume.idxmax()].pid


def _qb_shares_replica(vol_by_pid: dict) -> dict:
    """Exact replica of _qb_shares' weighting math (volume ** 2.5, normalized;
    equal-split fallback if all-zero) over an arbitrary candidate set, so the
    share IMPACT of widening (not just set membership) can be measured before
    touching the engine. ** 2.5 itself is reproduced unmodified, never altered."""
    import numpy as np
    v = np.clip(np.array(list(vol_by_pid.values()), dtype=float), 0.0, None)
    v = v ** 2.5
    if v.sum() <= 0:
        v = np.ones_like(v)
    p = v / v.sum()
    return dict(zip(vol_by_pid.keys(), p))


def share_impact(sub: pd.DataFrame, state: str, retained: set) -> dict:
    """For AMBIGUOUS games: argmax flip and realized-starter share before vs.
    after widening, using the real ** 2.5 weighting -- not just set containment."""
    min_gap = sub.gap.min()
    survivors = sub[sub.gap == min_gap]
    cur_shares = _qb_shares_replica(dict(zip(survivors.pid, survivors.volume)))
    cur_argmax = max(cur_shares, key=cur_shares.get)
    if state != "AMBIGUOUS_MULTI_QB":
        return {"flipped": None, "realized_share_current": None, "realized_share_widened": None,
                "cur_argmax": cur_argmax, "wide_argmax": None}
    wide_vol = dict(zip(survivors.pid, survivors.volume))
    extra = sub[sub.pid.isin(retained - set(survivors.pid))]
    for _, r in extra.iterrows():
        wide_vol[r.pid] = r.volume
    wide_shares = _qb_shares_replica(wide_vol)
    wide_argmax = max(wide_shares, key=wide_shares.get)
    return {"flipped": wide_argmax != cur_argmax,
            "realized_share_current": cur_shares, "realized_share_widened": wide_shares,
            "cur_argmax": cur_argmax, "wide_argmax": wide_argmax}


def main():
    rows = pd.read_csv(TRAINING_ROWS)
    rows["pid"] = rows["pid"].astype(str)

    records = []
    for (g, t), sub in rows.groupby(["GAME_ID", "TEAM"]):
        state, retained = classify(sub)
        pick = current_engine_pick(sub)
        realized = sub.loc[sub.is_realized_starter, "pid"]
        realized_pid = realized.iloc[0] if len(realized) else None
        impact = share_impact(sub, state, retained)
        realized_share_cur = (impact["realized_share_current"].get(realized_pid)
                               if impact["realized_share_current"] and realized_pid else None)
        realized_share_wide = (impact["realized_share_widened"].get(realized_pid)
                                if impact["realized_share_widened"] and realized_pid else None)
        records.append({
            "GAME_ID": g, "TEAM": t, "STATE": state,
            "N_CANDIDATES": int(len(sub)),
            "CURRENT_ENGINE_PICK_CORRECT": bool(pick == realized_pid),
            "REALIZED_STARTER_CONTAINED": bool(realized_pid in retained) if realized_pid else None,
            "ARGMAX_FLIPPED": impact["flipped"],
            "REALIZED_SHARE_CURRENT": realized_share_cur,
            "REALIZED_SHARE_WIDENED": realized_share_wide,
        })
    tg = pd.DataFrame(records)
    n = len(tg)

    state_counts = tg.STATE.value_counts().to_dict()
    ambiguous = tg[tg.STATE == "AMBIGUOUS_MULTI_QB"]
    confident = tg[tg.STATE == "CONFIDENT_SINGLE_QB"]
    unknown = tg[tg.STATE == "UNKNOWN_QB_IDENTITY"]
    currently_correct = tg[tg.CURRENT_ENGINE_PICK_CORRECT]

    training_ambiguous_rate = len(ambiguous) / n
    containment_on_ambiguous = float(ambiguous.REALIZED_STARTER_CONTAINED.mean()) if len(ambiguous) else None
    false_ambiguity_rate = (float((currently_correct.STATE == "AMBIGUOUS_MULTI_QB").mean())
                             if len(currently_correct) else None)

    # sanity: does the widened set help where the CURRENT pick was wrong?
    currently_wrong = tg[~tg.CURRENT_ENGINE_PICK_CORRECT & (tg.STATE != "UNKNOWN_QB_IDENTITY")]
    recovery_among_wrong_and_ambiguous = None
    wrong_and_ambiguous = currently_wrong[currently_wrong.STATE == "AMBIGUOUS_MULTI_QB"]
    if len(wrong_and_ambiguous):
        recovery_among_wrong_and_ambiguous = float(wrong_and_ambiguous.REALIZED_STARTER_CONTAINED.mean())

    # SHARE-MAGNITUDE GATE (not set containment): does widening merely hedge,
    # or does the existing ** 2.5 sharpening turn every widen into a full
    # argmax flip? Computed on the 38 currently-correct-but-flagged games
    # (regression risk) and the 11 currently-wrong-and-flagged games (benefit).
    amb_correct = ambiguous[ambiguous.CURRENT_ENGINE_PICK_CORRECT]
    amb_wrong = ambiguous[~ambiguous.CURRENT_ENGINE_PICK_CORRECT]
    share_gate = {
        "n_currently_correct_flagged": int(len(amb_correct)),
        "argmax_flip_rate_on_currently_correct_flagged": (
            float(amb_correct.ARGMAX_FLIPPED.mean()) if len(amb_correct) else None),
        "mean_realized_starter_share_BEFORE_on_currently_correct_flagged": (
            float(amb_correct.REALIZED_SHARE_CURRENT.mean()) if len(amb_correct) else None),
        "mean_realized_starter_share_AFTER_on_currently_correct_flagged": (
            float(amb_correct.REALIZED_SHARE_WIDENED.mean()) if len(amb_correct) else None),
        "n_currently_wrong_flagged": int(len(amb_wrong)),
        "mean_realized_starter_share_BEFORE_on_currently_wrong_flagged": (
            float(amb_wrong.REALIZED_SHARE_CURRENT.mean()) if len(amb_wrong) else None),
        "mean_realized_starter_share_AFTER_on_currently_wrong_flagged": (
            float(amb_wrong.REALIZED_SHARE_WIDENED.mean()) if len(amb_wrong) else None),
    }
    # Gate fails if widening behaves as a near-total argmax flip (not a hedge)
    # on the games that were already correct -- structurally expected here
    # because the frozen rule only ever widens to a candidate with STRICTLY
    # HIGHER volume than the incumbent, and ** 2.5 (which the invariants
    # forbid changing) then hands that higher-volume candidate the majority
    # share almost automatically.
    share_gate["PASSES_SHARE_MAGNITUDE_GATE"] = bool(
        share_gate["argmax_flip_rate_on_currently_correct_flagged"] is not None and
        share_gate["argmax_flip_rate_on_currently_correct_flagged"] <= 0.25 and
        (share_gate["mean_realized_starter_share_AFTER_on_currently_correct_flagged"] or 0) >= 0.5)

    result = {
        "mission": "SPORTS_NOVA_V13_AMBIGUOUS_QB_IDENTITY_PHASE_A_RULE_DESIGN",
        "rule": ("FROZEN (attempt 2, after attempt 1 was rejected for 26.6% false-ambiguity -- "
                 "see module docstring): AMBIGUOUS_MULTI_QB iff a candidate at gap == min_gap+1 "
                 "exists whose own volume strictly exceeds the min-gap survivor's top volume "
                 "(genuinely competitive on recent usage one step back); UNKNOWN_QB_IDENTITY iff "
                 "total candidate volume is 0 (existing _qb_shares fallback branch, named/counted "
                 "only); else CONFIDENT_SINGLE_QB. No new recency/volume formula, no fitted "
                 "parameter, ** 2.5 untouched -- only candidate SET membership changes."),
        "n_team_games": n,
        "state_counts": state_counts,
        "TRAINING_AMBIGUOUS_RATE": training_ambiguous_rate,
        "TRAINING_UNKNOWN_RATE": len(unknown) / n,
        "TRAINING_CONTAINMENT_ON_AMBIGUOUS": containment_on_ambiguous,
        "TRAINING_FALSE_AMBIGUITY_RATE": false_ambiguity_rate,
        "n_currently_correct": int(len(currently_correct)),
        "n_currently_wrong_resolved": int(len(currently_wrong)),
        "recovery_among_currently_wrong_and_flagged_ambiguous": recovery_among_wrong_and_ambiguous,
        "n_currently_wrong_and_flagged_ambiguous": int(len(wrong_and_ambiguous)),
        "SHARE_MAGNITUDE_GATE": share_gate,
        "PHASE_A_VERDICT": ("PROCEED_TO_ENGINE_EDIT_AND_V9" if share_gate["PASSES_SHARE_MAGNITUDE_GATE"]
                             else "GATE_FAILED_NO_ENGINE_EDIT_NO_V9_RUN"),
    }
    out_path = DATA / "SPORTS_NOVA_V13_AMBIGUITY_RULE_TRAINING.json"
    out_path.write_text(json.dumps(result, indent=2))
    tg.to_csv(DATA / "SPORTS_NOVA_V13_AMBIGUITY_RULE_TRAINING_ROWS.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
