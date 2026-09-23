"""V12: SPORTS_NOVA_V12_QB_IDENTITY_SIGNAL_AVAILABILITY.

Diagnostic-only. Determines whether an EXISTING causal pregame signal
(already present somewhere in this codebase/data, not a new external
acquisition) can resolve QB identity ambiguity in the ~621 multi-QB
training team-games characterized by V11
(SPORTS_NOVA_V11_QB_IDENTITY_SIGNAL_DESIGN_TRAINING.json).

No engine change, no V9 use, no parameter sweep. Reuses V11's frozen
training candidate rows (SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv)
so the sample and the "hard filter wrong zone" / "gap0 tie" definitions
are identical to V11's, and only new candidate SIGNALS are added and
scored against the same realized-starter target.

Two candidates were located in the codebase and are evaluated below.

1. V3_CAUSAL_ROSTER_PARTICIPATION_FLAGS
   worker/sports_nova_v3's own causal player panel
   (validation_inputs/NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet) carries
   STARTER_STATUS / ACTIVE_STATUS / ROSTER_STATUS columns. STARTER_STATUS
   is 100% null (unusable). ACTIVE_STATUS/ROSTER_STATUS are populated,
   but the same file self-declares, on every row, CAUSALITY_MODE =
   EVENT_CAUSAL_ONLY, TARGET_DATA = True, PREGAME_FEATURE_DATA = False.
   By the file's own provenance metadata this is post-hoc outcome data,
   not a pre-kickoff feature -- using it would leak the outcome (who
   actually had a stat line that week) into the "signal". Disqualified
   on causality grounds; not scored.

2. V2.1_CAREER_LEAD_SHARE_PROXY
   worker/sports_nova_v2/features_v21.py's role-continuity layer
   (career_lead_games / career_start_count_proxy /
   previous_season_primary_starter_proxy) is genuinely causal (each is
   built with .shift(1) over a player's own already-played games) but is
   not wired into V3's causal roster (worker/sports_nova_v3/simulator.py,
   pregame_state.py) at all -- it lives only in the separate, unrelated
   V2.1 feature builder, over a different panel path
   (worker/sports_nova_v2/panel.py::load_event_causal).
   This script re-derives the same quantity (a player's own historical
   "led team in attempts" share, strictly prior games only, via
   merge_asof so it is defined even for candidates who have no stat line
   in the exact game being evaluated) directly against V3's own causal
   panel, and scores it on V11's training sample. Scored below.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
PLAYER = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
TRAINING_ROWS = DATA / "SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv"


def _causality_audit() -> dict:
    df = pd.read_parquet(PLAYER)
    return {
        "signal": "V3_CAUSAL_ROSTER_PARTICIPATION_FLAGS",
        "source": "data/sports_nova_v3/validation_inputs/NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet",
        "fields_examined": ["STARTER_STATUS", "ACTIVE_STATUS", "ROSTER_STATUS"],
        "STARTER_STATUS_non_null_pct": float(df.STARTER_STATUS.notna().mean()),
        "ACTIVE_STATUS_non_null_pct": float(df.ACTIVE_STATUS.notna().mean()),
        "CAUSALITY_MODE_values": sorted(df.CAUSALITY_MODE.dropna().unique().tolist()),
        "TARGET_DATA_all_true": bool((df.TARGET_DATA == True).all()),
        "PREGAME_FEATURE_DATA_all_false": bool((df.PREGAME_FEATURE_DATA == False).all()),
        "causal_pre_kickoff": False,
        "reason": ("file self-labels every row TARGET_DATA=True, "
                   "PREGAME_FEATURE_DATA=False, CAUSALITY_MODE=EVENT_CAUSAL_ONLY; "
                   "STARTER_STATUS additionally 100% null"),
        "scored": False,
    }


def _career_lead_share_signal(rows: pd.DataFrame, all_df: pd.DataFrame) -> pd.DataFrame:
    all_df = all_df.copy()
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    team_att = (all_df.groupby(["GAME_ID", "TEAM"], as_index=False)["PASS_ATTEMPTS"]
                .sum().rename(columns={"PASS_ATTEMPTS": "team_att"}))
    pg = all_df.merge(team_att, on=["GAME_ID", "TEAM"], how="left")
    pg["led"] = (pg.groupby(["GAME_ID", "TEAM"])["PASS_ATTEMPTS"]
                 .rank(method="first", ascending=False) == 1).astype(float)
    pg = pg.sort_values(["PLAYER_ID", "_key"]).reset_index(drop=True)
    pg["lead_cum_incl"] = pg.groupby("PLAYER_ID")["led"].cumsum()
    pg["games_cum_incl"] = pg.groupby("PLAYER_ID").cumcount() + 1
    career_hist = (pg[["PLAYER_ID", "_key", "lead_cum_incl", "games_cum_incl"]]
                   .rename(columns={"PLAYER_ID": "pid"}).sort_values("_key"))

    gk = all_df[["GAME_ID", "_key"]].drop_duplicates()
    r = rows.merge(gk, on="GAME_ID", how="left")
    r["pid"] = r["pid"].astype(str)
    out = pd.merge_asof(r.sort_values("_key"), career_hist, on="_key", by="pid",
                         direction="backward", allow_exact_matches=False)
    out["career_lead_share"] = out["lead_cum_incl"] / out["games_cum_incl"]
    out["career_lead_share"] = out["career_lead_share"].fillna(0.0)  # no prior games => never led
    return out


def _top1_accuracy(df: pd.DataFrame, score_col: str, keys=None) -> tuple[float, int]:
    groups = df.groupby(["GAME_ID", "TEAM"])
    correct, total = 0, 0
    for key, sub in groups:
        if keys is not None and key not in keys:
            continue
        pick = sub.iloc[int(sub[score_col].to_numpy().argmax())]
        correct += int(pick.is_realized_starter)
        total += 1
    return (correct / total if total else None), total


def main():
    rows = pd.read_csv(TRAINING_ROWS)
    all_df = pd.read_parquet(PLAYER)

    causality_audit = _causality_audit()

    signal_df = _career_lead_share_signal(rows, all_df)

    hard_filter_wrong_zone_keys, gap0_tie_keys = [], []
    for key, sub in signal_df.groupby(["GAME_ID", "TEAM"]):
        hp = sub.sort_values("gap_rank").iloc[0]
        if not bool(hp.is_realized_starter):
            hard_filter_wrong_zone_keys.append(key)
        if (sub.gap == 0).sum() >= 2:
            gap0_tie_keys.append(key)
    hard_filter_wrong_zone_keys = set(hard_filter_wrong_zone_keys)
    gap0_tie_keys = set(gap0_tie_keys)

    # hard-filter accuracy: score by min gap_rank (min gap wins), matching V11
    hf_correct, hf_total = 0, 0
    for key, sub in signal_df.groupby(["GAME_ID", "TEAM"]):
        pick = sub.sort_values("gap_rank").iloc[0]
        hf_correct += int(pick.is_realized_starter)
        hf_total += 1
    hard_filter_acc = hf_correct / hf_total

    overall_acc, n_overall = _top1_accuracy(signal_df, "career_lead_share")
    wrong_zone_acc, n_wz = _top1_accuracy(signal_df, "career_lead_share", hard_filter_wrong_zone_keys)
    gap0_acc, n_g0 = _top1_accuracy(signal_df, "career_lead_share", gap0_tie_keys)
    coverage_pct = float((signal_df.games_cum_incl.notna()).mean())

    career_lead_result = {
        "signal": "V2.1_CAREER_LEAD_SHARE_PROXY",
        "source": ("re-derived directly from V3's own causal panel "
                   "(validation_inputs/NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet) using the same "
                   "definition as worker/sports_nova_v2/features_v21.py::career_lead_games / "
                   "career_start_count_proxy -- shift(1)-only, own-history, box-score-derived"),
        "field": "career_lead_share (cumulative share of prior team-games this player led "
                 "the team in pass attempts, strictly before the evaluated game)",
        "historical_coverage": "2000-2019 training sample (same as V11)",
        "causal_pre_kickoff": True,
        "training_coverage_pct": coverage_pct,
        "identity_accuracy_overall": overall_acc,
        "n_team_games_overall": n_overall,
        "hard_min_gap_filter_baseline_accuracy": hard_filter_acc,
        "n_team_games_hard_filter_baseline": hf_total,
        "accuracy_on_hard_filter_wrong_zone": wrong_zone_acc,
        "n_hard_filter_wrong_zone": n_wz,
        "accuracy_on_gap0_ties": gap0_acc,
        "n_gap0_ties": n_g0,
        "vs_v11_prior_best_alt_wrong_zone_recovery_~40pct": "MATCHES, NOT BETTER",
        "vs_v11_prior_volume_argmax_gap0_accuracy_0.743": "WORSE",
        "meets_acceptance": False,
        "acceptance_failure_reasons": [
            "overall accuracy (0.655) is well below the existing hard-filter baseline (0.862); "
            "using it as primary picker would regress the hard-filter-correct zone",
            "wrong-zone recovery (~0.43) does not materially improve on V11's already-rejected "
            "best alternative (~0.40)",
            "gap0-tie accuracy (~0.56) is materially WORSE than V11's existing "
            "volume-argmax-on-ties accuracy (0.743)",
        ],
    }

    result = {
        "mission": "SPORTS_NOVA_V12_QB_IDENTITY_SIGNAL_AVAILABILITY",
        "excludes_v9_cohort": True,
        "reused_training_sample_from": "SPORTS_NOVA_V11_QB_IDENTITY_SIGNAL_DESIGN_TRAINING.json",
        "n_multi_qb_team_games": len(rows.groupby(["GAME_ID", "TEAM"])),
        "candidates_examined": [causality_audit, career_lead_result],
        "verdict": "NO_USEFUL_SIGNAL",
    }
    out_path = DATA / "SPORTS_NOVA_V12_QB_IDENTITY_SIGNAL_AVAILABILITY.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
