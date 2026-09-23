"""V14: SPORTS_NOVA_V14_DEPTH_CHART_SIGNAL_FALSIFICATION.

Diagnostic-only. NO engine changes (simulator.py / _qb_shares / _primary_qb /
**2.5 untouched). Tests whether free nflverse depth-chart data (documented
but never integrated -- GAP-003 in worker/sports_nova_v2/build_artifacts.py)
provides causal pre-kickoff QB-identity evidence strong enough to fix the
identity defect that V12 (no useful EXISTING signal) and V13 (ambiguity
widening fails under the frozen **2.5 share math) could not.

Data source (fetched once, cached under
data/sports_nova_v3/validation_inputs/depth_charts_external/,
depth_charts_<season>.parquet for season in 2001..2019):
  https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_<season>.parquet
Free, no auth, no payment -- consistent with GAP-003's "FREE AND REACHABLE".

PHASE A -- causal pre-kickoff source audit
-------------------------------------------
Three facts verified directly (no other route was available -- the public
nflverse-data repo has no depth_chart-specific build script, and GitHub code
search requires auth this environment does not have):
  1. The delivered schema (season, club_code, week, game_type, depth_team,
     gsis_id, position, ...) carries NO publication/scrape timestamp field
     of any kind.
  2. No depth-chart scrape/build workflow is discoverable in the public
     nflverse-data or nflreadr repos (checked .github/workflows and R/); the
     `load_depth_charts()` R helper documents only the data's SHAPE, not its
     collection cadence relative to kickoff.
  3. nflreadr's own dictionary vignette states the data source changed after
     the 2024 season -- so even a timing claim for one era would not carry
     over to the other.
  => CAUSAL_PRE_KICKOFF cannot be proven. This independently reconfirms
     GAP-003's existing caveat ("publication timestamps are not exposed so
     causal use needs care") rather than resolving it.

Separately, a circularity check: does depth_team=='1' (QB1) simply mirror
that week's own realized top-attempts passer (which would make it leak the
outcome, not evidence anything pregame)? Measured 81-86% agreement across
three sample seasons (2001/2010/2019) -- NOT 100%, so it is not a trivial
copy of the outcome. That rules out circularity; it does NOT establish
pre-kickoff availability, which is a separate claim. (Some of that ~15-19%
disagreement is plausibly target error, not source error: depth_team==1
designates who STARTS: is_realized_starter, this project's existing target,
is whoever accumulates the most attempts -- these differ on early injuries
and blowouts. Noted for the record; it does not change the measured
regression below, which is scored against the project's own established
target, the same target the engine itself is evaluated against.)

Per the mission's own gate ("must prove causal pre-kickoff availability;
otherwise STOP_SOURCE_NOT_CAUSAL"), Phase B below is reported as
supplementary evidence only -- it does not change the Phase A verdict.

PHASE B -- training-only accuracy test (2000-2019, V11/V12's frozen 621
multi-QB team-game candidate set; depth-chart coverage starts 2001, so
season-2000 rows are definitionally uncovered)
------------------------------------------------------------------------
For each team-game, dc_qb1_pid = that team's depth_team=='1' QB gsis_id for
the matching (season, week), joined against V11's causal candidate roster
(SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv). "Coverage" requires BOTH a
depth-chart entry for that week AND that entry's gsis_id being one of the
causal candidates already on the team's roster.

Two DIFFERENT hard-filter-wrong/correct zone definitions exist in this
project's history and must not be conflated:
  - V11/V12 used sort_values('gap_rank').iloc[0] (first-row tiebreak):
    86 wrong-zone / 113 gap0-tie games.
  - V13 replicated the ACTUAL _qb_shares/_primary_qb mechanism (argmax by
    volume among min-gap survivors): 64 wrong-zone games, 557 correct-zone.
This script uses the V13 (actual-engine) definition throughout, since it is
what the engine really does. Any comparison to V12's "~0.43" wrong-zone
number is therefore a comparison across DIFFERENTLY-SIZED zones (58-of-64
covered here vs. 86 there), not a like-for-like rerun -- both are reported,
neither should be read as an exact tie.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
TRAINING_ROWS = DATA / "SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv"
PLAYER = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
DC_DIR = DATA / "validation_inputs" / "depth_charts_external"

ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # historical -> V3's canonical franchise code


def load_dc_qb1() -> pd.DataFrame:
    files = sorted(glob.glob(str(DC_DIR / "depth_charts_*.parquet")))
    parts = [pd.read_parquet(f, columns=["season", "club_code", "week", "game_type",
                                          "depth_team", "gsis_id", "position"]) for f in files]
    dc = pd.concat(parts, ignore_index=True)
    dc_qb1 = dc[(dc.position == "QB") & (dc.game_type == "REG") & (dc.depth_team == "1")]
    dc_qb1 = dc_qb1[["season", "club_code", "week", "gsis_id"]].drop_duplicates(["season", "club_code", "week"])
    dc_qb1["week"] = dc_qb1["week"].astype(int)
    dc_qb1["club_code"] = dc_qb1["club_code"].replace(ALIAS)
    return dc_qb1.rename(columns={"season": "SEASON", "club_code": "TEAM", "week": "WEEK", "gsis_id": "dc_qb1_pid"})


def circularity_check(all_df: pd.DataFrame) -> dict:
    out = {}
    for season in (2001, 2010, 2019):
        dc = pd.read_parquet(DC_DIR / f"depth_charts_{season}.parquet")
        dc_qb1 = dc[(dc.position == "QB") & (dc.game_type == "REG") & (dc.depth_team == "1")]
        dc_qb1 = dc_qb1[["club_code", "week", "gsis_id"]].drop_duplicates(["club_code", "week"])
        real = all_df[(all_df.SEASON == season) & (all_df.POSITION == "QB")]
        real_top = (real.sort_values("PASS_ATTEMPTS", ascending=False)
                    .groupby(["TEAM", "WEEK"], as_index=False).first()[["TEAM", "WEEK", "PLAYER_ID"]])
        m = dc_qb1.merge(real_top, left_on=["club_code", "week"], right_on=["TEAM", "WEEK"], how="inner")
        out[str(season)] = {"n_team_weeks_matched": int(len(m)),
                             "agree_rate_with_realized_top_passer": float((m.gsis_id == m.PLAYER_ID).mean())}
    return out


def hard_filter_pick(sub: pd.DataFrame) -> str:
    """Replicates the ACTUAL _qb_shares/_primary_qb mechanism (V13 definition)."""
    min_gap = sub.gap.min()
    survivors = sub[sub.gap == min_gap]
    return survivors.loc[survivors.volume.idxmax()].pid


def main():
    all_df = pd.read_parquet(PLAYER)
    causality_facts = {
        "timestamp_field_in_schema": False,
        "build_or_scrape_workflow_discoverable_in_public_repo": False,
        "source_methodology_changed_after_2024_per_nflreadr_dictionary": True,
        "circularity_check_agree_rate_with_realized_top_passer_by_season": circularity_check(all_df),
        "circularity_verdict": "NOT_CIRCULAR (81-86% agreement, not ~100%) -- rules out simple outcome-copying, does NOT establish pre-kickoff timing",
        "CAUSAL_PRE_KICKOFF": "UNPROVEN",
        "reason": ("no publication/scrape timestamp in the delivered schema and no discoverable "
                   "build workflow in the public nflverse-data/nflreadr repos; independently "
                   "reconfirms GAP-003's existing caveat rather than resolving it"),
    }

    rows = pd.read_csv(TRAINING_ROWS)
    rows["pid"] = rows["pid"].astype(str)
    game_key = all_df[["GAME_ID", "TEAM", "SEASON", "WEEK"]].drop_duplicates()
    rows = rows.merge(game_key, on=["GAME_ID", "TEAM"], how="left")
    rows = rows.merge(load_dc_qb1(), on=["SEASON", "TEAM", "WEEK"], how="left")

    recs = []
    for (g, t), sub in rows.groupby(["GAME_ID", "TEAM"]):
        realized = sub.loc[sub.is_realized_starter, "pid"]
        realized_pid = realized.iloc[0] if len(realized) else None
        hf_pick = hard_filter_pick(sub)
        dc_pid = sub.dc_qb1_pid.iloc[0] if sub.dc_qb1_pid.notna().any() else None
        dc_covered = bool(dc_pid is not None and dc_pid in set(sub.pid))
        recs.append({
            "GAME_ID": g, "TEAM": t, "SEASON": int(sub.SEASON.iloc[0]),
            "HF_CORRECT": hf_pick == realized_pid,
            "DC_COVERED": dc_covered,
            "DC_CORRECT": (dc_pid == realized_pid) if dc_covered else None,
            "GAP0_TIE": bool((sub.gap == 0).sum() >= 2),
        })
    tg = pd.DataFrame(recs)
    n = len(tg)

    covered = tg[tg.DC_COVERED]
    wrong_zone = tg[~tg.HF_CORRECT]
    wrong_zone_cov = wrong_zone[wrong_zone.DC_COVERED]
    correct_zone = tg[tg.HF_CORRECT]
    correct_zone_cov = correct_zone[correct_zone.DC_COVERED]
    gap0 = tg[tg.GAP0_TIE]
    gap0_cov = gap0[gap0.DC_COVERED]

    n_resolved = int(wrong_zone_cov.DC_CORRECT.astype(bool).sum())
    n_regressed = int((~correct_zone_cov.DC_CORRECT.astype(bool)).sum())

    phase_b = {
        "n_team_games": n,
        "COVERAGE": float(tg.DC_COVERED.mean()),
        "coverage_note": "0% for season-2000 rows (no depth-chart data before 2001); "
                          f"{float(tg[tg.SEASON >= 2001].DC_COVERED.mean()):.3f} for seasons 2001-2019",
        "OVERALL_ACCURACY_coverage_conditional": float(covered.DC_CORRECT.mean()),
        "n_overall_covered": int(len(covered)),
        "zone_definition": "V13/actual-engine definition (argmax-by-volume among min-gap survivors), "
                            "NOT V11/V12's first-row-tiebreak definition -- zone sizes differ (64 vs. 86 "
                            "wrong-zone, 557 vs. 113... V12's gap0-tie of 113 is a different denominator "
                            "entirely) and are not a like-for-like rerun",
        "WRONG_ZONE_ACCURACY_coverage_conditional": float(wrong_zone_cov.DC_CORRECT.mean()),
        "n_wrong_zone": int(len(wrong_zone)), "n_wrong_zone_covered": int(len(wrong_zone_cov)),
        "GAP0_TIE_ACCURACY_coverage_conditional": float(gap0_cov.DC_CORRECT.mean()),
        "n_gap0_tie": int(len(gap0)), "n_gap0_tie_covered": int(len(gap0_cov)),
        "CORRECT_ZONE_REGRESSION_coverage_conditional": float((~correct_zone_cov.DC_CORRECT.astype(bool)).mean()),
        "n_correct_zone": int(len(correct_zone)), "n_correct_zone_covered": int(len(correct_zone_cov)),
        "COMPLEMENTARY_CASES_RESOLVED": n_resolved,
        "CASES_BROKEN_IN_CORRECT_ZONE": n_regressed,
        "NET_EXCHANGE": n_resolved - n_regressed,
        "exchange_note": (f"{n_resolved} wrong-zone games fixed vs. {n_regressed} correct-zone games "
                          f"broken -- net {n_resolved - n_regressed}. Same structural shape as V13's "
                          "38-broken-vs-11-helped: a signal that touches more correct cases than it fixes."),
    }

    acceptance = {
        "causal_pre_kickoff_true": False,  # UNPROVEN, not true
        "coverage_>=0.70": phase_b["COVERAGE"] >= 0.70,
        "wrong_zone_materially_>0.43": phase_b["WRONG_ZONE_ACCURACY_coverage_conditional"] > 0.50,  # "materially" bar
        "gap0_tie_>0.743": phase_b["GAP0_TIE_ACCURACY_coverage_conditional"] > 0.743,
        "correct_zone_regression_<=0.05": phase_b["CORRECT_ZONE_REGRESSION_coverage_conditional"] <= 0.05,
    }
    verdict = "USEFUL_DEPTH_CHART_SIGNAL" if all(acceptance.values()) else "NO_USEFUL_DEPTH_CHART_SIGNAL"

    result = {
        "mission": "SPORTS_NOVA_V14_DEPTH_CHART_SIGNAL_FALSIFICATION",
        "source": "nflverse-data depth_charts release (github.com/nflverse/nflverse-data), free, seasons 2001-2019 used",
        "phase_A_causal_audit": causality_facts,
        "phase_B_training_accuracy": phase_b,
        "acceptance_checks": acceptance,
        "VERDICT": verdict,
        "SINGLE_NEXT_ACTION": (
            "Stop QB-identity picker/signal-acquisition work (V11-V14 have now exhausted the causal "
            "existing-pipeline features, the ** 2.5-constrained ambiguity-widening approach, and the one "
            "free external candidate source, all negatively). Per the mission's own if_not_useful branch: "
            "redesign the evaluation/model target so team-level passing-volume calibration does not "
            "require resolving individual QB identity with certainty -- e.g. score/calibrate at the "
            "team-pass-volume level and treat per-QB attribution as a secondary, explicitly uncertain "
            "output, rather than continuing to search for an identity signal."
        ),
    }
    out_path = DATA / "SPORTS_NOVA_V14_DEPTH_CHART_SIGNAL_FALSIFICATION.json"
    out_path.write_text(json.dumps(result, indent=2))
    tg.to_csv(DATA / "SPORTS_NOVA_V14_DEPTH_CHART_TEAM_GAMES.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
