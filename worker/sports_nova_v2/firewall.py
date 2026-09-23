"""Temporal firewall checks.

These are assertions about the data, not unit tests of code paths: they run on
the real panel and fail the build if any feature could have seen the future.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import panel as P


def check_team_join_causality(el: pd.DataFrame) -> dict:
    """The newest game feeding a team/opponent feature must kick off strictly
    before the row's own kickoff."""
    own_bad = int((el.own_prev_kickoff >= el.kickoff_timestamp_utc).sum())
    opp_bad = int((el.opp_prev_kickoff >= el.kickoff_timestamp_utc).sum())
    return {
        "own_team_rows_violating": own_bad,
        "opponent_rows_violating": opp_bad,
        "own_team_rows_checked": int(el.own_prev_kickoff.notna().sum()),
        "opponent_rows_checked": int(el.opp_prev_kickoff.notna().sum()),
        "PASS": own_bad == 0 and opp_bad == 0,
    }


def check_shift1_player(full: pd.DataFrame) -> dict:
    """A player's own trailing mean must never embed the current game. Verified
    by recomputing one window the slow, explicit way.

    Must run on the *unfiltered* panel: trailing features are built over a
    player's whole career, so recomputing them on the eligibility-filtered
    frame would compare against a truncated history.
    """
    df = full.sort_values(["player_id", "kickoff_timestamp_utc"])
    viol = 0
    checked = 0
    for pid, sub in df.groupby("player_id"):
        if len(sub) < 6:
            continue
        y = sub.passing_yards.values
        got = sub.pass_yds_w3.values
        for i in range(1, len(sub)):
            exp = np.mean(y[max(0, i - 3):i])
            if np.isfinite(got[i]):
                checked += 1
                if abs(got[i] - exp) > 1e-8:
                    viol += 1
        if checked > 40000:
            break
    return {"rows_checked": checked, "violations": viol, "PASS": viol == 0}


def check_no_current_game_correlation(el: pd.DataFrame, features: list[str]) -> dict:
    """A feature that accidentally embeds the current game shows a near-perfect
    correlation with the target. Flag anything implausibly high."""
    y = el[C.TARGET].values
    flags = []
    for f in features:
        v = el[f].values.astype(float)
        m = np.isfinite(v) & np.isfinite(y)
        if m.sum() < 500:
            continue
        r = float(np.corrcoef(v[m], y[m])[0, 1])
        if abs(r) > 0.90:
            flags.append({"feature": f, "corr_with_target": round(r, 4)})
    return {"suspicious_features": flags, "PASS": len(flags) == 0}


def check_season_boundary(full: pd.DataFrame) -> dict:
    """Season-to-date features must reset at a season boundary: the first game
    of a player's season carries no within-season history.

    Runs on the unfiltered panel for the same reason as the shift(1) check.
    """
    df = full.sort_values(["player_id", "season", "kickoff_timestamp_utc"])
    first = df.groupby(["player_id", "season"]).head(1)
    bad = int((first.season_games_prior != 0).sum())
    return {"first_games_checked": int(len(first)), "violations": bad,
            "PASS": bad == 0}


def check_walkforward_cutoffs(folds) -> dict:
    bad = [f.season for f in folds if not f.cutoff_ok]
    return {"folds": len(folds), "violating_seasons": bad, "PASS": not bad}


def check_no_market_data(el: pd.DataFrame) -> dict:
    P._assert_no_market_columns(el)
    return {"PASS": True, "note": "no odds/spread/total/vegas columns present"}


def run_all(el: pd.DataFrame, full: pd.DataFrame, folds=None) -> dict:
    res = {
        "team_opponent_join_causality": check_team_join_causality(el),
        "player_shift1": check_shift1_player(full),
        "season_boundary": check_season_boundary(full),
        "target_correlation_scan": check_no_current_game_correlation(el, F.V2_FEATURES),
        "market_data_absent": check_no_market_data(el),
    }
    if folds is not None:
        res["walkforward_cutoffs"] = check_walkforward_cutoffs(folds)
    res["TEMPORAL_FIREWALL"] = "PASS" if all(
        v.get("PASS") for v in res.values() if isinstance(v, dict)) else "FAIL"
    return res


if __name__ == "__main__":
    el, _, full = F.build_v2_panel(return_full=True)
    print(json.dumps(run_all(el, full), indent=2, default=str))
