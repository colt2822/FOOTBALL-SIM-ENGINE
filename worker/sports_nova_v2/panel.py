"""Builds the causal player-game panel.

`build_qb_panel` reproduces V1's eligibility and features exactly; the V2
feature layers are added on top of the *same rows* so that any V1-vs-V2
difference is model quality and never row selection.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

MARKET_TERMS = ("odds", "line", "moneyline", "spread", "vig", "book", "vegas")


def _assert_no_market_columns(df: pd.DataFrame) -> None:
    # "line" is a substring of legitimate names (e.g. offensive_line); match on
    # token boundaries instead.
    bad = []
    for col in df.columns:
        toks = set(col.lower().replace("-", "_").split("_"))
        if toks & {"odds", "moneyline", "spread", "vig", "vegas"}:
            bad.append(col)
        if "line" in toks and "yardline" not in col.lower():
            bad.append(col)
    assert not bad, f"market columns present in dataset: {sorted(set(bad))}"


def load_event_causal() -> pd.DataFrame:
    df = pd.read_parquet(C.CAUSAL_PARQUET)
    df = df[df.event_time_status == "EXACT"].copy()
    _assert_no_market_columns(df)
    return df


def _canon_map() -> dict:
    doc = json.loads(C.ALIAS_MAP.read_text())
    return {k: v["canonical_id"] for k, v in doc["team_alias_map"].items()}


def build_qb_panel() -> pd.DataFrame:
    """QB rows, ordered by kickoff, with V1's four features attached.

    Eligibility is applied by the caller so that V2 can attach extra features
    before the (identical) filter runs.
    """
    df = load_event_causal()
    qb = df[(df.position == "QB") & (df.attempts > 0)].copy()
    qb["kickoff_timestamp_utc"] = pd.to_datetime(qb["m_kickoff_timestamp_utc"], utc=True)
    qb = qb.sort_values(["player_id", "kickoff_timestamp_utc"]).reset_index(drop=True)

    g = qb.groupby("player_id")
    qb["games_played_prior"] = g.cumcount()
    qb["career_prior_pass_yds"] = g["passing_yards"].transform(
        lambda s: s.shift(1).expanding().mean())
    qb["career_prior_attempts"] = g["attempts"].transform(
        lambda s: s.shift(1).expanding().mean())
    qb["trailing4_pass_yds"] = g["passing_yards"].transform(
        lambda s: s.shift(1).rolling(4).mean())
    qb["trailing4_attempts"] = g["attempts"].transform(
        lambda s: s.shift(1).rolling(4).mean())

    qb["season_td_pass_yds"] = qb.groupby(["player_id", "season"])["passing_yards"].transform(
        lambda s: s.shift(1).expanding().mean())
    qb["season_td_pass_yds"] = qb["season_td_pass_yds"].fillna(qb["career_prior_pass_yds"])

    canon = _canon_map()
    qb["team_canon"] = qb["recent_team"].map(canon)
    qb["opp_canon"] = qb["opponent_team"].map(canon)
    return qb


def apply_v1_eligibility(qb: pd.DataFrame) -> pd.DataFrame:
    """V1's filter, verbatim: >=4 prior games and all four V1 features present."""
    return qb[qb.games_played_prior >= C.MIN_PRIOR_GAMES].dropna(
        subset=C.V1_FEATURES).copy()


def row_key(df: pd.DataFrame) -> set:
    return set(map(tuple, df[["player_id", "season", "week"]].values))
