"""V2.1: separate offseason elapsed time from in-season inactivity.

V2 clips `days_rest` to [0,30], so a season opener (~215 days) and a mid-season
30-day absence land on the same value. Those are two different latent states:
on the eligible panel the ceiling cohort splits into 796 season-opener rows
averaging 221.6 passing yards and 1,035 in-season-absence rows averaging 91.8.

This module keeps V2's feature set untouched and adds a layer that names the
state explicitly, plus role-continuity features derived from box scores only
(who led his team in pass attempts, and how much of the team's volume he
carried). No starter labels are invented; "led the team in attempts in a game
already played" is the derivable proxy.

V2 is not modified. Every function here is additive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import panel as P

OFFSEASON_GAP_CAP = 400.0


# ---------------------------------------------------------------------------
def team_game_tables(full_causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team pass-attempt totals per team-game, and a within-season game index.

    Built from every position, since non-QBs occasionally throw and the share
    denominator must be the team's real attempt total.
    """
    canon = P._canon_map()
    df = full_causal.copy()
    df["team_canon"] = df["recent_team"].map(canon)
    df = df[df.team_canon.notna()]

    team_att = (df.groupby(["season", "week", "team_canon"], as_index=False)
                ["attempts"].sum().rename(columns={"attempts": "team_attempts"}))

    idx = team_att[["season", "week", "team_canon"]].drop_duplicates()
    idx = idx.sort_values(["season", "team_canon", "week"])
    idx["team_game_no"] = idx.groupby(["season", "team_canon"]).cumcount()
    return team_att, idx


def add_role_continuity(qb: pd.DataFrame, full_causal: pd.DataFrame) -> pd.DataFrame:
    """Layer V2.1. All features use only games already played."""
    df = qb.copy()
    team_att, idx = team_game_tables(full_causal)

    df = df.merge(team_att, on=["season", "week", "team_canon"], how="left")
    df = df.merge(idx, on=["season", "week", "team_canon"], how="left")
    df = df.sort_values(["player_id", "kickoff_timestamp_utc"]).reset_index(drop=True)

    # --- what happened in the game he actually played, then shifted forward ---
    df["own_share_this_game"] = (df["attempts"] /
                                 df["team_attempts"].replace(0, np.nan)).clip(0, 1)
    df["led_team_this_game"] = (
        df.groupby(["season", "week", "team_canon"])["attempts"]
        .rank(method="first", ascending=False) == 1).astype(float)

    g = df.groupby("player_id")
    df["prev_game_attempt_share"] = g["own_share_this_game"].shift(1)
    df["prev_game_was_team_lead"] = g["led_team_this_game"].shift(1)
    df["career_lead_games"] = g["led_team_this_game"].transform(
        lambda s: s.shift(1).expanding().sum())
    df["career_lead_share"] = (df["career_lead_games"] /
                               df["games_played_prior"].replace(0, np.nan))

    # --- state separation ---
    df["prev_season"] = g["season"].shift(1)
    df["prev_team"] = g["team_canon"].shift(1)
    df["prev_team_game_no"] = g["team_game_no"].shift(1)

    crossed = (df["prev_season"].notna() & (df["prev_season"] != df["season"]))
    first_ever = df["prev_season"].isna()

    df["is_season_opener"] = (df["season_games_prior"] == 0).astype(float)
    df["is_team_week1"] = (df["week"] == 1).astype(float)
    df["crossed_season_boundary"] = crossed.astype(float)

    # Raw elapsed days, uncapped at 30, so an offseason is numerically distinct.
    raw_days = (g["kickoff_timestamp_utc"].diff().dt.total_seconds() / 86400.0)
    df["offseason_gap_days"] = np.where(crossed, raw_days.clip(0, OFFSEASON_GAP_CAP), 0.0)
    # In-season rest only; an offseason row carries no in-season rest value.
    df["in_season_days_rest"] = np.where(crossed | first_ever, np.nan,
                                         raw_days.clip(0, 30))

    # Team games he was absent for, within the same season. This is the signal
    # `days_rest` was proxying badly: it is 0 across an offseason by definition.
    missed = df["team_game_no"] - df["prev_team_game_no"] - 1
    df["team_games_missed"] = np.where(crossed | first_ever, 0.0,
                                       missed.clip(lower=0))
    df["same_team_as_prior_game"] = np.where(
        first_ever, np.nan, (df["prev_team"] == df["team_canon"]).astype(float))

    # --- prior-season standing ---
    ps = (df.groupby(["player_id", "season"])
          .agg(ps_attempts=("attempts", "sum"),
               ps_lead_games=("led_team_this_game", "sum"),
               ps_games=("attempts", "size"),
               ps_team=("team_canon", "last"))
          .reset_index())
    ps_team_att = (team_att.groupby(["season", "team_canon"], as_index=False)
                   ["team_attempts"].sum()
                   .rename(columns={"team_attempts": "ps_team_attempts",
                                    "team_canon": "ps_team"}))
    ps = ps.merge(ps_team_att, on=["season", "ps_team"], how="left")
    ps["ps_attempt_share"] = (ps["ps_attempts"] /
                              ps["ps_team_attempts"].replace(0, np.nan)).clip(0, 1)
    ps["season"] = ps["season"] + 1          # becomes the *prior* season's record
    ps = ps.rename(columns={
        "ps_attempts": "prior_season_attempts",
        "ps_lead_games": "prior_season_lead_games",
        "ps_games": "prior_season_games",
        "ps_attempt_share": "prior_season_attempt_share",
        "ps_team": "prior_season_team"})

    df = df.merge(ps[["player_id", "season", "prior_season_attempts",
                      "prior_season_lead_games", "prior_season_games",
                      "prior_season_attempt_share", "prior_season_team"]],
                  on=["player_id", "season"], how="left")
    df["same_team_as_prior_season"] = np.where(
        df["prior_season_team"].isna(), np.nan,
        (df["prior_season_team"] == df["team_canon"]).astype(float))
    df["has_prior_season"] = df["prior_season_games"].notna().astype(float)

    # The causal panel has attempts and team-game identity, but no official
    # starter designation.  Preserve that distinction explicitly: these are
    # measurable role proxies, never invented starter labels.
    df["prior_season_start_share_proxy"] = (
        df["prior_season_lead_games"] /
        df["prior_season_games"].replace(0, np.nan)
    )
    df["career_start_count_proxy"] = df["career_lead_games"]
    df["previous_season_primary_starter_proxy"] = np.where(
        df["prior_season_lead_games"].notna(),
        (df["prior_season_lead_games"] >= df["prior_season_games"] * 0.5).astype(float),
        np.nan,
    )
    return df


# ---------------------------------------------------------------------------
STATE_FEATURES = ["is_season_opener", "is_team_week1", "crossed_season_boundary",
                  "offseason_gap_days", "in_season_days_rest", "team_games_missed"]

ROLE_FEATURES = ["prev_game_attempt_share", "prev_game_was_team_lead",
                 "career_lead_games", "career_lead_share",
                 "same_team_as_prior_game", "same_team_as_prior_season",
                 "prior_season_attempts", "prior_season_lead_games",
                 "prior_season_games", "prior_season_attempt_share",
                 "has_prior_season", "prior_season_start_share_proxy",
                 "career_start_count_proxy", "previous_season_primary_starter_proxy"]

V21_NEW_FEATURES = STATE_FEATURES + ROLE_FEATURES

# Candidate feature sets. A is V2 exactly; the rest are the fixes under test.
V2_BASE = F.V2_FEATURES
_NO_REST = [f for f in V2_BASE if f != "days_rest"]

CANDIDATES = {
    "A_current_v2": V2_BASE,
    "B_days_rest_plus_opener_flag": V2_BASE + ["is_season_opener", "is_team_week1"],
    "C_state_decomposition": _NO_REST + STATE_FEATURES,
    "D_drop_days_rest": _NO_REST,
    "E_role_continuity": _NO_REST + STATE_FEATURES + ROLE_FEATURES,
}

FEATURE_GROUPS_V21 = dict(F.FEATURE_GROUPS)
FEATURE_GROUPS_V21["rest_state"] = STATE_FEATURES
FEATURE_GROUPS_V21["role_continuity"] = ROLE_FEATURES


def build_v21_panel(return_full: bool = False):
    """V2's pipeline plus the V2.1 layer, on exactly V1's eligible rows."""
    qb = P.build_qb_panel()
    qb = F.mask_air_yards_era(qb)
    qb = F.harmonize_schema_split(qb)
    qb = F.add_player_state(qb)
    qb = F.add_shrinkage(qb)

    kickoffs = (qb.groupby(["season", "week", "team_canon"])["kickoff_timestamp_utc"]
                .min().rename("kickoff").reset_index())
    led = F.load_team_game_ledger(kickoffs)
    led = F.add_team_trailing(led)

    qb = F.attach_context(qb, led)
    qb = F.add_interactions(qb)
    qb = add_role_continuity(qb, P.load_event_causal())

    eligible = P.apply_v1_eligibility(qb)
    if return_full:
        return eligible, led, qb
    return eligible, led


# ---------------------------------------------------------------------------
# evaluation buckets
# ---------------------------------------------------------------------------
def buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Named row masks for bucketed reporting."""
    lead = df["career_lead_share"].fillna(0)
    ceiling = df["days_rest"] >= 29.9
    crossed = df["crossed_season_boundary"] == 1
    return {
        "WEEK_1": df.week == 1,
        "WEEKS_2_4": df.week.between(2, 4),
        "WEEKS_5_PLUS": df.week >= 5,
        "RETURNING_PRIMARY_QB": (df["prior_season_lead_games"].fillna(0) >= 8),
        "NEW_STARTER_LOW_PRIOR_START_SHARE": (lead < 0.5),
        "NORMAL_REST": df["days_rest"] <= 8,
        # The three states V2 collapses onto days_rest == 30. A player whose
        # previous game was last season is crossing an offseason even if he
        # debuts in week 6, so he is not an in-season absence.
        "SEASON_OPENER": ceiling & (df.week == 1),
        "LATE_SEASON_DEBUT": ceiling & (df.week > 1) & crossed,
        "LONG_IN_SEASON_REST": ceiling & (df.week > 1) & ~crossed,
    }
