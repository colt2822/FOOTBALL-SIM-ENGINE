"""V2 feature layers.

Every feature is built from information available strictly before the row's own
kickoff. Two rules are enforced mechanically rather than by convention:

  1. Any trailing statistic is `shift(1)` on the entity's own kickoff-ordered
     sequence, so an entity never sees its own current game.
  2. Team and opponent features are shifted on the *team's* sequence and then
     joined to the player row, so a defense never sees the game it is being
     used to predict.

Air-yards-derived fields are zero-filled before 2006 in the source panel; they
are masked to NaN so a sentinel never gets read as a real value.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import config as C

AIR_YARDS_ERA = 2006
AIR_YARDS_COLS = ["passing_air_yards", "passing_yards_after_catch", "pacr",
                  "air_yards_share", "wopr", "racr", "receiving_air_yards"]

# Season-varying availability -> never used as model inputs.
EXCLUDED_SEASON_VARYING = ["dakota", "passing_cpoe", "game_id"]

WINDOWS = [3, 5, 8]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def mask_air_yards_era(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pre = df.season < AIR_YARDS_ERA
    for c in AIR_YARDS_COLS:
        if c in df.columns:
            df.loc[pre, c] = np.nan
    return df


def harmonize_schema_split(df: pd.DataFrame) -> pd.DataFrame:
    """1999-2024 uses `sacks`/`interceptions`; 2025 uses the renamed columns."""
    df = df.copy()
    if "sacks_suffered" in df.columns:
        df["sacks_taken"] = df["sacks"].fillna(df["sacks_suffered"])
    else:
        df["sacks_taken"] = df["sacks"]
    if "passing_interceptions" in df.columns:
        df["ints_thrown"] = df["interceptions"].fillna(df["passing_interceptions"])
    else:
        df["ints_thrown"] = df["interceptions"]
    return df


def trailing(frame: pd.DataFrame, by: str, col: str, window: int | None) -> pd.Series:
    """shift(1) rolling mean (window) or expanding mean (window=None)."""
    g = frame.groupby(by)[col]
    if window is None:
        return g.transform(lambda s: s.shift(1).expanding().mean())
    return g.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def trailing_ratio(frame: pd.DataFrame, by: str, num: str, den: str,
                   window: int | None) -> pd.Series:
    """Ratio of trailing sums. Correct for rate stats: a 40-attempt game should
    weigh more than a 10-attempt game, which a mean-of-ratios would not do."""
    g = frame.groupby(by)
    if window is None:
        n = g[num].transform(lambda s: s.shift(1).expanding().sum())
        d = g[den].transform(lambda s: s.shift(1).expanding().sum())
    else:
        n = g[num].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        d = g[den].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    return n / d.replace(0, np.nan)


def trailing_std(frame: pd.DataFrame, by: str, col: str, window: int) -> pd.Series:
    return frame.groupby(by)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=2).std())


# ---------------------------------------------------------------------------
# Layer 2 - hierarchical shrinkage
# ---------------------------------------------------------------------------
def league_prior(frame: pd.DataFrame, col: str) -> pd.Series:
    """Causal expanding league mean over completed weeks.

    Computed at (season, week) granularity from games already played, then
    broadcast back, so no row contributes to its own prior.
    """
    wk = (frame.groupby(["season", "week"])[col].agg(["sum", "count"])
          .sort_index().reset_index())
    wk["cum_sum"] = wk["sum"].cumsum().shift(1)
    wk["cum_n"] = wk["count"].cumsum().shift(1)
    wk["prior"] = wk["cum_sum"] / wk["cum_n"]
    return frame.merge(wk[["season", "week", "prior"]], on=["season", "week"],
                       how="left")["prior"].values


def estimate_shrinkage_k(frame: pd.DataFrame, col: str, by: str = "player_id",
                         max_season: int = 2005) -> float:
    """Empirical-Bayes k = within-player variance / between-player variance.

    Estimated only on seasons strictly before the first OOS season, so the same
    frozen constant is causal for every walk-forward fold.
    """
    sub = frame[frame.season <= max_season]
    grp = sub.groupby(by)[col]
    counts = grp.count()
    keep = counts[counts >= 4].index
    sub = sub[sub[by].isin(keep)]
    if len(sub) < 100:
        return 10.0
    grp = sub.groupby(by)[col]
    within = float(grp.var().mean())
    between = float(grp.mean().var())
    if not np.isfinite(within) or not np.isfinite(between) or between <= 0:
        return 10.0
    return float(np.clip(within / between, 1.0, 200.0))


def shrink(raw: pd.Series | np.ndarray, n: pd.Series | np.ndarray,
           prior: pd.Series | np.ndarray, k: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    n = np.asarray(n, dtype=float)
    prior = np.asarray(prior, dtype=float)
    out = (n * np.nan_to_num(raw) + k * prior) / (n + k)
    # With no prior observations the shrunk value is exactly the prior.
    return np.where(np.isnan(raw), prior, out)


# ---------------------------------------------------------------------------
# Layer 1 - player state
# ---------------------------------------------------------------------------
def add_player_state(qb: pd.DataFrame) -> pd.DataFrame:
    df = qb.copy()
    df["ypa_game"] = df.passing_yards / df.attempts.replace(0, np.nan)
    df["comp_pct_game"] = df.completions / df.attempts.replace(0, np.nan)
    df["epa_per_att_game"] = df.passing_epa / df.attempts.replace(0, np.nan)
    df["adot_game"] = df.passing_air_yards / df.attempts.replace(0, np.nan)

    for w in WINDOWS:
        df[f"pass_yds_w{w}"] = trailing(df, "player_id", "passing_yards", w)
        df[f"attempts_w{w}"] = trailing(df, "player_id", "attempts", w)
        df[f"ypa_w{w}"] = trailing_ratio(df, "player_id", "passing_yards", "attempts", w)
        df[f"comp_pct_w{w}"] = trailing_ratio(df, "player_id", "completions", "attempts", w)
        df[f"epa_att_w{w}"] = trailing_ratio(df, "player_id", "passing_epa", "attempts", w)
        df[f"sack_rate_w{w}"] = trailing_ratio(df, "player_id", "sacks_taken", "attempts", w)
        df[f"int_rate_w{w}"] = trailing_ratio(df, "player_id", "ints_thrown", "attempts", w)
        df[f"td_rate_w{w}"] = trailing_ratio(df, "player_id", "passing_tds", "attempts", w)
        df[f"adot_w{w}"] = trailing_ratio(df, "player_id", "passing_air_yards", "attempts", w)
        df[f"carries_w{w}"] = trailing(df, "player_id", "carries", w)
        df[f"rush_yds_w{w}"] = trailing(df, "player_id", "rushing_yards", w)
        df[f"first_downs_w{w}"] = trailing(df, "player_id", "passing_first_downs", w)

    df["pass_yds_std5"] = trailing_std(df, "player_id", "passing_yards", 5)
    df["attempts_std5"] = trailing_std(df, "player_id", "attempts", 5)
    df["pass_yds_std8"] = trailing_std(df, "player_id", "passing_yards", 8)

    df["career_ypa"] = trailing_ratio(df, "player_id", "passing_yards", "attempts", None)
    df["career_comp_pct"] = trailing_ratio(df, "player_id", "completions", "attempts", None)
    df["career_sack_rate"] = trailing_ratio(df, "player_id", "sacks_taken", "attempts", None)
    df["career_epa_att"] = trailing_ratio(df, "player_id", "passing_epa", "attempts", None)
    df["career_prior_carries"] = trailing(df, "player_id", "carries", None)

    df["season_td_attempts"] = df.groupby(["player_id", "season"])["attempts"].transform(
        lambda s: s.shift(1).expanding().mean())
    df["season_td_attempts"] = df["season_td_attempts"].fillna(df["career_prior_attempts"])
    df["season_games_prior"] = df.groupby(["player_id", "season"]).cumcount()

    df["days_rest"] = (df.groupby("player_id")["kickoff_timestamp_utc"]
                       .diff().dt.total_seconds() / 86400.0).clip(0, 30)
    df["is_home"] = df["m_is_home"].astype(float) if "m_is_home" in df.columns else np.nan
    df["is_post"] = (df.season_type == "POST").astype(float)
    return df


def add_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 2: player-window -> player-career -> league, three levels."""
    df = df.copy()
    specs = {
        "pass_yds": ("passing_yards", "career_prior_pass_yds"),
        "attempts": ("attempts", "career_prior_attempts"),
        "ypa": ("ypa_game", "career_ypa"),
    }
    ks = {}
    for name, (raw_col, career_col) in specs.items():
        prior = league_prior(df, raw_col)
        k1 = estimate_shrinkage_k(df, raw_col)
        ks[f"{name}_k_career"] = k1
        n_career = df["games_played_prior"].values
        career_shrunk = shrink(df[career_col].values, n_career, prior, k1)
        df[f"{name}_career_shrunk"] = career_shrunk
        df[f"{name}_league_prior"] = prior

        for w in WINDOWS:
            col = f"{name}_w{w}" if name != "pass_yds" else f"pass_yds_w{w}"
            n_w = np.minimum(n_career, w)
            df[f"{col}_shrunk"] = shrink(df[col].values, n_w, career_shrunk, k1)
    df.attrs["shrinkage_k"] = ks
    return df


# ---------------------------------------------------------------------------
# Layers 3 & 4 - team environment and opponent
# ---------------------------------------------------------------------------
def load_team_game_ledger(kickoffs: pd.DataFrame) -> pd.DataFrame:
    """pbp team-game ledger, canonicalised and stamped with kickoff time."""
    files = sorted(glob.glob(str(C.PBP_AGG / "team_game_ledger_*.parquet")))
    led = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    from .panel import _canon_map
    canon = _canon_map()
    led["team_canon"] = led["team"].map(canon)
    led = led[led.team_canon.notna()]
    pre = led.season < AIR_YARDS_ERA
    for c in ["off_adot", "def_adot"]:
        led.loc[pre, c] = np.nan
    led = led.merge(kickoffs, on=["season", "week", "team_canon"], how="left")
    led = led.sort_values(["team_canon", "season", "week"]).reset_index(drop=True)
    return led


OFF_METRICS = ["off_plays", "off_pass_rate", "off_neutral_pass_rate",
               "off_early_pass_rate", "off_rz_pass_rate", "off_sec_per_play",
               "off_epa_per_play", "off_pass_epa_per_db", "off_rush_epa_per_carry",
               "off_success_rate", "off_points", "off_expl_pass_rate",
               "off_sack_rate", "off_yds_per_pass_play", "off_pass_plays",
               "off_pass_yds", "off_adot", "off_comp_rate"]

DEF_METRICS = ["def_plays", "def_pass_rate", "def_sec_per_play",
               "def_epa_per_play", "def_pass_epa_per_db", "def_rush_epa_per_carry",
               "def_success_rate", "def_expl_pass_rate", "def_sack_rate",
               "def_qb_hit_rate", "def_yds_per_pass_play", "def_pass_plays",
               "def_pass_yds", "def_comp_rate", "def_adot"]

TEAM_WINDOWS = [5]


def add_team_trailing(led: pd.DataFrame) -> pd.DataFrame:
    """Trailing team form on the team's own game sequence, shift(1)."""
    led = led.sort_values(["team_canon", "season", "week"]).reset_index(drop=True)
    for m in OFF_METRICS + DEF_METRICS:
        for w in TEAM_WINDOWS:
            led[f"{m}_w{w}"] = trailing(led, "team_canon", m, w)
        led[f"{m}_s2d"] = led.groupby(["team_canon", "season"])[m].transform(
            lambda s: s.shift(1).expanding().mean())
    # Latest kickoff feeding each row's trailing window, for the leakage assert.
    led["prev_kickoff"] = led.groupby("team_canon")["kickoff"].shift(1)
    return led


def attach_context(df: pd.DataFrame, led: pd.DataFrame) -> pd.DataFrame:
    """Join own-team offence and opponent defence onto each player row."""
    off_cols = [f"{m}_w{w}" for m in OFF_METRICS for w in TEAM_WINDOWS] + \
               [f"{m}_s2d" for m in OFF_METRICS]
    def_cols = [f"{m}_w{w}" for m in DEF_METRICS for w in TEAM_WINDOWS] + \
               [f"{m}_s2d" for m in DEF_METRICS]

    own = led[["season", "week", "team_canon", "prev_kickoff"] + off_cols].rename(
        columns={"prev_kickoff": "own_prev_kickoff"})
    opp = led[["season", "week", "team_canon", "prev_kickoff"] + def_cols].rename(
        columns={"team_canon": "opp_canon", "prev_kickoff": "opp_prev_kickoff"})

    out = df.merge(own, on=["season", "week", "team_canon"], how="left")
    out = out.merge(opp, on=["season", "week", "opp_canon"], how="left")
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 3 matchup interactions and the expected-volume chain."""
    df = df.copy()

    # Raw products, not products of z-scores: a panel-wide mean/std would pull
    # future seasons into a feature for a past row. Per-fold standardisation in
    # the linear models handles scale, and trees are scale-free.
    def x(a, b):
        return df[a].astype(float) * df[b].astype(float)

    df["x_qb_epa_vs_def_epa"] = x("epa_att_w5", "def_pass_epa_per_db_w5")
    df["x_qb_ypa_vs_def_ypp"] = x("ypa_w5", "def_yds_per_pass_play_w5")
    df["x_qb_sack_vs_def_sack"] = x("sack_rate_w5", "def_sack_rate_w5")
    df["x_qb_yds_vs_def_yds"] = x("pass_yds_w5", "def_pass_yds_w5")
    df["x_qb_adot_vs_def_adot"] = x("adot_w5", "def_adot_w5")
    df["x_pace_vs_def_pace"] = x("off_sec_per_play_w5", "def_sec_per_play_w5")
    df["x_passrate_vs_def_plays"] = x("off_pass_rate_w5", "def_plays_w5")

    # Expected volume: blend own offence with what the defence has allowed,
    # then take the QB's own trailing share of team dropbacks.
    df["exp_plays"] = df[["off_plays_w5", "def_plays_w5"]].mean(axis=1)
    df["exp_pass_rate"] = df[["off_pass_rate_w5", "def_pass_rate_w5"]].mean(axis=1)
    df["exp_team_dropbacks"] = df["exp_plays"] * df["exp_pass_rate"]
    df["qb_dropback_share_w5"] = (df["attempts_w5"] /
                                  df["off_pass_plays_w5"].replace(0, np.nan)).clip(0, 1.2)
    df["exp_attempts"] = df["exp_team_dropbacks"] * df["qb_dropback_share_w5"]
    df["exp_pass_yds"] = df["exp_attempts"] * df["ypa_w5"]
    df["exp_pass_yds_shrunk"] = df["exp_attempts"] * df["ypa_w5_shrunk"]
    df["exp_pass_yds_matchup"] = df["exp_attempts"] * (
        0.5 * df["ypa_w5"] + 0.5 * df["def_yds_per_pass_play_w5"])
    return df


PLAYER_FEATURES = (
    [f"pass_yds_w{w}" for w in WINDOWS] + [f"attempts_w{w}" for w in WINDOWS] +
    [f"ypa_w{w}" for w in WINDOWS] + [f"comp_pct_w{w}" for w in WINDOWS] +
    [f"epa_att_w{w}" for w in WINDOWS] + [f"sack_rate_w{w}" for w in WINDOWS] +
    [f"int_rate_w{w}" for w in WINDOWS] + [f"td_rate_w{w}" for w in WINDOWS] +
    [f"adot_w{w}" for w in WINDOWS] + [f"carries_w{w}" for w in WINDOWS] +
    [f"rush_yds_w{w}" for w in WINDOWS] + [f"first_downs_w{w}" for w in WINDOWS] +
    ["pass_yds_std5", "attempts_std5", "pass_yds_std8",
     "career_ypa", "career_comp_pct", "career_sack_rate", "career_epa_att",
     "career_prior_carries", "season_td_attempts", "season_games_prior",
     "days_rest", "is_home", "is_post", "games_played_prior", "week"] +
    C.V1_FEATURES
)

SHRINK_FEATURES = (
    [f"pass_yds_w{w}_shrunk" for w in WINDOWS] +
    [f"attempts_w{w}_shrunk" for w in WINDOWS] +
    [f"ypa_w{w}_shrunk" for w in WINDOWS] +
    ["pass_yds_career_shrunk", "attempts_career_shrunk", "ypa_career_shrunk",
     "pass_yds_league_prior", "attempts_league_prior", "ypa_league_prior"]
)

TEAM_FEATURES = [f"{m}_w{w}" for m in OFF_METRICS for w in TEAM_WINDOWS] + \
                [f"{m}_s2d" for m in OFF_METRICS]

OPP_FEATURES = [f"{m}_w{w}" for m in DEF_METRICS for w in TEAM_WINDOWS] + \
               [f"{m}_s2d" for m in DEF_METRICS]

INTERACTION_FEATURES = [
    "x_qb_epa_vs_def_epa", "x_qb_ypa_vs_def_ypp", "x_qb_sack_vs_def_sack",
    "x_qb_yds_vs_def_yds", "x_qb_adot_vs_def_adot", "x_pace_vs_def_pace",
    "x_passrate_vs_def_plays", "exp_plays", "exp_pass_rate",
    "exp_team_dropbacks", "qb_dropback_share_w5", "exp_attempts",
    "exp_pass_yds", "exp_pass_yds_shrunk", "exp_pass_yds_matchup",
]

FEATURE_GROUPS = {
    "player_state": PLAYER_FEATURES,
    "shrinkage": SHRINK_FEATURES,
    "team_environment": TEAM_FEATURES,
    "opponent": OPP_FEATURES,
    "interactions": INTERACTION_FEATURES,
}

V2_FEATURES = (PLAYER_FEATURES + SHRINK_FEATURES + TEAM_FEATURES +
               OPP_FEATURES + INTERACTION_FEATURES)


def build_v2_panel(return_full: bool = False):
    """Full V2 panel on exactly V1's eligible rows."""
    from . import panel as P

    qb = P.build_qb_panel()
    qb = mask_air_yards_era(qb)
    qb = harmonize_schema_split(qb)
    qb = add_player_state(qb)
    qb = add_shrinkage(qb)

    kickoffs = (qb.groupby(["season", "week", "team_canon"])["kickoff_timestamp_utc"]
                .min().rename("kickoff").reset_index())
    led = load_team_game_ledger(kickoffs)
    led = add_team_trailing(led)

    qb = attach_context(qb, led)
    qb = add_interactions(qb)
    eligible = P.apply_v1_eligibility(qb)
    if return_full:
        return eligible, led, qb
    return eligible, led
