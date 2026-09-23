"""Causal possession-TD hazard, fail-closed.

P(offensive TD on a possession | block yards, plays, offense-perspective score differential, clock) fit by unpenalized logistic regression on the frozen
drive-block table, strictly before the simulated game's season-week (same window as the V27 FG estimator).  The table is hash-verified before it is read; any problem
raises HazardError -- there is no default rate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from worker.sports_nova_v27_causal_fg_rate import causal_fg as fg27
from .config import (ALIGN_GRID, DIFF_CLIP, DRIVE_RELPATH, DRIVE_SHA256, PLAYER_RELPATH, PLAYER_SHA256, PLAYS_CLIP, SUPPORT_PCT, YARD_KNOTS)

ROOT = fg27.ROOT
_COLS = ["game_id", "season", "week", "touchdowns", "plays", "pass_yards", "rush_yards", "start_score_diff", "start_game_seconds_remaining"]
_T: dict = {}


class HazardError(RuntimeError):
    """The TD hazard could not be produced; the simulation must not run."""


def drives() -> pd.DataFrame:
    if "d" not in _T:
        path = ROOT / DRIVE_RELPATH
        try:
            got = fg27._sha256(path)
        except OSError as e:
            raise HazardError(f"drive table unavailable: {e}") from e
        if got != DRIVE_SHA256:
            raise HazardError(f"drive table hash mismatch: {got}")
        d = pd.read_parquet(path, columns=_COLS)
        d["key"] = d["season"].astype(int) * 100 + d["week"].astype(int)
        d["yds"] = d["pass_yards"].astype(float) + d["rush_yards"].astype(float)
        d["diff"] = d["start_score_diff"].astype(float).fillna(0.0)
        d["sec"] = d["start_game_seconds_remaining"].astype(float)
        d["td"] = (d["touchdowns"] > 0).astype(int)
        _T["d"] = d
    return _T["d"]


def league_td_rates(season: int, week: int) -> tuple[float, float]:
    """Pooled TD/reception and TD/carry over the same causal window (hash-verified player table).  Reference level for the team finishing-propensity multiplier."""
    if "pl" not in _T:
        path = ROOT / PLAYER_RELPATH
        try:
            got = fg27._sha256(path)
        except OSError as e:
            raise HazardError(f"player table unavailable: {e}") from e
        if got != PLAYER_SHA256:
            raise HazardError(f"player table hash mismatch: {got}")
        pl = pd.read_parquet(path, columns=["SEASON", "WEEK", "RECEPTIONS", "PASS_TD", "RUSH_ATTEMPTS", "RUSH_TD"])
        pl["key"] = pl["SEASON"].astype(int) * 100 + pl["WEEK"].astype(int)
        _T["pl"] = pl
    pl = _T["pl"]
    w = pl[(pl["key"] < int(season) * 100 + int(week)) & (pl["SEASON"] >= max(1999, int(season) - 5))]
    rec, car = float(w["RECEPTIONS"].sum()), float(w["RUSH_ATTEMPTS"].sum())
    if len(w) == 0 or rec <= 0 or car <= 0:
        raise HazardError("empty league TD-rate window")
    lp, lr = float(w["PASS_TD"].sum()) / rec, float(w["RUSH_TD"].sum()) / car
    if not (0.0 < lp < 0.3 and 0.0 < lr < 0.2):
        raise HazardError(f"implausible league TD rates {lp} {lr}")
    return lp, lr


def recent_form_slope(season: int, week: int) -> float:
    """Causal empirical slope of a team's NEXT-game total yards (relative to that week's league mean) on its recent_form (trailing-8-game pass yards relative to the league), over the
    same 5-season window.  recent_form enters the simulator as a 1:1 efficiency multiplier; this is how much of it history says persists.  Clipped to [0, 1]."""
    league_td_rates(season, week)                                              # loads + verifies the player table
    if "tg" not in _T:
        d = pd.read_parquet(ROOT / PLAYER_RELPATH, columns=["SEASON", "WEEK", "TEAM", "PASS_YARDS", "RUSH_YARDS"])
        tg = d.groupby(["TEAM", "SEASON", "WEEK"], as_index=False).agg(py=("PASS_YARDS", "sum"), ry=("RUSH_YARDS", "sum")).sort_values(["TEAM", "SEASON", "WEEK"])
        tg["ty"] = tg["py"] + tg["ry"]
        tg["py8"] = tg.groupby("TEAM")["py"].transform(lambda x: x.shift(1).rolling(8, min_periods=8).mean())
        tg["key"] = tg["SEASON"].astype(int) * 100 + tg["WEEK"].astype(int)
        tg["rf"] = tg["py8"] / tg.groupby("key")["py8"].transform("mean") - 1.0
        tg["target"] = tg["ty"] / tg.groupby("key")["ty"].transform("mean") - 1.0
        _T["tg"] = tg
    tg = _T["tg"]
    w = tg[(tg["key"] < int(season) * 100 + int(week)) & (tg["SEASON"] >= max(1999, int(season) - 5))].dropna(subset=["rf"])
    if len(w) < 200 or float(w["rf"].var()) <= 0:
        raise HazardError("empty recent_form slope window")
    slope = float(np.cov(w["rf"], w["target"])[0, 1] / w["rf"].var())
    if not np.isfinite(slope):
        raise HazardError("non-finite recent_form slope")
    return float(np.clip(slope, 0.0, 1.0))


def window(season: int, week: int) -> pd.DataFrame:
    d = drives()
    cutoff = int(season) * 100 + int(week)
    p = d[(d["key"] < cutoff) & (d["season"] >= max(1999, int(season) - 5))]
    if len(p) == 0 or int(p["key"].max()) >= cutoff:
        raise HazardError("empty or temporally invalid hazard window")
    return p


def design(yards, plays, diff, sec, lo: float, hi: float) -> np.ndarray:
    y = np.clip(np.asarray(yards, dtype=float), lo, hi)
    p = np.clip(np.asarray(plays, dtype=float), 0, PLAYS_CLIP)
    dd = np.clip(np.asarray(diff, dtype=float), -DIFF_CLIP, DIFF_CLIP) / 10.0
    s = np.asarray(sec, dtype=float)
    sh = np.where(s > 1800, s - 1800, s)
    cols = [y / 100.0] + [np.maximum(y - k, 0.0) / 100.0 for k in YARD_KNOTS]
    cols += [p / 10.0, (p == 1).astype(float), (p == 2).astype(float), (p == 3).astype(float)]
    cols += [dd, np.maximum(dd, 0.0)]
    cols += [(sh < 120).astype(float), (sh < 300).astype(float), (s <= 1800).astype(float)]
    return np.column_stack(cols)


@dataclass(frozen=True)
class Hazard:
    coef: np.ndarray
    intercept: float
    lo: float
    hi: float
    season: int
    week: int
    n_train: int
    train_rate: float
    max_key_used: int
    align_ref: tuple | None = None           # (ref_yard_quantiles, hist_yard_quantiles, grid) for the POST_HOC alignment; None = raw hazard
    league_pass_td_rate: float = float('nan')   # pooled TD/reception over the same causal window (reference for the team finishing multiplier)
    league_rush_td_rate: float = float('nan')   # pooled TD/carry

    def map_yards(self, yards):
        if self.align_ref is None:
            return np.asarray(yards, dtype=float)
        ref_q, hist_q, grid = self.align_ref
        u = np.interp(np.asarray(yards, dtype=float), ref_q, grid)
        return np.interp(u, grid, hist_q)

    def p_td(self, yards, plays, diff, sec):
        scalar = np.ndim(yards) == 0
        x = design(np.atleast_1d(self.map_yards(yards)), np.atleast_1d(plays), np.atleast_1d(diff), np.atleast_1d(sec), self.lo, self.hi)
        z = x @ self.coef + self.intercept
        out = 1.0 / (1.0 + np.exp(-z))
        return float(out[0]) if scalar else out

    def evidence(self) -> dict:
        return {"league_pass_td_rate": self.league_pass_td_rate, "league_rush_td_rate": self.league_rush_td_rate, "hazard_season": self.season, "hazard_week": self.week, "hazard_n_train": self.n_train, "hazard_train_rate": self.train_rate,
                "hazard_max_key_used": self.max_key_used, "hazard_yard_support": [self.lo, self.hi], "hazard_aligned": self.align_ref is not None}


def _fit(season: int, week: int) -> Hazard:
    from sklearn.linear_model import LogisticRegression
    p = window(season, week)
    lo, hi = (float(np.percentile(p["yds"], q)) for q in SUPPORT_PCT)
    x = design(p["yds"].to_numpy(), p["plays"].to_numpy(), p["diff"].to_numpy(), p["sec"].to_numpy(), lo, hi)
    y = p["td"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-9).fit(x, y)
    coef, b = m.coef_[0].astype(float), float(m.intercept_[0])
    if not (np.isfinite(coef).all() and np.isfinite(b)):
        raise HazardError("non-finite hazard coefficients")
    pr = 1.0 / (1.0 + np.exp(-(x @ coef + b)))
    if abs(float(pr.mean()) - float(y.mean())) > 1e-4:                       # unpenalized intercept => fitted mean equals observed rate
        raise HazardError("hazard did not converge (fitted mean != observed rate)")
    lp, lr = league_td_rates(int(season), int(week))
    return Hazard(coef, b, lo, hi, int(season), int(week), int(len(p)), float(y.mean()), int(p["key"].max()), None, lp, lr)


@lru_cache(maxsize=None)
def hazard_for(season: int, week: int) -> Hazard:
    return _fit(int(season), int(week))


def hazard_for_game(game_id: str, align: bool = False, ref_quantiles=None) -> Hazard:
    season, week = fg27.parse_game_key(game_id)
    h = hazard_for(season, week)
    if not align:
        return h
    if ref_quantiles is None:
        raise HazardError("alignment requested without a frozen sim-yardage reference")
    p = window(season, week)
    grid = np.linspace(0.0, 1.0, ALIGN_GRID)
    hist_q = np.quantile(p["yds"].to_numpy(), grid)
    ref_q = np.asarray(ref_quantiles, dtype=float)
    if len(ref_q) != ALIGN_GRID or not np.all(np.diff(ref_q) >= 0):
        raise HazardError("invalid alignment reference")
    ref_q = ref_q + np.arange(ALIGN_GRID) * 1e-9                              # strictly increasing for np.interp
    return Hazard(h.coef, h.intercept, h.lo, h.hi, h.season, h.week, h.n_train, h.train_rate, h.max_key_used, (ref_q, hist_q, grid), h.league_pass_td_rate, h.league_rush_td_rate)
