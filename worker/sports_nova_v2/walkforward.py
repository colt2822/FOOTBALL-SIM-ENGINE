"""Chronological walk-forward: for OOS season s, train on every eligible row
from a season strictly before s. Expanding window, matching V1."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class FoldResult:
    season: int
    train_n: int
    test_n: int
    cutoff_ok: bool
    test_index: np.ndarray
    pred: np.ndarray
    train_resid: np.ndarray
    extra: dict = field(default_factory=dict)


def walk_forward(eligible: pd.DataFrame, fit_predict, seasons=None,
                 min_train: int = 50):
    """`fit_predict(train_df, test_df) -> (pred_test, pred_train, extra)`."""
    seasons = seasons or [s for s in sorted(eligible.season.unique())
                          if s >= C.OOS_START_SEASON]
    folds = []
    for s in seasons:
        train = eligible[eligible.season < s]
        test = eligible[eligible.season == s]
        if len(train) < min_train or len(test) == 0:
            continue
        pred_te, pred_tr, extra = fit_predict(train, test)
        cutoff_ok = bool(train.kickoff_timestamp_utc.max() < test.kickoff_timestamp_utc.min())
        folds.append(FoldResult(
            season=int(s), train_n=int(len(train)), test_n=int(len(test)),
            cutoff_ok=cutoff_ok, test_index=test.index.values,
            pred=np.asarray(pred_te, dtype=float),
            train_resid=np.asarray(train[C.TARGET].values - pred_tr, dtype=float),
            extra=extra or {},
        ))
    return folds


def assemble(eligible: pd.DataFrame, folds: list[FoldResult]) -> pd.DataFrame:
    """Stitch folds into one OOS frame, carrying the fold id for later
    season-scoped residual work."""
    parts = []
    for f in folds:
        block = eligible.loc[f.test_index].copy()
        block["pred"] = f.pred
        block["oos_season"] = f.season
        parts.append(block)
    return pd.concat(parts).sort_index()
