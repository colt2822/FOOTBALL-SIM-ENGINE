"""Layer 6 candidate models.

Each model exposes the same `fit_predict(train, test)` contract used by the
walk-forward driver, so V1 and every V2 variant are scored on identical rows
by identical machinery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config as C

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:  # pragma: no cover
    HAVE_LGB = False


class ImputeStandardize:
    """Train-fitted median impute + z-scale, with explicit missingness flags.

    Rows are never dropped for missing features: dropping would silently remove
    the low-history QBs that are the hardest and most valuable to predict.
    """

    def __init__(self, indicator_threshold: float = 0.005):
        self.indicator_threshold = indicator_threshold

    def fit(self, X: pd.DataFrame):
        self.cols_ = list(X.columns)
        self.median_ = X.median(numeric_only=True)
        self.median_ = self.median_.reindex(self.cols_).fillna(0.0)
        miss_rate = X.isna().mean()
        self.indicator_cols_ = [c for c in self.cols_
                                if miss_rate[c] > self.indicator_threshold]
        Z = self._raw(X)
        self.mu_ = Z.mean(axis=0)
        self.sigma_ = Z.std(axis=0) + 1e-6
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        base = X[self.cols_].fillna(self.median_)
        if self.indicator_cols_:
            ind = X[self.indicator_cols_].isna().astype(float)
            return np.hstack([base.values, ind.values])
        return base.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return (self._raw(X) - self.mu_) / self.sigma_

    def feature_names(self) -> list[str]:
        return self.cols_ + [f"{c}__isna" for c in self.indicator_cols_]


def make_ridge(features: list[str], alpha: float = 10.0):
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
        prep = ImputeStandardize().fit(train[features])
        Xtr, Xte = prep.transform(train[features]), prep.transform(test[features])
        m = Ridge(alpha=alpha).fit(Xtr, train[C.TARGET].values)
        return m.predict(Xte), m.predict(Xtr), {"model": m, "prep": prep}
    return fit_predict


def make_lgbm(features: list[str], params: dict | None = None,
              n_estimators: int = 500):
    base = dict(
        objective="l2", learning_rate=0.04, num_leaves=15, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_lambda=5.0, max_depth=5, verbose=-1, n_jobs=4, seed=C.SEED,
    )
    base.update(params or {})

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
        m = lgb.LGBMRegressor(n_estimators=n_estimators, **base)
        m.fit(train[features], train[C.TARGET].values)
        return m.predict(test[features]), m.predict(train[features]), {"model": m}
    return fit_predict


def make_blend(features: list[str], alpha: float = 10.0, weight: float = 0.5,
               params: dict | None = None, n_estimators: int = 500):
    """Convex blend of the regularised linear model and the GBM.

    Linear extrapolates sensibly for unusual feature values; the GBM captures
    matchup non-linearities. The blend weight is chosen on DEV only.
    """
    ridge_fp = make_ridge(features, alpha)
    lgbm_fp = make_lgbm(features, params, n_estimators)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
        r_te, r_tr, _ = ridge_fp(train, test)
        g_te, g_tr, _ = lgbm_fp(train, test)
        return (weight * g_te + (1 - weight) * r_te,
                weight * g_tr + (1 - weight) * r_tr, {})
    return fit_predict


def lgbm_gain_importance(train: pd.DataFrame, features: list[str],
                         params: dict | None = None) -> pd.Series:
    fp = make_lgbm(features, params)
    _, _, extra = fp(train, train.head(1))
    m = extra["model"]
    return pd.Series(m.booster_.feature_importance("gain"),
                     index=features).sort_values(ascending=False)
