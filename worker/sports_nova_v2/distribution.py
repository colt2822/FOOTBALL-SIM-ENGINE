"""Layer 7/8: turn a point prediction into a calibrated distribution.

Two residual regimes are compared:

  homoskedastic  - one pooled residual sample, V1's approach.
  heteroskedastic- residuals standardised by a predicted scale, so a QB
                   expected to throw 45 times gets a wider distribution than
                   one expected to throw 20.

Residual pools are always sourced causally: the pool used for OOS season s is
built only from seasons strictly before s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge

from . import config as C
from .models import ImputeStandardize

SCALE_FEATURES = ["exp_attempts", "attempts_w5", "attempts_w3", "pass_yds_std5",
                  "career_prior_attempts", "off_plays_w5", "exp_team_dropbacks"]


class ScaleModel:
    """Predicts residual magnitude, giving a per-row sigma.

    Fit on training residuals only. sqrt(pi/2) rescales mean-absolute-error to
    a standard deviation under a Gaussian, keeping sigma on a familiar scale.
    """

    def __init__(self, features: list[str] | None = None, alpha: float = 10.0):
        self.features = features or SCALE_FEATURES
        self.alpha = alpha

    def fit(self, train: pd.DataFrame, resid: np.ndarray):
        feats = [f for f in self.features if f in train.columns]
        self.used_ = feats
        self.prep_ = ImputeStandardize().fit(train[feats])
        self.m_ = Ridge(alpha=self.alpha).fit(
            self.prep_.transform(train[feats]), np.abs(resid))
        self.floor_ = float(np.percentile(np.abs(resid), 5)) or 1.0
        self.mean_abs_ = float(np.mean(np.abs(resid)))
        return self

    def sigma(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.m_.predict(self.prep_.transform(df[self.used_]))
        return np.clip(raw, self.floor_, None) * np.sqrt(np.pi / 2)


def empirical_p_over(pred: np.ndarray, resid_pool: np.ndarray,
                     thresholds: np.ndarray) -> np.ndarray:
    """P(pred + resid > t). Non-increasing in t by construction."""
    draws = pred[:, None] + resid_pool[None, :]
    return (draws[:, :, None] > thresholds[None, None, :]).mean(axis=1)


def p_over_scalar(pred: np.ndarray, resid_pool: np.ndarray,
                  threshold: float) -> np.ndarray:
    return np.array([(p + resid_pool > threshold).mean() for p in pred])


def p_over_hetero(pred: np.ndarray, sigma: np.ndarray, z_pool: np.ndarray,
                  threshold: float) -> np.ndarray:
    return np.array([((p + s * z_pool) > threshold).mean()
                     for p, s in zip(pred, sigma)])


def samples_hetero(pred: np.ndarray, sigma: np.ndarray,
                   z_pool: np.ndarray, n_draws: int, rng) -> np.ndarray:
    idx = rng.integers(0, len(z_pool), size=(len(pred), n_draws))
    return pred[:, None] + sigma[:, None] * z_pool[idx]


def dense_curve(point: float, sample: np.ndarray, lo=C.CURVE_LO,
                hi=C.CURVE_HI, step=C.CURVE_STEP) -> dict:
    grid = np.arange(lo, hi + step, step)
    draws = point + sample
    p = (draws[:, None] > grid[None, :]).mean(axis=0)
    # Guard against float noise breaking the monotone guarantee.
    p = np.minimum.accumulate(p)
    return {str(round(float(t), 1)): float(v) for t, v in zip(grid, p)}


def summary_stats(point: float, sample: np.ndarray) -> dict:
    draws = point + sample
    return {
        "MEAN": float(draws.mean()), "MEDIAN": float(np.median(draws)),
        "STD": float(draws.std()),
        "P10": float(np.quantile(draws, .10)), "P25": float(np.quantile(draws, .25)),
        "P50": float(np.quantile(draws, .50)), "P75": float(np.quantile(draws, .75)),
        "P90": float(np.quantile(draws, .90)),
    }


# ---------------------------------------------------------------------------
# Layer 8 - calibration, fit causally on earlier OOS seasons only
# ---------------------------------------------------------------------------
class CausalCalibrator:
    """Maps raw probabilities to calibrated ones.

    For OOS season s the mapping is fit on pooled OOS rows from seasons before
    s. Those predictions came from models that never saw season s, so nothing
    from the evaluation period reaches the calibrator.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.fitted_ = False

    def fit(self, p: np.ndarray, y: np.ndarray, min_n: int = 400):
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
        if len(p) < min_n or len(np.unique(y)) < 2:
            return self
        if self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.m_.fit(p, y)
        elif self.method == "platt":
            self.m_ = LogisticRegression(C=1e6, solver="lbfgs")
            self.m_.fit(p.reshape(-1, 1), y)
        else:
            raise ValueError(self.method)
        self.fitted_ = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if not self.fitted_:
            return p
        if self.method == "isotonic":
            return np.clip(self.m_.predict(p), 0.0, 1.0)
        return self.m_.predict_proba(p.reshape(-1, 1))[:, 1]
