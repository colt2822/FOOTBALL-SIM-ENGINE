"""Walk-forward driver for the V1-vs-V2 comparison.

Chronology discipline: model family, hyperparameters and distribution method
are chosen on DEV (2006-2019). TEST (2020-2025) is scored once, with the
choices frozen. Pooled 2006-2025 is reported for continuity with V1's
published number.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import distribution as D
from . import metrics as M
from .walkforward import assemble, walk_forward

N_DRAWS = 400


def run_candidate(el: pd.DataFrame, fit_predict, resid_mode: str = "train",
                  hetero: bool = True, calibration: str = "isotonic",
                  threshold: float = C.PROB_THRESHOLD, seed: int = C.SEED,
                  seasons: list[int] | None = None):
    """One full walk-forward pass, producing point predictions and calibrated
    exceedance probabilities for every OOS row."""
    rng = np.random.default_rng(seed)
    folds = walk_forward(el, fit_predict, seasons=seasons)
    assert all(f.cutoff_ok for f in folds), "temporal cutoff violated"

    prior_oos_resid: list[np.ndarray] = []
    cal_p: list[np.ndarray] = []
    cal_y: list[np.ndarray] = []

    blocks = []
    for f in folds:
        train = el[el.season < f.season]
        test = el.loc[f.test_index]
        y_te = test[C.TARGET].values

        if resid_mode == "prior_oos" and prior_oos_resid:
            pool = np.concatenate(prior_oos_resid)
        else:
            pool = f.train_resid

        if hetero:
            scale = D.ScaleModel().fit(train, f.train_resid)
            sig_tr = scale.sigma(train)
            z_pool = f.train_resid / np.clip(sig_tr, 1e-6, None)
            if resid_mode == "prior_oos" and prior_oos_resid:
                z_pool = np.concatenate(prior_oos_resid)
            sigma_te = scale.sigma(test)
            p_raw = D.p_over_hetero(f.pred, sigma_te, z_pool, threshold)
            draws = D.samples_hetero(f.pred, sigma_te, z_pool, N_DRAWS, rng)
        else:
            sigma_te = np.full(len(test), np.std(pool))
            p_raw = D.p_over_scalar(f.pred, pool, threshold)
            idx = rng.integers(0, len(pool), size=(len(test), N_DRAWS))
            draws = f.pred[:, None] + pool[idx]

        # Calibrator sees only earlier OOS seasons.
        cal = D.CausalCalibrator(calibration)
        if cal_p:
            cal.fit(np.concatenate(cal_p), np.concatenate(cal_y))
        p_cal = cal.transform(p_raw)

        y_bin = (y_te > threshold).astype(int)
        cal_p.append(p_raw)
        cal_y.append(y_bin)

        resid_te = y_te - f.pred
        prior_oos_resid.append(resid_te / np.clip(sigma_te, 1e-6, None)
                               if hetero else resid_te)

        block = test[["player_id", "season", "week", "games_played_prior",
                      C.TARGET]].copy()
        block["pred"] = f.pred
        block["sigma"] = sigma_te
        block["p_raw"] = p_raw
        block["p_cal"] = p_cal
        block["oos_season"] = f.season
        block["q10"] = np.quantile(draws, 0.10, axis=1)
        block["q25"] = np.quantile(draws, 0.25, axis=1)
        block["q50"] = np.quantile(draws, 0.50, axis=1)
        block["q75"] = np.quantile(draws, 0.75, axis=1)
        block["q90"] = np.quantile(draws, 0.90, axis=1)
        block["crps"] = _crps_rows(y_te, draws)

        # Threshold sweep, so calibration can be judged across the range of
        # lines that actually get posted rather than at one point.
        for t in C.THRESH_LIST_MAIN:
            if hetero:
                pt = D.p_over_hetero(f.pred, sigma_te, z_pool, float(t))
            else:
                pt = D.p_over_scalar(f.pred, pool, float(t))
            block[f"p_over_{t}"] = pt
        blocks.append(block)

    return pd.concat(blocks).sort_index(), folds


def _crps_rows(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    t1 = np.mean(np.abs(s - np.asarray(y)[:, None]), axis=1)
    w = (2 * np.arange(1, n + 1) - n - 1)
    t2 = (2.0 / (n * n)) * (s * w).sum(axis=1)
    return t1 - 0.5 * t2


def score(oos: pd.DataFrame, seasons: list[int] | None = None,
          prob_col: str = "p_cal", threshold: float = C.PROB_THRESHOLD) -> dict:
    d = oos if seasons is None else oos[oos.oos_season.isin(seasons)]
    if len(d) == 0:
        return {}
    out = M.score_block(d[C.TARGET].values, d.pred.values, d[prob_col].values,
                        threshold)
    out["CRPS"] = float(d.crps.mean())
    out["PINBALL"] = M.pinball_loss(
        d[C.TARGET].values,
        {0.1: d.q10.values, 0.25: d.q25.values, 0.5: d.q50.values,
         0.75: d.q75.values, 0.9: d.q90.values})
    out["QUANTILE_COVERAGE"] = M.quantile_coverage(
        d[C.TARGET].values,
        {0.1: d.q10.values, 0.25: d.q25.values, 0.5: d.q50.values,
         0.75: d.q75.values, 0.9: d.q90.values})
    return out


def score_all_periods(oos: pd.DataFrame, prob_col: str = "p_cal") -> dict:
    return {
        "DEV_2006_2019": score(oos, C.DEV_SEASONS, prob_col),
        "TEST_2020_2025": score(oos, C.TEST_SEASONS, prob_col),
        "POOLED_2006_2025": score(oos, None, prob_col),
    }


def improvement(v1: dict, v2: dict) -> dict:
    out = {}
    for k in ("MAE", "RMSE", "BRIER", "LOGLOSS", "ECE", "CRPS"):
        if k in v1 and k in v2 and v1[k]:
            out[f"{k}_IMPROVEMENT_PCT"] = round(100.0 * (v1[k] - v2[k]) / v1[k], 3)
    return out
