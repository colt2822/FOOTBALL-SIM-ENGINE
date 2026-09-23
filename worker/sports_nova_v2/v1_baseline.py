"""Exact reimplementation of the V1 champion, run inside the V2 harness.

Purpose is the reproduction gate: unless this returns V1's published
MAE/RMSE/OOS_N, no V1-vs-V2 comparison produced here is trustworthy.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config as C
from . import metrics as M
from . import panel as P
from .walkforward import assemble, walk_forward


def v1_fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    Xtr, ytr = train[C.V1_FEATURES].values, train[C.TARGET].values
    Xte = test[C.V1_FEATURES].values
    mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    model = Ridge(alpha=C.RIDGE_ALPHA)
    model.fit((Xtr - mu) / sigma, ytr)
    return model.predict((Xte - mu) / sigma), model.predict((Xtr - mu) / sigma), {}


def v1_pooled_residual_probs(oos: pd.DataFrame, folds, threshold: float):
    """V1's probability rule: P(over) from that fold's *training* residual pool."""
    resid_by_season = {f.season: f.train_resid for f in folds}
    return np.array([
        float((pred + resid_by_season[int(s)] > threshold).mean())
        for pred, s in zip(oos.pred.values, oos.oos_season.values)
    ])


def run_v1() -> dict:
    qb = P.build_qb_panel()
    eligible = P.apply_v1_eligibility(qb)
    folds = walk_forward(eligible, v1_fit_predict)
    assert all(f.cutoff_ok for f in folds), "temporal cutoff violated in V1 repro"
    oos = assemble(eligible, folds)

    p_over = v1_pooled_residual_probs(oos, folds, C.PROB_THRESHOLD)
    y = oos[C.TARGET].values
    block = M.score_block(y, oos.pred.values, p_over, C.PROB_THRESHOLD)

    baselines = {
        "trailing_4_mean": M.mae(y, oos.trailing4_pass_yds.values),
        "season_to_date_mean": M.mae(y, oos.season_td_pass_yds.values),
        "player_historical_prior": M.mae(y, oos.career_prior_pass_yds.values),
    }
    best = min(baselines, key=baselines.get)

    pub = C.V1_PUBLISHED
    repro = {
        "OOS_N_match": block["N"] == pub["OOS_N"],
        "MAE_abs_diff": abs(block["MAE"] - pub["MODEL_MAE"]),
        "RMSE_abs_diff": abs(block["RMSE"] - pub["MODEL_RMSE"]),
        "BRIER_abs_diff": abs(block["BRIER"] - pub["BRIER"]),
        "ECE_abs_diff": abs(block["ECE"] - pub["ECE"]),
    }
    repro["GATE_PASS"] = bool(
        repro["OOS_N_match"] and repro["MAE_abs_diff"] < 1e-6
        and repro["RMSE_abs_diff"] < 1e-6 and repro["BRIER_abs_diff"] < 1e-6)

    return {
        "eligible_rows": int(len(eligible)),
        "train_n_final": int(len(eligible)),
        "metrics": block,
        "baselines_mae": baselines,
        "best_baseline": best,
        "reproduction": repro,
        "oos": oos,
        "folds": folds,
        "p_over": p_over,
    }


if __name__ == "__main__":
    r = run_v1()
    out = {k: v for k, v in r.items() if k not in ("oos", "folds", "p_over")}
    print(json.dumps(out, indent=2, default=str))
