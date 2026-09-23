"""Scoring functions. Definitions are pinned to V1's so improvement
percentages compare like with like."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.clip(p, EPS, 1 - EPS) - np.asarray(y)) ** 2))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """V1's definition: equal-width bins on [0,1], count-weighted gap."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    frame = pd.DataFrame({"p": p, "y": y})
    frame["bin"] = pd.cut(frame.p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    total = 0.0
    for _, grp in frame.groupby("bin", observed=True):
        if len(grp) == 0:
            continue
        total += len(grp) / len(frame) * abs(grp.p.mean() - grp.y.mean())
    return float(total)


def score_block(y: np.ndarray, pred: np.ndarray, p_over: np.ndarray,
                threshold: float) -> dict:
    y_bin = (np.asarray(y) > threshold).astype(int)
    return {
        "N": int(len(y)),
        "MAE": mae(y, pred),
        "RMSE": rmse(y, pred),
        "BRIER": brier(y_bin, p_over),
        "LOGLOSS": logloss(y_bin, p_over),
        "ECE": ece(y_bin, p_over),
    }


def quantile_coverage(y: np.ndarray, qpreds: dict[float, np.ndarray]) -> list[dict]:
    out = []
    for q, vals in sorted(qpreds.items()):
        cov = float(np.mean(np.asarray(y) <= np.asarray(vals)))
        out.append({"nominal_quantile": q, "empirical_coverage": round(cov, 4),
                    "gap": round(cov - q, 4)})
    return out


def pinball_loss(y: np.ndarray, qpreds: dict[float, np.ndarray]) -> float:
    y = np.asarray(y, dtype=float)
    losses = []
    for q, vals in qpreds.items():
        d = y - np.asarray(vals, dtype=float)
        losses.append(np.mean(np.maximum(q * d, (q - 1) * d)))
    return float(np.mean(losses))


def crps_from_samples(y: np.ndarray, samples: np.ndarray) -> float:
    """CRPS by the sample-based identity, E|X-y| - 0.5*E|X-X'|.

    `samples` is (n_obs, n_draws). The second term uses the sorted-sample
    closed form so cost stays O(n log n) per row rather than O(n^2).
    """
    y = np.asarray(y, dtype=float)
    s = np.sort(np.asarray(samples, dtype=float), axis=1)
    n = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    w = (2 * np.arange(1, n + 1) - n - 1)
    term2 = (2.0 / (n * n)) * (s * w).sum(axis=1)
    return float(np.mean(term1 - 0.5 * term2))
