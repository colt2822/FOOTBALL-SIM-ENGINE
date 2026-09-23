"""Chronological walk-forward metrics and frozen ablation bookkeeping."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable
import math
import numpy as np

REQUIRED_METRICS = ("MAE", "RMSE", "Brier", "LogLoss", "ECE", "CRPS",
                    "interval_coverage", "joint_brier", "correlation_error")

def _arr(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float)

def ece(probabilities, outcomes, bins: int = 10) -> float:
    p, y = _arr(probabilities), _arr(outcomes)
    if len(p) == 0:
        return float("nan")
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= lo) & ((p <= hi) if hi == 1 else (p < hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total

def crps_from_samples(observed, samples) -> float:
    y = _arr(observed)
    s = np.sort(np.asarray(samples, dtype=float), axis=1)
    if len(y) != s.shape[0] or s.shape[1] == 0:
        raise ValueError("CRPS sample shape mismatch")
    n = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    weights = 2 * np.arange(1, n + 1) - n - 1
    term2 = (2.0 / (n * n)) * (s * weights).sum(axis=1)
    return float(np.mean(term1 - .5 * term2))

def metric_report(observed, predicted, *, probabilities=None, binary_outcomes=None,
                  samples=None, intervals: dict[float, tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, float | int | None]:
    y, pred = _arr(observed), _arr(predicted)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("observations and predictions must be nonempty and aligned")
    out: dict[str, float | int | None] = {
        "N": int(len(y)), "MAE": float(np.mean(np.abs(y - pred))),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
    }
    if probabilities is not None and binary_outcomes is not None:
        p, b = np.clip(_arr(probabilities), 1e-12, 1 - 1e-12), _arr(binary_outcomes)
        if len(p) != len(b):
            raise ValueError("probability arrays must align")
        out.update({"Brier": float(np.mean((p - b) ** 2)),
                    "LogLoss": float(-np.mean(b * np.log(p) + (1 - b) * np.log(1 - p))),
                    "ECE": ece(p, b)})
    else:
        out.update({"Brier": None, "LogLoss": None, "ECE": None})
    out["CRPS"] = crps_from_samples(y, samples) if samples is not None else None
    if intervals:
        out["interval_coverage"] = {str(level): float(np.mean((y >= lo) & (y <= hi)))
                                     for level, (lo, hi) in intervals.items()}
    else:
        out["interval_coverage"] = None
    return out

@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    cutoff: Any
    cutoff_ok: bool

def chronological_folds(rows: Iterable[dict[str, Any]], *, time_key: str = "kickoff",
                        game_key: str = "game_id", min_train: int = 1) -> list[Fold]:
    data = sorted(list(rows), key=lambda r: r[time_key])
    games = []
    for row in data:
        if row.get(game_key) not in games:
            games.append(row.get(game_key))
    folds = []
    for i in range(min_train, len(games)):
        train_games, test_game = set(games[:i]), games[i]
        train = tuple(j for j, r in enumerate(data) if r.get(game_key) in train_games)
        test = tuple(j for j, r in enumerate(data) if r.get(game_key) == test_game)
        train_times = [data[j][time_key] for j in train]
        test_times = [data[j][time_key] for j in test]
        ok = bool(train and test and max(train_times) < min(test_times))
        folds.append(Fold(i, train, test, max(train_times) if train else None, ok))
    return folds

def validate_walk_forward(dataset, frozen_protocol=None) -> dict[str, Any]:
    rows = list(dataset or [])
    if not rows:
        return {"status": "NO_DATA", "folds": [], "metrics": {}, "required_metrics": REQUIRED_METRICS}
    folds = chronological_folds(rows)
    if any(not f.cutoff_ok for f in folds):
        return {"status": "FAIL", "reason": "temporal fold overlap", "folds": len(folds)}
    reports = []
    for fold in folds:
        test = [rows[i] for i in fold.test_indices]
        if not all("actual" in r and "predicted" in r for r in test):
            continue
        reports.append(metric_report([r["actual"] for r in test], [r["predicted"] for r in test],
            probabilities=[r["probability"] for r in test] if all("probability" in r for r in test) else None,
            binary_outcomes=[r["binary_outcome"] for r in test] if all("binary_outcome" in r for r in test) else None))
    return {"status": "PASS" if reports else "NO_DATA", "folds": len(folds),
            "temporal_firewall": "PASS", "fold_reports": reports,
            "required_metrics": REQUIRED_METRICS,
            "data_status": "EMPIRICAL_ONLY_IF_SOURCE_AVAILABILITY_VERIFIED"}

def compare_v3_v2(v3: dict[str, float], v2: dict[str, float]) -> dict[str, Any]:
    keys = [k for k in ("MAE", "RMSE", "Brier", "LogLoss", "ECE", "CRPS") if k in v3 and k in v2]
    return {key: {"v3": v3[key], "v2": v2[key], "delta_v3_minus_v2": v3[key] - v2[key]} for key in keys}

def run_dev_ablations(*, runner: Callable[[str], dict[str, Any]] | None = None) -> dict[str, Any]:
    names = ("NO_SCRIPT", "NO_MATCHUP", "NO_RECENT_FORM", "NO_SHRINKAGE", "NO_PERSONNEL",
             "PERMUTED_DEPENDENCE", "NO_EPISTEMIC")
    if runner is None:
        return {name: {"status": "NOT_RUN", "reason": "requires frozen causal evaluation rows"} for name in names}
    return {name: runner(name) for name in names}
