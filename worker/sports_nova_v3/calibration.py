"""Chronology-gated probability calibration."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
import hashlib
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

@dataclass(frozen=True)
class CalibrationArtifact:
    method: str
    cutoff: datetime
    n_fit: int
    status: str
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    coef: float | None = None
    intercept: float | None = None
    sha256: str = ""

def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def fit_component_calibration(probabilities: Iterable[float], outcomes: Iterable[int],
                              available_at: Iterable[datetime], *, cutoff: datetime,
                              method: str = "isotonic", min_n: int = 400) -> CalibrationArtifact:
    p, y, times = np.asarray(list(probabilities), float), np.asarray(list(outcomes), int), list(available_at)
    if not (len(p) == len(y) == len(times)):
        raise ValueError("calibration arrays must align")
    keep = np.array([t <= cutoff for t in times], dtype=bool)
    p, y = p[keep], y[keep]
    if len(p) < min_n or len(np.unique(y)) < 2:
        payload = {"method": method, "cutoff": cutoff.isoformat(), "n_fit": int(len(p)), "status": "INSUFFICIENT_SUPPORT"}
        return CalibrationArtifact(method, cutoff, len(p), "INSUFFICIENT_SUPPORT", sha256=_hash(payload))
    p = np.clip(p, 0, 1)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        x = tuple(float(v) for v in model.X_thresholds_)
        yy = tuple(float(v) for v in model.y_thresholds_)
        payload = {"method": method, "cutoff": cutoff.isoformat(), "n_fit": len(p), "x": x, "y": yy}
        return CalibrationArtifact(method, cutoff, len(p), "FIT", x, yy, sha256=_hash(payload))
    if method == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(p.reshape(-1, 1), y)
        coef, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
        payload = {"method": method, "cutoff": cutoff.isoformat(), "n_fit": len(p), "coef": coef, "intercept": intercept}
        return CalibrationArtifact(method, cutoff, len(p), "FIT", coef=coef, intercept=intercept, sha256=_hash(payload))
    raise ValueError(method)

def apply_frozen_calibration(artifact: CalibrationArtifact, probabilities: Iterable[float]) -> np.ndarray:
    p = np.clip(np.asarray(list(probabilities), float), 0, 1)
    if artifact.status != "FIT":
        return p
    if artifact.method == "isotonic":
        return np.clip(np.interp(p, np.asarray(artifact.x), np.asarray(artifact.y)), 0, 1)
    logits = artifact.intercept + artifact.coef * p
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

def validate_sample_coherence(raw_samples, calibrated_samples) -> dict[str, object]:
    """Calibration may change probabilities only; it cannot alter path identities."""
    if getattr(raw_samples, "n_sims", None) != getattr(calibrated_samples, "n_sims", None):
        return {"status": "FAIL", "reason": "path count changed"}
    if getattr(raw_samples, "state_hash", None) != getattr(calibrated_samples, "state_hash", None):
        return {"status": "FAIL", "reason": "state hash changed"}
    return {"status": "PASS", "note": "calibration retains aligned simulation state"}
