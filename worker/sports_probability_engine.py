"""Price-blind NFL probability-engine foundation for SPORTS-NOVA.

This module models football outcomes only.  It deliberately accepts no market
features, performs expanding chronological evaluation, and keeps the four
timestamps/provenance fields required to audit every generated feature.

V1 is intentionally modest: regularized ridge point models plus empirical
residual distributions.  Distribution-family upgrades are expected to compete
against this baseline out of sample; they are not assumed to be improvements.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .continuous_validation import canonical_hash


ENGINE_ID = "SPORTS_NOVA_PROBABILITY_ENGINE_V1"
FEATURE_SCHEMA_VERSION = "sports_nova_nfl_features.v1"
JOINT_SCHEMA_VERSION = "sports_nova_nfl_joint_simulation.v1"

PLAYER_TARGET_COLUMNS = {
    "PASS_ATTEMPTS": ("attempts",),
    "COMPLETIONS": ("completions",),
    "PASS_YARDS": ("passing_yards",),
    "PASS_TD": ("passing_tds",),
    "RUSH_ATTEMPTS": ("carries",),
    "RUSH_YARDS": ("rushing_yards",),
    "TARGETS": ("targets",),
    "RECEPTIONS": ("receptions",),
    "RECEIVING_YARDS": ("receiving_yards",),
    # Player scoring TDs; passing TDs are a separate QB target.
    "TD": ("rushing_tds", "receiving_tds"),
}

POSITION_TARGETS = {
    "QB": ("PASS_ATTEMPTS", "COMPLETIONS", "PASS_YARDS", "PASS_TD", "RUSH_YARDS"),
    "RB": ("RUSH_ATTEMPTS", "RUSH_YARDS", "TARGETS", "RECEPTIONS", "RECEIVING_YARDS", "TD"),
    "WR": ("TARGETS", "RECEPTIONS", "RECEIVING_YARDS", "TD"),
    "TE": ("TARGETS", "RECEPTIONS", "RECEIVING_YARDS", "TD"),
}

TEAM_TARGETS = ("OFFENSIVE_PLAYS", "PASS_ATTEMPTS", "RUSH_ATTEMPTS", "POINTS")
TEAM_TARGET_COLUMNS = {
    "OFFENSIVE_PLAYS": ("offensive_plays",),
    "PASS_ATTEMPTS": ("pass_attempts",),
    "RUSH_ATTEMPTS": ("rush_attempts",),
    "POINTS": ("points",),
}
MODEL_COMPONENTS = ("PLAY_VOLUME", "PASS_RUN_DISTRIBUTION", "PLAYER_OPPORTUNITY", "PLAYER_EFFICIENCY")
CALIBRATION_METHODS = ("PLATT", "ISOTONIC")
NONNEGATIVE_TARGETS = {
    "PASS_ATTEMPTS", "COMPLETIONS", "PASS_TD", "RUSH_ATTEMPTS",
    "TARGETS", "RECEPTIONS", "TD", "OFFENSIVE_PLAYS", "POINTS",
}

_PROHIBITED_KEY_PARTS = {
    "sportsbook", "odds", "moneyline", "spread", "vig", "overround",
    "marketprice", "marketprobability", "impliedprobability", "bettingline",
    "closingline", "sportsbookline", "marketline",
}
_PROHIBITED_EXACT_KEYS = {"price", "line"}


def _key_token(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def assert_price_blind(payload: Any, *, path: str = "$", _key_context: bool = False) -> None:
    """Reject market-derived fields anywhere in a feature/config payload.

    Values are not keyword-scanned because ordinary prose may describe the
    boundary.  Keys are the executable contract and fail closed.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            token = _key_token(key)
            if token in _PROHIBITED_EXACT_KEYS or any(part in token for part in _PROHIBITED_KEY_PARTS):
                raise ValueError(f"PROHIBITED_MARKET_INPUT:{path}.{key}")
            assert_price_blind(value, path=f"{path}.{key}", _key_context=True)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_price_blind(value, path=f"{path}[{index}]", _key_context=_key_context)


def _utc(value: datetime | str, *, field: str) -> datetime:
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            value = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"INVALID_{field.upper()}") from exc
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field.upper()}_MUST_BE_TIMEZONE_AWARE")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def player_target_value(row: Mapping[str, Any], target: str) -> float:
    columns = PLAYER_TARGET_COLUMNS.get(target)
    if columns is None:
        raise ValueError(f"UNSUPPORTED_PLAYER_TARGET:{target}")
    missing = [column for column in columns if row.get(column) is None]
    if missing:
        raise ValueError(f"MISSING_TARGET_COLUMNS:{','.join(missing)}")
    return float(sum(float(row[column]) for column in columns))


def team_target_value(row: Mapping[str, Any], target: str) -> float:
    columns = TEAM_TARGET_COLUMNS.get(target)
    if columns is None:
        raise ValueError(f"UNSUPPORTED_TEAM_TARGET:{target}")
    missing = [column for column in columns if row.get(column) is None]
    if missing:
        raise ValueError(f"MISSING_TARGET_COLUMNS:{','.join(missing)}")
    return float(sum(float(row[column]) for column in columns))


def validate_position_target(position: str, target: str) -> None:
    if target not in POSITION_TARGETS.get(position, ()):
        raise ValueError(f"TARGET_NOT_SUPPORTED_FOR_POSITION:{position}:{target}")


@dataclass(frozen=True)
class FeatureCell:
    value: float | None
    source: tuple[str, ...]
    event_time: str | None
    available_at: str | None
    as_of_time: str
    support_n: int
    status: str = "AVAILABLE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": list(self.source),
            "event_time": self.event_time,
            "available_at": self.available_at,
            "as_of_time": self.as_of_time,
            "support_n": self.support_n,
            "status": self.status,
        }


def _feature_cell(
    history: Sequence[Mapping[str, Any]], *, target: str, as_of: datetime,
    value_getter=player_target_value,
) -> FeatureCell:
    if not history:
        return FeatureCell(None, (), None, None, _iso(as_of), 0, "NO_PRIOR_DATA")
    values = [value_getter(row, target) for row in history]
    event_times = [_utc(row["event_time"], field="event_time") for row in history]
    available_times = [_utc(row["available_at"], field="available_at") for row in history]
    return FeatureCell(
        float(np.mean(values)),
        tuple(sorted({str(row["source"]) for row in history})),
        _iso(max(event_times)),
        _iso(max(available_times)),
        _iso(as_of),
        len(values),
    )


def build_player_rolling_features(
    rows: Sequence[Mapping[str, Any]], *, target: str, trailing_window: int = 4,
) -> list[dict[str, Any]]:
    """Create leakage-safe prior-game features for one player target.

    Each input row is one completed player-game outcome with ``event_time``,
    ``available_at``, ``as_of_time`` (the pregame prediction time), ``source``,
    ``player_id``, ``game_id``, ``season``, and source target columns.  Historical
    rows enter a feature only when both their event and availability precede the
    current prediction as-of time.  The predicted game's outcome is never used.
    """
    if trailing_window < 1:
        raise ValueError("TRAILING_WINDOW_MUST_BE_POSITIVE")
    assert_price_blind(rows)
    prepared: list[dict[str, Any]] = []
    required = ("player_id", "game_id", "season", "source", "event_time", "available_at", "as_of_time")
    for index, original in enumerate(rows):
        row = dict(original)
        missing = [field for field in required if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"MISSING_TEMPORAL_OR_IDENTITY_FIELDS:{index}:{','.join(missing)}")
        event_time = _utc(row["event_time"], field="event_time")
        available_at = _utc(row["available_at"], field="available_at")
        as_of = _utc(row["as_of_time"], field="as_of_time")
        if as_of > event_time:
            raise ValueError(f"AS_OF_AFTER_EVENT:{index}")
        if available_at < event_time:
            raise ValueError(f"OUTCOME_AVAILABLE_BEFORE_EVENT:{index}")
        player_target_value(row, target)
        row.update(_event_dt=event_time, _available_dt=available_at, _as_of_dt=as_of)
        prepared.append(row)

    prepared.sort(key=lambda row: (row["_event_dt"], str(row["game_id"]), str(row["player_id"])))
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    for row in prepared:
        player_history = [
            prior for prior in histories[str(row["player_id"])]
            if prior["_event_dt"] < row["_event_dt"] and prior["_available_dt"] <= row["_as_of_dt"]
        ]
        season_history = [prior for prior in player_history if prior["season"] == row["season"]]
        features = {
            f"{target.lower()}_trailing_{trailing_window}": _feature_cell(
                player_history[-trailing_window:], target=target, as_of=row["_as_of_dt"]
            ).as_dict(),
            f"{target.lower()}_season_to_date": _feature_cell(
                season_history, target=target, as_of=row["_as_of_dt"]
            ).as_dict(),
            f"{target.lower()}_career_prior": _feature_cell(
                player_history, target=target, as_of=row["_as_of_dt"]
            ).as_dict(),
        }
        result.append({
            "engine_id": ENGINE_ID,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "player_id": str(row["player_id"]),
            "game_id": str(row["game_id"]),
            "season": row["season"],
            "position": row.get("position"),
            "target": target,
            "event_time": _iso(row["_event_dt"]),
            "as_of_time": _iso(row["_as_of_dt"]),
            "target_available_at": _iso(row["_available_dt"]),
            "observed": player_target_value(row, target),
            "features": features,
        })
        histories[str(row["player_id"])].append(row)
    return result


def build_team_rolling_features(
    rows: Sequence[Mapping[str, Any]], *, target: str, trailing_window: int = 4,
) -> list[dict[str, Any]]:
    """Create the same causal feature contract for team-volume/scoring targets."""
    if trailing_window < 1:
        raise ValueError("TRAILING_WINDOW_MUST_BE_POSITIVE")
    assert_price_blind(rows)
    prepared: list[dict[str, Any]] = []
    required = ("team_id", "game_id", "season", "source", "event_time", "available_at", "as_of_time")
    for index, original in enumerate(rows):
        row = dict(original)
        missing = [field for field in required if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"MISSING_TEMPORAL_OR_IDENTITY_FIELDS:{index}:{','.join(missing)}")
        event_time = _utc(row["event_time"], field="event_time")
        available_at = _utc(row["available_at"], field="available_at")
        as_of = _utc(row["as_of_time"], field="as_of_time")
        if as_of > event_time:
            raise ValueError(f"AS_OF_AFTER_EVENT:{index}")
        if available_at < event_time:
            raise ValueError(f"OUTCOME_AVAILABLE_BEFORE_EVENT:{index}")
        team_target_value(row, target)
        row.update(_event_dt=event_time, _available_dt=available_at, _as_of_dt=as_of)
        prepared.append(row)

    prepared.sort(key=lambda row: (row["_event_dt"], str(row["game_id"]), str(row["team_id"])))
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    for row in prepared:
        team_history = [
            prior for prior in histories[str(row["team_id"])]
            if prior["_event_dt"] < row["_event_dt"] and prior["_available_dt"] <= row["_as_of_dt"]
        ]
        season_history = [prior for prior in team_history if prior["season"] == row["season"]]
        features = {
            f"{target.lower()}_trailing_{trailing_window}": _feature_cell(
                team_history[-trailing_window:], target=target, as_of=row["_as_of_dt"], value_getter=team_target_value
            ).as_dict(),
            f"{target.lower()}_season_to_date": _feature_cell(
                season_history, target=target, as_of=row["_as_of_dt"], value_getter=team_target_value
            ).as_dict(),
            f"{target.lower()}_career_prior": _feature_cell(
                team_history, target=target, as_of=row["_as_of_dt"], value_getter=team_target_value
            ).as_dict(),
        }
        result.append({
            "engine_id": ENGINE_ID,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "team_id": str(row["team_id"]),
            # walk_forward_ridge accepts a generic entity id through player_id.
            "player_id": f"TEAM:{row['team_id']}",
            "game_id": str(row["game_id"]),
            "season": row["season"],
            "position": "TEAM",
            "target": target,
            "event_time": _iso(row["_event_dt"]),
            "as_of_time": _iso(row["_as_of_dt"]),
            "target_available_at": _iso(row["_available_dt"]),
            "observed": team_target_value(row, target),
            "features": features,
        })
        histories[str(row["team_id"])].append(row)
    return result


def validate_feature_row(row: Mapping[str, Any]) -> None:
    assert_price_blind(row)
    as_of = _utc(row["as_of_time"], field="as_of_time")
    event = _utc(row["event_time"], field="event_time")
    if as_of > event:
        raise ValueError("AS_OF_AFTER_EVENT")
    for name, cell in row.get("features", {}).items():
        required = {"source", "event_time", "available_at", "as_of_time", "status", "support_n", "value"}
        if not required.issubset(cell):
            raise ValueError(f"INCOMPLETE_FEATURE_PROVENANCE:{name}")
        if _utc(cell["as_of_time"], field="as_of_time") != as_of:
            raise ValueError(f"FEATURE_AS_OF_MISMATCH:{name}")
        if cell["status"] == "AVAILABLE":
            feature_event = _utc(cell["event_time"], field="event_time")
            feature_available = _utc(cell["available_at"], field="available_at")
            if feature_event >= event or feature_available > as_of:
                raise ValueError(f"TEMPORAL_FIREWALL_VIOLATION:{name}")


def _flat_features(row: Mapping[str, Any], feature_names: Sequence[str]) -> list[float] | None:
    values: list[float] = []
    for name in feature_names:
        cell = row.get("features", {}).get(name)
        if not cell or cell.get("value") is None:
            return None
        values.append(float(cell["value"]))
    return values


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float | int]:
    if not observed or len(observed) != len(predicted):
        raise ValueError("METRICS_REQUIRE_EQUAL_NONEMPTY_SERIES")
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {"N": int(y.size), "MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}


def calibration_table(probabilities: Sequence[float], outcomes: Sequence[int], *, bins: int = 10) -> list[dict[str, Any]]:
    if bins < 2 or len(probabilities) != len(outcomes):
        raise ValueError("INVALID_CALIBRATION_INPUT")
    table: list[dict[str, Any]] = []
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities) if lo <= p <= hi and (index == bins - 1 or p < hi)]
        table.append({
            "bin": index, "lower": lo, "upper": hi, "N": len(members),
            "mean_probability": float(np.mean([probabilities[i] for i in members])) if members else None,
            "observed_rate": float(np.mean([outcomes[i] for i in members])) if members else None,
        })
    return table


def probability_metrics(probabilities: Sequence[float], outcomes: Sequence[int], *, bins: int = 10) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("PROBABILITY_METRICS_REQUIRE_EQUAL_NONEMPTY_SERIES")
    eps = 1e-12
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    y = np.asarray(outcomes, dtype=float)
    table = calibration_table(p.tolist(), y.astype(int).tolist(), bins=bins)
    ece = sum(row["N"] / len(p) * abs(row["mean_probability"] - row["observed_rate"]) for row in table if row["N"])
    return {
        "N": len(p),
        "BRIER": float(np.mean((p - y) ** 2)),
        "LOGLOSS": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "ECE": float(ece),
        "CALIBRATION_TABLE": table,
    }


def distribution_from_samples(
    samples: Sequence[float], *, preserve_samples: bool = True,
    lower_bound: float | None = None,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("DISTRIBUTION_REQUIRES_FINITE_SAMPLES")
    if lower_bound is not None:
        values = np.maximum(values, lower_bound)
    minimum_half = 0.5 if float(values.min()) >= 0 else math.floor(float(values.min())) - 0.5
    maximum_half = math.ceil(float(values.max())) + 0.5
    thresholds = np.arange(minimum_half, maximum_half + 0.001, 1.0)
    curve = [
        {
            "threshold": float(threshold),
            "P_OVER": float(np.mean(values > threshold)),
            "P_AT_OR_BELOW": float(np.mean(values <= threshold)),
        }
        for threshold in thresholds
    ]
    quantiles = np.percentile(values, [10, 25, 50, 75, 90])
    result = {
        "EXPECTED_VALUE": float(np.mean(values)),
        "MEDIAN": float(np.median(values)),
        "STD": float(np.std(values, ddof=0)),
        "P10": float(quantiles[0]), "P25": float(quantiles[1]),
        "P50": float(quantiles[2]), "P75": float(quantiles[3]), "P90": float(quantiles[4]),
        "DENSE_PROBABILITY_CURVE": curve,
        "SAMPLE_N": int(values.size),
    }
    if preserve_samples:
        result["RAW_DISTRIBUTION_SAMPLES"] = values.tolist()
    return result


def empirical_residual_distribution(
    expected_value: float, residuals: Sequence[float], *, preserve_samples: bool = True,
    lower_bound: float | None = None,
) -> dict[str, Any]:
    if not residuals:
        raise ValueError("EMPIRICAL_DISTRIBUTION_REQUIRES_TRAINING_RESIDUALS")
    return distribution_from_samples(
        [expected_value + float(value) for value in residuals],
        preserve_samples=preserve_samples, lower_bound=lower_bound,
    )


def fit_calibrator(
    method: str, probabilities: Sequence[float], outcomes: Sequence[int], *,
    available_at: Sequence[datetime | str], training_cutoff: datetime | str,
) -> dict[str, Any]:
    """Fit Platt/isotonic using observations available by the training cutoff."""
    method = method.upper()
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"UNSUPPORTED_CALIBRATION_METHOD:{method}")
    if not probabilities or not (len(probabilities) == len(outcomes) == len(available_at)):
        raise ValueError("INVALID_CALIBRATION_INPUT")
    cutoff = _utc(training_cutoff, field="training_cutoff")
    if any(_utc(value, field="available_at") > cutoff for value in available_at):
        raise ValueError("CALIBRATION_FUTURE_OUTCOME_LEAKAGE")
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(outcomes, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("CALIBRATION_REQUIRES_BOTH_OUTCOMES")
    if method == "PLATT":
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(logits, y)
        parameters = {"coefficient": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}
    else:
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        parameters = {
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
    return {
        "METHOD": method,
        "TRAINING_CUTOFF": _iso(cutoff),
        "TRAINING_N": len(p),
        "PARAMETERS": parameters,
    }


def apply_calibrator(calibrator: Mapping[str, Any], probabilities: Sequence[float]) -> list[float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    method = calibrator["METHOD"]
    parameters = calibrator["PARAMETERS"]
    if method == "PLATT":
        z = parameters["intercept"] + parameters["coefficient"] * np.log(p / (1.0 - p))
        return (1.0 / (1.0 + np.exp(-z))).tolist()
    if method == "ISOTONIC":
        return np.interp(p, parameters["x_thresholds"], parameters["y_thresholds"]).tolist()
    raise ValueError(f"UNSUPPORTED_CALIBRATION_METHOD:{method}")


def walk_forward_ridge(
    rows: Sequence[Mapping[str, Any]], *, feature_names: Sequence[str],
    thresholds: Sequence[float] = (), min_train_rows: int = 30, alpha: float = 1.0,
) -> dict[str, Any]:
    """Expanding-window Ridge evaluation with event-batch isolation.

    Outcomes can train a fold only when their ``target_available_at`` is no later
    than the earliest as-of time in the evaluation batch.  Rows sharing an event
    time are predicted together before any of their outcomes can enter training.
    """
    if not feature_names:
        raise ValueError("FEATURE_NAMES_REQUIRED")
    for row in rows:
        validate_feature_row(row)
    ordered = sorted(rows, key=lambda row: (_utc(row["event_time"], field="event_time"), str(row["game_id"]), str(row["player_id"])))
    event_groups: dict[datetime, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        event_groups[_utc(row["event_time"], field="event_time")].append(row)

    predictions: list[dict[str, Any]] = []
    for event_time in sorted(event_groups):
        batch = event_groups[event_time]
        batch_as_of = min(_utc(row["as_of_time"], field="as_of_time") for row in batch)
        training = [
            row for row in ordered
            if _utc(row["event_time"], field="event_time") < event_time
            and _utc(row["target_available_at"], field="target_available_at") <= batch_as_of
            and _flat_features(row, feature_names) is not None
        ]
        evaluable = [row for row in batch if _flat_features(row, feature_names) is not None]
        if len(training) < min_train_rows or not evaluable:
            continue
        x_train = np.asarray([_flat_features(row, feature_names) for row in training], dtype=float)
        y_train = np.asarray([row["observed"] for row in training], dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_train, y_train)
        fitted = model.predict(x_train)
        residuals = y_train - fitted
        for row in evaluable:
            target = str(row.get("target"))
            expected = float(model.predict([_flat_features(row, feature_names)])[0])
            samples = expected + residuals
            if target in NONNEGATIVE_TARGETS:
                expected = max(0.0, expected)
                samples = np.maximum(expected + residuals, 0.0)
            baseline_predictions = {
                name: float(row["features"][name]["value"]) for name in feature_names
                if row["features"][name]["value"] is not None
            }
            predictions.append({
                "game_id": row["game_id"], "player_id": row["player_id"],
                "event_time": row["event_time"], "as_of_time": row["as_of_time"],
                "observed": float(row["observed"]), "predicted": expected,
                "baselines": baseline_predictions,
                "P_OVER": {str(float(t)): float(np.mean(samples > float(t))) for t in thresholds},
                "training_n": len(training),
            })
    if not predictions:
        return {"STATUS": "INSUFFICIENT_DATA", "N": 0, "PREDICTIONS": []}

    observed = [row["observed"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    baseline_metrics: dict[str, Any] = {}
    for name in feature_names:
        paired = [(row["observed"], row["baselines"].get(name)) for row in predictions]
        paired = [(y, p) for y, p in paired if p is not None]
        baseline_metrics[name] = regression_metrics([x[0] for x in paired], [x[1] for x in paired]) if paired else {"N": 0}
    probability_by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        key = str(float(threshold))
        probability_by_threshold[key] = probability_metrics(
            [row["P_OVER"][key] for row in predictions],
            [int(row["observed"] > threshold) for row in predictions],
        )
    advanced = regression_metrics(observed, predicted)
    comparable = [metrics for metrics in baseline_metrics.values() if metrics.get("N") == advanced["N"]]
    beats_all = bool(comparable) and all(advanced["RMSE"] < metrics["RMSE"] for metrics in comparable)
    return {
        "STATUS": "COMPLETE", "MODEL": "REGULARIZED_RIDGE_EMPIRICAL_RESIDUAL_V1",
        "ALPHA": alpha, "FEATURES": list(feature_names), "METRICS": advanced,
        "BASELINES": baseline_metrics, "BEATS_ALL_SIMPLE_BASELINES_OOS": beats_all,
        "PROBABILITY_METRICS_BY_THRESHOLD": probability_by_threshold,
        "PREDICTIONS": predictions,
    }


def build_joint_simulation_schema(*, game_id: str, home_team_id: str, away_team_id: str) -> dict[str, Any]:
    """Return a simulator-ready dependency schema without asserting independence."""
    return {
        "SCHEMA": JOINT_SCHEMA_VERSION,
        "GAME_ID": game_id,
        "TEAM_IDS": {"HOME": home_team_id, "AWAY": away_team_id},
        "MODEL_COMPONENTS": list(MODEL_COMPONENTS),
        "LATENT_GAME_FACTORS": ["PACE", "PASS_RATE", "GAME_SCRIPT", "SCORING_ENVIRONMENT"],
        "DEPENDENCIES": [
            ["TEAM_PLAYS", "PLAYER_OPPORTUNITY"],
            ["QB_PASS_ATTEMPTS", "GAME_SCRIPT"],
            ["QB_PASS_YARDS", "RECEIVER_YARDS"],
            ["RECEIVER_TARGET_SHARES", "TEAM_TARGETS"],
            ["RB_CARRIES", "TEAM_LEAD_STATE"],
            ["RB_CARRIES", "QB_PASS_ATTEMPTS"],
        ],
        "COHERENCE_CONSTRAINTS": [
            "SUM_PLAYER_TARGETS_EQUALS_TEAM_TARGETS",
            "SUM_PLAYER_CARRIES_EQUALS_TEAM_RUSH_ATTEMPTS",
            "TEAM_PASS_ATTEMPTS_PLUS_TEAM_RUSH_ATTEMPTS_IS_BOUNDED_BY_TEAM_PLAYS",
            "RECEIVING_YARDS_ARE_CONDITIONED_ON_TEAM_PASSING_YARDS",
        ],
        "INDEPENDENCE_ASSUMED": False,
        "SIMULATION_COUNT_TARGET": 10000,
        "V1_CORRELATION_STATUS": "SCHEMA_READY_NOT_ESTIMATED",
    }


def freeze_model_artifact(
    *, model_id: str, version: str, target: str, training_cutoff: datetime | str,
    feature_schema: Mapping[str, Any], data_provenance: Mapping[str, Any],
    model_parameters: Mapping[str, Any], validation_metrics: Mapping[str, Any],
    calibration_metrics: Mapping[str, Any], created_at: datetime | str | None = None,
) -> dict[str, Any]:
    assert_price_blind({
        "feature_schema": feature_schema,
        "model_parameters": model_parameters,
        "data_provenance": data_provenance,
    })
    payload = {
        "MODEL_ID": model_id,
        "VERSION": version,
        "TARGET": target,
        "TRAINING_CUTOFF": _iso(_utc(training_cutoff, field="training_cutoff")),
        "FEATURE_SCHEMA": dict(feature_schema),
        "DATA_PROVENANCE": dict(data_provenance),
        "MODEL_PARAMETERS": dict(model_parameters),
        "VALIDATION_METRICS": dict(validation_metrics),
        "CALIBRATION_METRICS": dict(calibration_metrics),
        "CREATED_AT": _iso(_utc(created_at or datetime.now(timezone.utc), field="created_at")),
        "RESEARCH_ONLY": True,
        "OPERATIONAL_APPROVAL": "NONE",
    }
    payload["SHA256"] = canonical_hash(payload)
    return payload


def verify_frozen_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "MODEL_ID", "VERSION", "TARGET", "TRAINING_CUTOFF", "FEATURE_SCHEMA",
        "DATA_PROVENANCE", "MODEL_PARAMETERS", "VALIDATION_METRICS",
        "CALIBRATION_METRICS", "CREATED_AT", "SHA256",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return {"STATUS": "INVALID", "MISSING": missing}
    clone = dict(artifact)
    expected = clone.pop("SHA256")
    actual = canonical_hash(clone)
    return {"STATUS": "VALID" if expected == actual else "CORRUPT", "EXPECTED": expected, "ACTUAL": actual}
