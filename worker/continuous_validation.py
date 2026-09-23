"""Continuous empirical validation primitives for NODE 3.

This module is deliberately research-only.  It composes with the existing
``worker.research`` execution engine and never submits orders or promotes a
model.  Functions fail closed when forward evidence is absent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sqlite3
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence


class ReplaySpeed(str, Enum):
    STEP = "STEP"
    MAX = "MAX"
    X1 = "1x"
    X2 = "2x"
    X10 = "10x"
    X25 = "25x"
    X50 = "50x"
    X100 = "100x"


class ReplaySource(str, Enum):
    LIVE = "LIVE"
    REPLAY = "REPLAY"
    FIXTURE = "FIXTURE"


class DatasetClass(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class ReplayStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


CALIBRATION_FAMILIES = (
    "MODEL_CALIBRATION", "FILL_CALIBRATION", "SLIPPAGE_CALIBRATION",
    "LATENCY_CALIBRATION", "ADVERSE_SELECTION_CALIBRATION",
    "CAPACITY_CALIBRATION", "EXECUTION_QUALITY_CALIBRATION",
    "EDGE_SURVIVAL_CALIBRATION",
)
DIVERGENCE_TYPES = (
    "EXPECTED", "DATA_DIFFERENCE", "VERSION_DIFFERENCE", "CONFIG_DIFFERENCE",
    "TIMING_DIFFERENCE", "MODEL_DIFFERENCE", "FEATURE_DIFFERENCE",
    "EXECUTION_MODEL_DIFFERENCE", "BUG_SUSPECTED",
)
DRIFT_STATES = ("STABLE", "WATCH", "DEGRADED", "FAILED", "INSUFFICIENT_DATA")
AUTO_UPDATE_POLICIES = (
    "SAFE_FOR_PERIODIC_RECALIBRATION", "REQUIRES_SHADOW_APPROVAL",
    "REQUIRES_FULL_REVALIDATION", "NEVER_AUTO_UPDATE",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def environment_fingerprint(*, feature_schema: str = "unknown", config_version: str = "unknown") -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        numpy_version = np.__version__
    except ImportError:
        numpy_version = "unavailable"
    try:
        import pandas as pd  # type: ignore
        pandas_version = pd.__version__
    except ImportError:
        pandas_version = "unavailable"
    return {
        "python": sys.version.split()[0], "numpy": numpy_version,
        "pandas": pandas_version, "platform": platform.platform(),
        "cpu_architecture": platform.machine(), "timezone": time.tzname,
        "locale": __import__("locale").getlocale(),
        "feature_schema": feature_schema, "config_version": config_version,
    }


@dataclass(frozen=True)
class ReplayPacket:
    payload: Mapping[str, Any]
    source_mode: str = ReplaySource.REPLAY.value
    replay_session_id: str = ""
    original_timestamp: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.payload)
        out.update(source_mode=self.source_mode, replay_session_id=self.replay_session_id)
        if self.original_timestamp is not None:
            out.setdefault("original_timestamp", self.original_timestamp)
        return out


def replay_packets(rows: Iterable[Mapping[str, Any]], *, replay_session_id: str, source: ReplaySource = ReplaySource.REPLAY) -> list[ReplayPacket]:
    """Attach provenance while retaining source timestamps and stable ordering."""
    indexed = list(enumerate(rows))
    def key(item: tuple[int, Mapping[str, Any]]) -> tuple[float, int]:
        raw = item[1].get("timestamp", item[1].get("ts", item[1].get("event_time", 0)))
        try: value = float(raw)
        except (TypeError, ValueError): value = 0.0
        return value, item[0]
    return [ReplayPacket(dict(item[1]), source.value, replay_session_id, key(item)[0]) for item in sorted(indexed, key=key)]


def validate_speed(speed: str | ReplaySpeed) -> str:
    value = speed.value if isinstance(speed, ReplaySpeed) else str(speed).upper() if str(speed).upper() in {"STEP", "MAX"} else str(speed)
    if value not in {s.value for s in ReplaySpeed}:
        raise ValueError(f"unsupported replay speed: {speed}")
    return value


def validate_dataset_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Classify NODE 2 metadata; absence is UNKNOWN, never silently complete."""
    if not manifest:
        return {"classification": DatasetClass.UNKNOWN.value, "reasons": ["manifest_missing"]}
    reasons: list[str] = []
    coverage = manifest.get("coverage", {})
    required = ("rule", "fee", "external_feed", "sequence", "timestamp")
    missing = [key for key in required if coverage.get(key) in {False, "missing", "unknown"}]
    if manifest.get("known_gaps") or missing:
        reasons.extend(["known_gaps"] if manifest.get("known_gaps") else [])
        reasons.extend(f"missing_{key}_coverage" for key in missing)
    if manifest.get("sequence_integrity") is False: reasons.append("sequence_integrity")
    if manifest.get("timestamp_integrity") is False: reasons.append("timestamp_integrity")
    if manifest.get("unsafe") is True: classification = DatasetClass.UNSAFE.value
    elif manifest.get("complete") is True and not reasons: classification = DatasetClass.COMPLETE.value
    elif reasons: classification = DatasetClass.PARTIAL.value
    else: classification = DatasetClass.UNKNOWN.value
    return {"classification": classification, "reasons": reasons, "coverage": dict(coverage)}


_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_sessions (
 replay_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, start_time REAL, end_time REAL,
 speed TEXT NOT NULL, strategies_json TEXT NOT NULL, models_json TEXT NOT NULL,
 config_version TEXT NOT NULL, software_versions_json TEXT NOT NULL,
 feature_schemas_json TEXT NOT NULL, participants_json TEXT NOT NULL,
 status TEXT NOT NULL, started_at REAL, completed_at REAL, replay_fingerprint TEXT
);"""


class ReplaySessionRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path); self.db.row_factory = sqlite3.Row; self.db.execute(_SESSION_SCHEMA); self.db.commit()

    def create(self, *, dataset_id: str, start_time: float | None, end_time: float | None, speed: str | ReplaySpeed, strategies: Sequence[str], model_versions: Mapping[str, str], config_version: str, software_versions: Mapping[str, str], feature_schemas: Mapping[str, str], participants: Sequence[str]) -> dict[str, Any]:
        replay_id = str(uuid.uuid4()); speed_value = validate_speed(speed); now = time.time()
        self.db.execute("INSERT INTO replay_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (replay_id, dataset_id, start_time, end_time, speed_value, json.dumps(list(strategies), sort_keys=True), json.dumps(dict(model_versions), sort_keys=True), config_version, json.dumps(dict(software_versions), sort_keys=True), json.dumps(dict(feature_schemas), sort_keys=True), json.dumps(list(participants), sort_keys=True), ReplayStatus.QUEUED.value, None, None, None)); self.db.commit()
        return self.get(replay_id)

    def transition(self, replay_id: str, status: ReplayStatus | str, *, fingerprint: str | None = None) -> dict[str, Any]:
        try: status = status if isinstance(status, ReplayStatus) else ReplayStatus(str(status))
        except ValueError as exc: raise ValueError("invalid replay status") from exc
        started = time.time() if status == ReplayStatus.RUNNING else None
        completed = time.time() if status in {ReplayStatus.COMPLETE, ReplayStatus.FAILED, ReplayStatus.CANCELLED} else None
        self.db.execute("UPDATE replay_sessions SET status=?, started_at=COALESCE(started_at, ?), completed_at=COALESCE(?, completed_at), replay_fingerprint=COALESCE(?, replay_fingerprint) WHERE replay_id=?", (status.value, started, completed, fingerprint, replay_id)); self.db.commit()
        return self.get(replay_id)

    def get(self, replay_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM replay_sessions WHERE replay_id=?", (replay_id,)).fetchone()
        if row is None: raise KeyError(replay_id)
        out = dict(row)
        for key in ("strategies_json", "models_json", "software_versions_json", "feature_schemas_json", "participants_json"): out[key.removesuffix("_json")] = json.loads(out.pop(key))
        return out

    def close(self) -> None:
        self.db.close()


def deterministic_replay_fingerprint(*, dataset: Any, software_version: str, config: Any, model: Any, strategy: Any, seed: int | None, output: Any) -> str:
    return canonical_hash({"dataset": dataset, "software_version": software_version, "config": config, "model": model, "strategy": strategy, "seed": seed, "output": output})


def compare_live_replay(live: Mapping[str, Any], replay: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fields = ("market_state", "features", "candidate", "nova_decision")
    differences = []
    for field_name in fields:
        if live.get(field_name) != replay.get(field_name):
            differences.append({"field": field_name, "type": _divergence_type(field_name, metadata or {})})
    return {"equal": not differences, "divergences": differences, "status": "MATCH" if not differences else "DIVERGED"}


def _divergence_type(field_name: str, metadata: Mapping[str, Any]) -> str:
    if metadata.get("data_version_different"): return "DATA_DIFFERENCE"
    if metadata.get("software_version_different"): return "VERSION_DIFFERENCE"
    if metadata.get("config_different"): return "CONFIG_DIFFERENCE"
    if metadata.get("model_different"): return "MODEL_DIFFERENCE"
    if metadata.get("timing_different"): return "TIMING_DIFFERENCE"
    return {"features": "FEATURE_DIFFERENCE", "candidate": "EXECUTION_MODEL_DIFFERENCE"}.get(field_name, "BUG_SUSPECTED")


def _insufficient(family: str, strategy: str, *, reason: str, version: str = "") -> dict[str, Any]:
    return {"calibration_type": family, "strategy": strategy, "status": "INSUFFICIENT_DATA", "statistical_evidence": False, "reason": reason, "current_version": version, "candidate_version": ""}


def _binary_metrics(predicted: Sequence[float], observed: Sequence[int]) -> dict[str, float]:
    if not predicted or len(predicted) != len(observed): return {}
    eps = 1e-12; p = [min(1 - eps, max(eps, float(x))) for x in predicted]; y = [int(x) for x in observed]
    brier = mean((a - b) ** 2 for a, b in zip(p, y)); logloss = -mean(b * math.log(a) + (1 - b) * math.log(1 - a) for a, b in zip(p, y))
    avg_p, avg_y = mean(p), mean(y); var = mean((x - avg_p) ** 2 for x in p)
    slope = sum((a - avg_p) * (b - avg_y) for a, b in zip(p, y)) / sum((a - avg_p) ** 2 for a in p) if var else 0.0
    intercept = avg_y - slope * avg_p
    bins = [[] for _ in range(10)]
    for a, b in zip(p, y): bins[min(9, int(a * 10))].append((a, b))
    ece = sum(len(bucket) / len(p) * abs(mean(a for a, _ in bucket) - mean(b for _, b in bucket)) for bucket in bins if bucket)
    return {"brier": brier, "log_loss": logloss, "ece": ece, "calibration_slope": slope, "calibration_intercept": intercept}


# Public re-export: worker.model_training reuses this rather than duplicating the metric math.
binary_classification_metrics = _binary_metrics


def calibrate_probability(rows: Sequence[Mapping[str, Any]], *, strategy: str, predicted_key: str = "predicted_probability", outcome_key: str = "observed_outcome", forward: bool = True, min_samples: int = 30, version: str = "current") -> dict[str, Any]:
    if not forward: return _insufficient("MODEL_CALIBRATION", strategy, reason="forward evidence required", version=version)
    usable = [(float(r[predicted_key]), int(r[outcome_key])) for r in rows if r.get(predicted_key) is not None and r.get(outcome_key) is not None]
    if len(usable) < min_samples: return _insufficient("MODEL_CALIBRATION", strategy, reason=f"need {min_samples} independent observations; got {len(usable)}", version=version)
    metrics = _binary_metrics([p for p, _ in usable], [y for _, y in usable]); return {"calibration_type": "MODEL_CALIBRATION", "strategy": strategy, "status": "COMPLETE", "sample_size": len(usable), "metrics": metrics, "current_version": version, "candidate_version": f"model_calibration_{version}", "independent_events": len({r.get("event_id", i) for i, r in enumerate(rows) if r.get(predicted_key) is not None and r.get(outcome_key) is not None})}


def calibrate_fill(rows: Sequence[Mapping[str, Any]], *, strategy: str, forward: bool = True, min_samples: int = 30) -> dict[str, Any]:
    result = calibrate_probability([{"predicted_probability": r.get("predicted_fill_probability"), "observed_outcome": r.get("observed_fill")} for r in rows], strategy=strategy, forward=forward, min_samples=min_samples)
    result["calibration_type"] = "FILL_CALIBRATION"
    return result


def calibrate_regression(rows: Sequence[Mapping[str, Any]], *, family: str, strategy: str, predicted_key: str, observed_key: str, forward: bool = True, min_samples: int = 30) -> dict[str, Any]:
    if family not in CALIBRATION_FAMILIES: raise ValueError(f"unknown calibration family: {family}")
    if not forward: return _insufficient(family, strategy, reason="forward evidence required")
    values = [(float(r[predicted_key]), float(r[observed_key])) for r in rows if r.get(predicted_key) is not None and r.get(observed_key) is not None and math.isfinite(float(r[predicted_key])) and math.isfinite(float(r[observed_key]))]
    if len(values) < min_samples: return _insufficient(family, strategy, reason=f"need {min_samples} observations; got {len(values)}")
    errors = [b - a for a, b in values]; return {"calibration_type": family, "strategy": strategy, "status": "COMPLETE", "sample_size": len(values), "metrics": {"mae": mean(abs(x) for x in errors), "bias": mean(errors), "rmse": math.sqrt(mean(x*x for x in errors)), "observed_mean": mean(b for _, b in values), "predicted_mean": mean(a for a, _ in values)}, "candidate_version": f"{family.lower()}_{canonical_hash(values)[:8]}"}


def _ks_statistic(ref: list[float], cur: list[float]) -> float:
    pooled = sorted(ref + cur)
    return max(abs(sum(x <= point for x in ref) / len(ref) - sum(x <= point for x in cur) / len(cur)) for point in pooled)


def population_stability_index(reference: Sequence[float], current: Sequence[float], *, buckets: int = 10) -> float:
    """Standard PSI: quantile-bucket the reference sample, compare bucket share drift."""
    ref = sorted(float(x) for x in reference)
    if not ref or not current:
        return float("inf")
    edges = sorted({ref[min(len(ref) - 1, round(q * (len(ref) - 1)))] for q in (i / buckets for i in range(buckets + 1))})
    if len(edges) < 2:
        return 0.0
    def shares(values: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) - 1)
        for v in values:
            idx = min(len(counts) - 1, max(0, next((i for i in range(len(counts)) if v <= edges[i + 1]), len(counts) - 1)))
            counts[idx] += 1
        eps = 1e-6
        return [max(eps, c / len(values)) for c in counts]
    ref_shares, cur_shares = shares(ref), shares(list(current))
    return sum((c - r) * math.log(c / r) for r, c in zip(ref_shares, cur_shares))


def wasserstein_distance_1d(reference: Sequence[float], current: Sequence[float]) -> float:
    """1-D earth-mover distance via the sorted-sample formula (no scipy dependency)."""
    ref, cur = sorted(float(x) for x in reference), sorted(float(x) for x in current)
    if not ref or not cur:
        return float("inf")
    pooled = sorted(set(ref + cur))
    total = 0.0
    for a, b in zip(pooled, pooled[1:]):
        f_ref = sum(x <= a for x in ref) / len(ref)
        f_cur = sum(x <= a for x in cur) / len(cur)
        total += abs(f_ref - f_cur) * (b - a)
    return total


DRIFT_METHODS = ("ks", "psi", "wasserstein")


def drift_status(reference: Sequence[float], current: Sequence[float], *, method: str = "ks", watch: float = 0.1, degraded: float = 0.25, minimum: int = 30) -> dict[str, Any]:
    if len(reference) < minimum or len(current) < minimum: return {"state": "INSUFFICIENT_DATA", "sample_size": len(current), "method": method}
    if method not in DRIFT_METHODS: raise ValueError(f"unknown drift method: {method}")
    ref, cur = sorted(map(float, reference)), sorted(map(float, current))
    if method == "ks":
        statistic = _ks_statistic(ref, cur)
    elif method == "psi":
        statistic, watch, degraded = population_stability_index(ref, cur), 0.1, 0.25
    else:
        spread = max(1e-9, mean(abs(x) for x in ref) or 1.0)
        statistic, watch, degraded = wasserstein_distance_1d(ref, cur) / spread, 0.1, 0.25
    state = "FAILED" if statistic >= degraded * 2 else "DEGRADED" if statistic >= degraded else "WATCH" if statistic >= watch else "STABLE"
    return {"state": state, "method": method, "statistic": statistic, "ks": statistic if method == "ks" else _ks_statistic(ref, cur), "reference_n": len(ref), "current_n": len(cur), "reference_mean": mean(ref), "current_mean": mean(cur)}


def drift_report(reference: Sequence[float], current: Sequence[float], *, minimum: int = 30) -> dict[str, Any]:
    """Run every supported drift method and report the worst (most-degraded) state."""
    rank = {"INSUFFICIENT_DATA": -1, "STABLE": 0, "WATCH": 1, "DEGRADED": 2, "FAILED": 3}
    by_method = {method: drift_status(reference, current, method=method, minimum=minimum) for method in DRIFT_METHODS}
    worst = max(by_method.values(), key=lambda r: rank[r["state"]])
    return {"overall_state": worst["state"], "by_method": by_method}


def event_counts(rows: Sequence[Mapping[str, Any]], *, event_key: str = "event_id") -> dict[str, int]:
    return {"rows": len(rows), "signals": sum(1 for r in rows if r.get("signal", r.get("candidate", False))), "opportunities": len({r.get("opportunity_episode_id", r.get(event_key, i)) for i, r in enumerate(rows)}), "independent_events": len({r.get(event_key, r.get("ticker", i)) for i, r in enumerate(rows)})}


def signal_funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = ("raw_opportunity", "candidate", "nova_validated", "risk_approved", "paper_order", "paper_fill", "outcome")
    return {"counts": {stage: sum(1 for r in rows if bool(r.get(stage))) for stage in stages}, "dropoff_reasons": {str(r.get("rejection_reason")): sum(1 for x in rows if x.get("rejection_reason") == r.get("rejection_reason")) for r in rows if r.get("rejection_reason")}}


def shadow_compare(rows: Sequence[Mapping[str, Any]], *, champion_prefix: str = "champion_", challenger_prefix: str = "challenger_") -> dict[str, Any]:
    if not rows: return {"status": "INSUFFICIENT_DATA", "sample_size": 0}
    fields = ("signal", "action", "fair_value", "net_edge", "latency", "capacity", "execution_quality")
    agreement = {f: mean(1.0 if r.get(champion_prefix + f) == r.get(challenger_prefix + f) else 0.0 for r in rows) for f in fields}
    return {"status": "COMPLETE", "sample_size": len(rows), "agreement": agreement, "counterfactual": {"champion_net_edge": mean(float(r.get(champion_prefix + "net_edge", 0)) for r in rows), "challenger_net_edge": mean(float(r.get(challenger_prefix + "net_edge", 0)) for r in rows)}}


def alpha_leakage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = ("source_data_delay", "node2_processing", "node2_to_4_network", "node4_queue", "node4_computation", "node4_to_1_network", "nova_queue", "nova_processing", "paper_execution_assumptions")
    totals = {stage: sum(float(r.get(stage, 0) or 0) for r in rows) for stage in stages}; ranked = sorted(totals, key=totals.get, reverse=True)
    by_strategy = {}
    for strategy in sorted({r.get("strategy") for r in rows if r.get("strategy")}):
        subset = [r for r in rows if r.get("strategy") == strategy]
        by_strategy[strategy] = {stage: sum(float(r.get(stage, 0) or 0) for r in subset) for stage in stages}
    return {"status": "COMPLETE" if rows else "INSUFFICIENT_DATA", "rows": len(rows), "edge_lost": totals, "ranked_components": ranked, "by_strategy": by_strategy}


def build_calibration_evidence(calibration: Mapping[str, Any], *, training_period: str, validation_period: str, old_error: float | None = None, stress_tests: Mapping[str, Any] | None = None, recommended_state: str = "MORE_DATA_REQUIRED") -> dict[str, Any]:
    packet = {"schema": "kalshi_calibration_evidence.v1", **dict(calibration), "training_period": training_period, "validation_period": validation_period, "old_error": old_error, "oos_comparison": calibration.get("metrics", {}), "stress_tests": dict(stress_tests or {}), "recommended_state": recommended_state}
    packet["artifact_checksum"] = canonical_hash(packet); return packet


def build_strategy_evidence(*, strategy: str, version: str, rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any], recommendation: str = "MORE_DATA_REQUIRED", **sections: Any) -> dict[str, Any]:
    packet = {"schema": "kalshi_strategy_evidence.v2", "strategy": strategy, "version": version, "generated_at": datetime.now(timezone.utc).isoformat(), "sample_counts": event_counts(rows), "metrics": dict(metrics), "recommendation": recommendation, **sections}
    packet["artifact_checksum"] = canonical_hash(packet); return packet


class ImmutableArtifactRegistry:
    def __init__(self, root: str | Path): self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def freeze(self, name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        checksum = canonical_hash(payload); path = self.root / f"{name}_{checksum[:12]}.json"
        content = json.dumps(dict(payload), sort_keys=True, indent=2, default=str)
        if path.exists():
            if path.read_text(encoding="utf-8") != content: raise ValueError("immutable artifact collision")
        else:
            # write-to-temp then atomic replace: a crash mid-write must never leave a
            # partially-written file at the final path looking like a valid artifact.
            tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        return {"name": name, "path": str(path), "checksum": checksum, "immutable": True}

    def verify(self, artifact: Mapping[str, Any]) -> bool:
        path = Path(str(artifact["path"])); payload = json.loads(path.read_text(encoding="utf-8")); return canonical_hash(payload) == artifact["checksum"]


ARTIFACT_VERIFICATION_STATES = ("VALID", "MISSING", "CORRUPT", "DEPENDENCY_INCOMPATIBLE", "FEATURE_SCHEMA_INCOMPATIBLE")


def verify_artifact_file(path: str | Path, *, checksum_field: str = "artifact_checksum", expected_feature_schema_version: str | None = None, expected_dependency_fingerprint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Routine capable of verifying any previously frozen JSON artifact on disk (spec section 27)."""
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING", "path": str(p)}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "CORRUPT", "path": str(p)}
    if not isinstance(payload, Mapping) or checksum_field not in payload:
        return {"status": "CORRUPT", "path": str(p)}
    clone = dict(payload); checksum = clone.pop(checksum_field)
    if canonical_hash(clone) != checksum:
        return {"status": "CORRUPT", "path": str(p)}
    if expected_feature_schema_version is not None and payload.get("feature_schema_version") not in (None, expected_feature_schema_version):
        return {"status": "FEATURE_SCHEMA_INCOMPATIBLE", "path": str(p)}
    if expected_dependency_fingerprint is not None:
        fingerprint = payload.get("dependency_fingerprint") or {}
        mismatched = {k: (fingerprint.get(k), v) for k, v in expected_dependency_fingerprint.items() if k in fingerprint and fingerprint.get(k) != v}
        if mismatched:
            return {"status": "DEPENDENCY_INCOMPATIBLE", "path": str(p), "mismatched": mismatched}
    return {"status": "VALID", "path": str(p)}


@dataclass(order=True)
class ResearchJob:
    sort_key: tuple[int, float] = field(init=False, repr=False)
    priority: int
    submitted_at: float = field(default_factory=time.time)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = "research"
    strategy: str = ""
    dataset: str = ""
    status: str = "QUEUED"
    cpu_limit: float | None = None
    ram_limit_mb: int | None = None
    started: float | None = None
    completed: float | None = None
    output_artifacts: list[str] = field(default_factory=list)
    def __post_init__(self) -> None: self.sort_key = (-int(self.priority), self.submitted_at)


class ResearchJobQueue:
    def __init__(self): self._jobs: list[ResearchJob] = []
    def submit(self, job: ResearchJob) -> str: self._jobs.append(job); self._jobs.sort(key=lambda x: x.sort_key); return job.job_id
    def claim(self) -> ResearchJob | None:
        for job in self._jobs:
            if job.status == "QUEUED": job.status = "RUNNING"; job.started = time.time(); return job
        return None
    def complete(self, job_id: str, artifacts: Sequence[str] = ()) -> ResearchJob:
        job = next(j for j in self._jobs if j.job_id == job_id); job.status = "COMPLETE"; job.completed = time.time(); job.output_artifacts = list(artifacts); return job
    def snapshot(self) -> dict[str, Any]: return {"depth": sum(j.status == "QUEUED" for j in self._jobs), "active": sum(j.status == "RUNNING" for j in self._jobs), "jobs": [asdict(j) for j in self._jobs]}


def no_live_trading_policy(payload: Mapping[str, Any]) -> None:
    forbidden = {"live_order", "order_submission", "broker_credentials", "broker_api_key", "promote", "production_deploy"}
    found = sorted(k for k in payload if str(k).lower() in forbidden)
    if found: raise ValueError("NODE 3 research payload contains forbidden operational authority: " + ", ".join(found))


def run_replay_session(job: Any, *, manifest: Mapping[str, Any] | None = None, registry: ReplaySessionRegistry | None = None) -> dict[str, Any]:
    """Run the existing replay executor with a validated research envelope."""
    from .research import run_replay_research
    validation = validate_dataset_manifest(manifest or job.parameters.get("dataset_manifest"))
    if validation["classification"] == DatasetClass.UNSAFE.value:
        raise ValueError("dataset manifest is UNSAFE")
    no_live_trading_policy(job.parameters)
    result = run_replay_research(job)
    result.result["dataset_validation"] = validation
    result.result["research_only"] = True
    result.result["can_promote"] = False
    if validation["classification"] != DatasetClass.COMPLETE.value:
        result.warnings.append("dataset is not COMPLETE; empirical conclusions are limited")
    return {"result": result.result, "metrics": result.metrics, "warnings": result.warnings}


def research_schedule() -> dict[str, str]:
    return {
        "hourly": "latency/freshness summary",
        "daily": "fill/slippage/adverse-selection calibration checks",
        "nightly": "replay consistency, drift scan, execution studies",
        "weekly": "torture, model comparison, capacity, strategy health report",
    }


RESEARCH_JOB_CLASSES = ("CRITICAL_RESEARCH", "NORMAL", "BATCH", "BACKGROUND")


SCHEDULE_JOB_STATUSES = ("IDLE", "RUNNING", "DISABLED")

_SCHEDULER_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_schedule (
    name TEXT PRIMARY KEY, interval_seconds REAL NOT NULL, job_class TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'IDLE',
    last_run_at REAL, next_run_at REAL NOT NULL, last_success_at REAL, last_failure_at REAL,
    failure_count INTEGER NOT NULL DEFAULT 0, last_error TEXT
);"""


class ResearchScheduler:
    """Interval scheduler, persisted to SQLite so cadences survive a process restart.

    Callers call run_pending() from their own loop/heartbeat tick; this never spawns
    threads or a process of its own, per the no-overengineering policy. Pass a real file
    path for durable operation; the default ":memory:" is for ephemeral/test use. Pass
    `clock` to inject a fake wall clock in tests instead of depending on real time.
    """

    def __init__(self, path: str | Path = ":memory:", *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        self.db = sqlite3.connect(path if path == ":memory:" else str(Path(path)))
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute(_SCHEDULER_SCHEMA)
        self.db.commit()

    def register(self, name: str, interval_seconds: float, *, job_class: str = "NORMAL", start_immediately: bool = True, enabled: bool = True) -> dict[str, Any]:
        if job_class not in RESEARCH_JOB_CLASSES:
            raise ValueError(f"unknown job class: {job_class}")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        now = self._clock()
        next_run = now if start_immediately else now + interval_seconds
        self.db.execute(
            "INSERT INTO research_schedule (name, interval_seconds, job_class, enabled, status, next_run_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET interval_seconds=excluded.interval_seconds, job_class=excluded.job_class, enabled=excluded.enabled",
            (name, float(interval_seconds), job_class, int(enabled), "IDLE", next_run),
        )
        self.db.commit()
        return self.get(name)

    def get(self, name: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM research_schedule WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        return dict(row)

    def set_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        self.db.execute("UPDATE research_schedule SET enabled=? WHERE name=?", (int(enabled), name))
        self.db.commit()
        return self.get(name)

    def due(self, *, now: float | None = None) -> list[str]:
        """Jobs whose next_run_at has arrived, including ones currently RUNNING: run_pending
        is responsible for deciding SKIP_ALREADY_RUNNING vs. proceeding for those."""
        now = now if now is not None else self._clock()
        rows = self.db.execute("SELECT name FROM research_schedule WHERE enabled=1 AND next_run_at<=?", (now,)).fetchall()
        return sorted(r["name"] for r in rows)

    def run_pending(self, runners: Mapping[str, Callable[[], Any]], *, now: float | None = None, allow_overlap: bool = False) -> dict[str, Any]:
        now = now if now is not None else self._clock()
        results: dict[str, Any] = {}
        for name in self.due(now=now):
            row = self.get(name)
            if row["status"] == "RUNNING" and not allow_overlap:
                results[name] = {"status": "SKIP_ALREADY_RUNNING"}
                continue
            self.db.execute("UPDATE research_schedule SET status='RUNNING' WHERE name=?", (name,))
            self.db.commit()
            runner = runners.get(name)
            try:
                output = runner() if runner else None
                self.db.execute(
                    "UPDATE research_schedule SET status='IDLE', last_run_at=?, next_run_at=?, last_success_at=?, last_error=NULL WHERE name=?",
                    (now, now + row["interval_seconds"], now, name),
                )
                self.db.commit()
                results[name] = {"status": "RAN", "output": output}
            except Exception:  # a failing research job must not take down the scheduler loop
                error = traceback.format_exc()
                self.db.execute(
                    "UPDATE research_schedule SET status='IDLE', last_run_at=?, next_run_at=?, last_failure_at=?, failure_count=failure_count+1, last_error=? WHERE name=?",
                    (now, now + row["interval_seconds"], now, error, name),
                )
                self.db.commit()
                results[name] = {"status": "FAILED", "error": error}
        return results

    def snapshot(self) -> dict[str, Any]:
        return {r["name"]: dict(r) for r in self.db.execute("SELECT * FROM research_schedule ORDER BY name").fetchall()}

    def close(self) -> None:
        self.db.close()


def default_research_scheduler(path: str | Path = ":memory:", *, clock: Callable[[], float] = time.time) -> ResearchScheduler:
    """Cadences matching research_schedule()'s hourly/daily/nightly/weekly intent, plus the
    finer-grained default schedule from spec section 19. Idempotent: re-registering on
    restart against the same persisted path does not reset last_run/failure history."""
    scheduler = ResearchScheduler(path, clock=clock)
    scheduler.register("latency_freshness_summary", 3600, job_class="NORMAL")
    scheduler.register("health_data_completeness_check", 300, job_class="CRITICAL_RESEARCH")
    scheduler.register("calibration_checks", 86400, job_class="NORMAL")
    scheduler.register("forward_paper_ingestion", 86400, job_class="NORMAL")
    scheduler.register("replay_consistency_and_drift_scan", 86400, job_class="BATCH")
    scheduler.register("cross_node_consistency_check", 86400, job_class="BATCH")
    scheduler.register("execution_studies_nightly", 86400, job_class="BATCH")
    scheduler.register("torture_and_strategy_health_report", 604800, job_class="BATCH")
    scheduler.register("champion_challenger_review", 604800, job_class="BATCH")
    scheduler.register("capacity_study", 604800, job_class="BACKGROUND")
    return scheduler


NODE_CONSISTENCY_PARTICIPANTS = ("node1", "node2", "node3", "node4")


def node_consistency_fixture() -> dict[str, Any]:
    """Deterministic shared inputs/expected outputs for the cross-node math consistency test.

    Every participating node is expected to compute the same YES/NO complement, fee, VWAP,
    capacity, net-edge, and hash values from this fixture; see node_consistency_check().
    """
    from .quant_research import capacity_curve, execution_edges

    row = {"yes_bid": 0.42, "yes_ask": 0.45, "asks": [(0.45, 10.0), (0.46, 20.0)], "bids": [(0.42, 15.0), (0.41, 25.0)]}
    fair_value, fee_per_contract, yes_price = 0.50, 0.02, 0.63
    edges = execution_edges(fair_value=fair_value, row=row, side="buy", quantity=10, fee_per_contract=fee_per_contract)
    capacity = capacity_curve(fair_value=fair_value, row=row, sizes=(1.0, 10.0, 25.0), side="buy", fee_per_contract=fee_per_contract)
    return {
        "row": row, "fair_value": fair_value, "fee_per_contract": fee_per_contract,
        "expected": {
            "yes_no_complement": {"yes": yes_price, "no": round(1.0 - yes_price, 10)},
            "vwap": edges["depth"]["vwap"],
            "net_executable_edge": edges["net_executable_edge"],
            "capacity_curve": capacity,
            "state_hash": canonical_hash(row),
            "rule_hash": canonical_hash({"rule_version": "r1", "fee_version": "f1"}),
            "feature_serialization": canonical_hash({"fair_value": fair_value, "row": row}),
            "candidate_serialization": canonical_hash({"strategy": "EDGE-001", "fair_value": fair_value, "net_edge": edges["net_executable_edge"]}),
        },
    }


def node_consistency_check(node_outputs: Mapping[str, Mapping[str, Any]], *, fixture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare each node's reported field values against the shared fixture and each other.

    node_outputs is {node_name: {field_name: value}}; a node may report a subset of fields.
    """
    fixture = fixture or node_consistency_fixture()
    expected = fixture["expected"]
    fields = sorted(expected)
    per_node: dict[str, Any] = {}
    for node, output in node_outputs.items():
        mismatched = [f for f in fields if f in output and canonical_hash(output[f]) != canonical_hash(expected[f])]
        per_node[node] = {"fields_checked": [f for f in fields if f in output], "matches_fixture": not mismatched, "mismatched_fields": mismatched}
    nodes = sorted(node_outputs)
    cross_node_mismatches = []
    for f in fields:
        by_node = {node: canonical_hash(node_outputs[node][f]) for node in nodes if f in node_outputs[node]}
        if len(set(by_node.values())) > 1:
            cross_node_mismatches.append({"field": f, "by_node_hash": by_node})
    consistent = bool(per_node) and all(v["matches_fixture"] for v in per_node.values()) and not cross_node_mismatches
    return {"status": "CONSISTENT" if consistent else "INCONSISTENT", "fixture_fields": fields, "per_node": per_node, "cross_node_mismatches": cross_node_mismatches}


RESEARCH_JOB_TYPES = ("REPLAY", "CALIBRATION", "TRAINING", "TORTURE", "DRIFT", "CONSISTENCY", "SHADOW_EVAL")
RESEARCH_JOB_STATUSES = ("QUEUED", "RUNNING", "COMPLETE", "FAILED", "INTERRUPTED", "CANCELLED")

_RESEARCH_JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_jobs (
    job_id TEXT PRIMARY KEY, job_type TEXT NOT NULL, strategy TEXT, dataset TEXT,
    priority INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, submitted_at REAL NOT NULL,
    started_at REAL, completed_at REAL, output_artifacts_json TEXT NOT NULL DEFAULT '[]',
    error TEXT
);"""


class ResearchJobRegistry:
    """Durable registry for REPLAY/CALIBRATION/TRAINING/TORTURE/DRIFT/CONSISTENCY/SHADOW_EVAL jobs.

    On construction, any job left RUNNING by a prior process (a crash or unclean restart)
    is marked INTERRUPTED rather than silently treated as complete (spec section 25).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(_RESEARCH_JOB_SCHEMA)
        self.db.commit()
        self.recovered_job_ids = self._recover_interrupted()

    def _recover_interrupted(self) -> list[str]:
        rows = self.db.execute("SELECT job_id FROM research_jobs WHERE status='RUNNING'").fetchall()
        ids = [r["job_id"] for r in rows]
        if ids:
            self.db.executemany("UPDATE research_jobs SET status='INTERRUPTED' WHERE job_id=?", [(i,) for i in ids])
            self.db.commit()
        return ids

    def submit(self, *, job_type: str, strategy: str = "", dataset: str = "", priority: int = 0, job_id: str | None = None) -> dict[str, Any]:
        if job_type not in RESEARCH_JOB_TYPES:
            raise ValueError(f"unknown research job type: {job_type}")
        job_id = job_id or str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO research_jobs (job_id, job_type, strategy, dataset, priority, status, submitted_at) VALUES (?,?,?,?,?,?,?)",
            (job_id, job_type, strategy, dataset, priority, "QUEUED", time.time()),
        )
        self.db.commit()
        return self.get(job_id)

    def claim(self) -> dict[str, Any] | None:
        row = self.db.execute("SELECT job_id FROM research_jobs WHERE status='QUEUED' ORDER BY priority DESC, submitted_at ASC LIMIT 1").fetchone()
        if row is None:
            return None
        self.db.execute("UPDATE research_jobs SET status='RUNNING', started_at=? WHERE job_id=?", (time.time(), row["job_id"]))
        self.db.commit()
        return self.get(row["job_id"])

    def complete(self, job_id: str, *, artifacts: Sequence[str] = ()) -> dict[str, Any]:
        self.db.execute("UPDATE research_jobs SET status='COMPLETE', completed_at=?, output_artifacts_json=? WHERE job_id=?", (time.time(), json.dumps(list(artifacts)), job_id))
        self.db.commit()
        return self.get(job_id)

    def fail(self, job_id: str, *, error: str) -> dict[str, Any]:
        self.db.execute("UPDATE research_jobs SET status='FAILED', completed_at=?, error=? WHERE job_id=?", (time.time(), error, job_id))
        self.db.commit()
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM research_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        out = dict(row)
        out["output_artifacts"] = json.loads(out.pop("output_artifacts_json"))
        return out

    def snapshot(self) -> dict[str, Any]:
        rows = [dict(r) for r in self.db.execute("SELECT * FROM research_jobs").fetchall()]
        for r in rows:
            r["output_artifacts"] = json.loads(r.pop("output_artifacts_json"))
        return {"depth": sum(r["status"] == "QUEUED" for r in rows), "active": sum(r["status"] == "RUNNING" for r in rows), "interrupted": sum(r["status"] == "INTERRUPTED" for r in rows), "jobs": rows}

    def close(self) -> None:
        self.db.close()


# --------------------------------------------------------------------------- #
# Cross-node consistency protocol (spec sections 20-24): machine-readable
# request/result envelopes, file/fixture ingestion, and comparison/evidence.
# --------------------------------------------------------------------------- #

NODE_CONSISTENCY_EXTERNAL_PARTICIPANTS = ("node1", "node2", "node4")
CONSISTENCY_RESULT_CLASSES = (
    "CONSISTENT", "VALUE_MISMATCH", "HASH_MISMATCH", "VERSION_MISMATCH",
    "CONFIG_MISMATCH", "DEPENDENCY_MISMATCH", "MISSING_NODE", "ERROR",
)


def build_node_consistency_request(*, fixture_version: str = "v1", config_version: str = "unknown", requested_by: str = "node3") -> dict[str, Any]:
    fixture = node_consistency_fixture()
    return {
        "schema": "node3_consistency_request.v1",
        "test_run_id": str(uuid.uuid4()),
        "fixture_version": fixture_version,
        "config_version": config_version,
        "requested_by": requested_by,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
        "tests": sorted(fixture["expected"]),
    }


def build_node_consistency_result(*, test_run_id: str, node_id: str, software_version: str, config_version: str, dependency_fingerprint: Mapping[str, Any], result: Mapping[str, Any], error: str | None = None) -> dict[str, Any]:
    """One node's machine-readable report of what it computed for a given test_run_id."""
    payload = {
        "schema": "node3_consistency_result.v1",
        "test_run_id": test_run_id,
        "node_id": node_id,
        "software_version": software_version,
        "config_version": config_version,
        "dependency_fingerprint": dict(dependency_fingerprint),
        "result": dict(result),
        "error": error,
        "timestamp": time.time(),
    }
    payload["result_hash"] = canonical_hash(result)
    return payload


def load_consistency_results_from_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """File/fixture ingestion path for integration testing before live node transport exists."""
    out: list[dict[str, Any]] = []
    for p in paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        out.extend(payload if isinstance(payload, list) else [payload])
    return out


def compare_consistency_results(request: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare node-reported results against the fixture's expected values.

    NODE 3's own local fixture execution is never sufficient to claim cluster
    consistency: if node1/node2/node4 have not actually reported, status is
    INTEGRATION_WAITING, never CONSISTENT (spec section 24).
    """
    reference_env = environment_fingerprint()
    by_node = {r["node_id"]: r for r in results if r.get("test_run_id") == request["test_run_id"]}
    missing = sorted(set(NODE_CONSISTENCY_EXTERNAL_PARTICIPANTS) - set(by_node))
    classifications: dict[str, str] = {node: "MISSING_NODE" for node in missing}
    mismatches: list[dict[str, Any]] = []
    for node in sorted(set(NODE_CONSISTENCY_EXTERNAL_PARTICIPANTS) & set(by_node)):
        r = by_node[node]
        if r.get("error"):
            classifications[node] = "ERROR"
            mismatches.append({"node": node, "kind": "ERROR", "detail": r["error"]})
            continue
        node_mismatches = []
        for test_name in request["tests"]:
            expected_hash = canonical_hash(request["fixture"]["expected"].get(test_name))
            reported = (r.get("result") or {}).get(test_name)
            reported_hash = canonical_hash(reported) if test_name in (r.get("result") or {}) else None
            if reported_hash == expected_hash:
                continue
            if r.get("config_version") not in (None, request.get("config_version")):
                kind = "CONFIG_MISMATCH"
            elif dict(r.get("dependency_fingerprint") or {}) != reference_env:
                kind = "DEPENDENCY_MISMATCH"
            elif r.get("software_version") not in (None, reference_env.get("config_version")):
                kind = "HASH_MISMATCH" if reported_hash is not None else "VALUE_MISMATCH"
            else:
                kind = "HASH_MISMATCH" if reported_hash is not None else "VALUE_MISMATCH"
            entry = {"node": node, "test": test_name, "kind": kind, "expected_hash": expected_hash, "reported_hash": reported_hash}
            node_mismatches.append(entry)
            mismatches.append(entry)
        classifications[node] = "CONSISTENT" if not node_mismatches else node_mismatches[0]["kind"]
    if missing:
        status = "INTEGRATION_WAITING"
    elif mismatches:
        status = "INCONSISTENT"
    else:
        status = "CONSISTENT"
    return {"status": status, "test_run_id": request["test_run_id"], "nodes_tested": sorted(by_node), "nodes_missing": missing, "classifications": classifications, "mismatches": mismatches}


def build_consistency_evidence(comparison: Mapping[str, Any], *, request: Mapping[str, Any]) -> dict[str, Any]:
    """Persistable, NOVA-consumable report of a cross-node consistency run (spec section 23)."""
    total_checks = len(comparison["nodes_tested"]) * len(request["tests"])
    failed = list(comparison["mismatches"])
    packet = {
        "schema": "node3_consistency_evidence.v1",
        "test_run_id": comparison["test_run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": comparison["status"],
        "nodes_tested": comparison["nodes_tested"],
        "nodes_missing": comparison["nodes_missing"],
        "checks_total": total_checks,
        "checks_failed": len(failed),
        "checks_passed": max(0, total_checks - len(failed)),
        "version_differences": [m for m in failed if m.get("kind") in {"VERSION_MISMATCH", "HASH_MISMATCH"}],
        "config_differences": [m for m in failed if m.get("kind") == "CONFIG_MISMATCH"],
        "dependency_differences": [m for m in failed if m.get("kind") == "DEPENDENCY_MISMATCH"],
        "exact_mismatches": failed,
        "live_orders_allowed": False,
    }
    packet["artifact_checksum"] = canonical_hash(packet)
    return packet


def node3_health(*, software_version: str = "unknown", config_version: str = "unknown", queue: ResearchJobQueue | None = None, last_replay: str | None = None, last_calibration: str | None = None, last_training_job: str | None = None, last_frozen_artifact: str | None = None, last_consistency_run: str | None = None, last_scheduler_run: Mapping[str, Any] | None = None, scheduler: ResearchScheduler | None = None) -> dict[str, Any]:
    usage = {"cpu_percent": None, "ram_percent": None, "disk_free_bytes": None}
    try:
        import shutil
        usage["disk_free_bytes"] = shutil.disk_usage(Path.cwd()).free
    except OSError:
        pass
    scheduler_snapshot = scheduler.snapshot() if scheduler else {}
    scheduler_health = "CONFIGURED" if scheduler_snapshot else "NOT_CONFIGURED"
    if scheduler_snapshot and any(row.get("status") == "RUNNING" for row in scheduler_snapshot.values()):
        scheduler_health = "BUSY"
    return {
        "node": "node3", "state": "ONLINE", "software_version": software_version, "config_version": config_version,
        "execution_authority": "NONE", "live_trading": False, "research_db": "configured",
        "scheduler": "configured", "job_queue": (queue.snapshot() if queue else {"depth": 0, "active": 0}),
        "last_successful_replay": last_replay, "last_calibration_run": last_calibration,
        "resource_loan_mode": "DISABLED",
        "training_service_health": "CONFIGURED",
        "scheduler_persistence_health": scheduler_health,
        "scheduler_persistence_detail": scheduler_snapshot,
        "consistency_test_status": "CONFIGURED" if last_consistency_run else "NEVER_RUN",
        "artifact_verification_status": "CONFIGURED",
        "last_training_job": last_training_job,
        "last_successful_frozen_artifact": last_frozen_artifact,
        "last_cross_node_consistency_run": last_consistency_run,
        "last_scheduler_execution": dict(last_scheduler_run or {}),
        **usage,
    }


def auto_update_policy(parameter: str) -> str:
    periodic = {"fill_calibration", "spread_distribution", "slippage_estimate", "latency_estimate"}
    shadow = {"execution_quality_score", "capacity_model", "adverse_selection_model"}
    full = {"fair_value_model", "strategy_threshold", "logical_assumption"}
    if parameter in periodic: return "SAFE_FOR_PERIODIC_RECALIBRATION"
    if parameter in shadow: return "REQUIRES_SHADOW_APPROVAL"
    if parameter in full: return "REQUIRES_FULL_REVALIDATION"
    return "NEVER_AUTO_UPDATE"


def validate_historical_versions(row: Mapping[str, Any]) -> dict[str, Any]:
    """Guard against applying current rules/fees to an historical decision."""
    violations = []
    if row.get("future_rule_version") is True or row.get("rule_version_effective_at") is not None and row.get("rule_version_effective_at") > row.get("decision_timestamp", float("inf")):
        violations.append("future_rule_version")
    if row.get("future_fee_version") is True or row.get("fee_version_effective_at") is not None and row.get("fee_version_effective_at") > row.get("decision_timestamp", float("inf")):
        violations.append("future_fee_version")
    return {"valid": not violations, "violations": violations}


def timestamp_integrity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Require economic occurrence and receipt timestamps to remain distinct."""
    occurrence = row.get("economic_occurrence_time", row.get("event_time"))
    receive = row.get("receive_time", row.get("timestamp"))
    violations = []
    if occurrence is None or receive is None: violations.append("missing_occurrence_or_receive_time")
    else:
        try:
            if float(receive) < float(occurrence): violations.append("receive_before_occurrence")
        except (TypeError, ValueError): violations.append("invalid_timestamp")
    return {"valid": not violations, "violations": violations, "economic_occurrence_time": occurrence, "receive_time": receive, "administrative_close_time": row.get("administrative_close_time")}


def reject_random_split(method: str) -> None:
    if str(method).lower() in {"random", "random_split", "shuffle"}:
        raise ValueError("random snapshot splits are forbidden for time-series research")


def forward_sample_ledger(periods: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mark forward periods as contaminated once they influence development."""
    ledger = []
    for period in periods:
        item = dict(period); item["untouched"] = not bool(item.get("used_for_model_development") or item.get("used_for_threshold_development")); ledger.append(item)
    return {"periods": ledger, "untouched_periods": sum(bool(p["untouched"]) for p in ledger), "contaminated_periods": sum(not bool(p["untouched"]) for p in ledger)}
