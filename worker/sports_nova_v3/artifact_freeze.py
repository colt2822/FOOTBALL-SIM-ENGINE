"""Content-addressed V3 model, simulation and replay artifacts."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import gzip
import numpy as np

def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")

def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def freeze_simulation(samples, model, runtime, *, output_path: str | Path | None = None) -> dict[str, Any]:
    """Create a stable manifest; optional path writing is explicit and local."""
    sample_hash = content_sha256(samples)
    model_hash = content_sha256(model)
    runtime_hash = content_sha256(runtime)
    manifest = {
        "schema_version": "sports_nova_v3_simulation_artifact.1",
        "game_id": samples.game_id,
        "n_sims": samples.n_sims,
        "seed": samples.seed,
        "model_version": samples.model_version,
        "model_sha256": model_hash,
        "sample_sha256": sample_hash,
        "runtime_sha256": runtime_hash,
        "state_hash": samples.state_hash,
        "reproducibility": "EXACT_WITHIN_RECORDED_RUNTIME",
        "status": samples.status,
    }
    manifest["artifact_sha256"] = content_sha256(manifest)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(manifest))
    return manifest

def freeze_bundle(samples, model, runtime, output_dir: str | Path) -> dict[str, Any]:
    """Write a deterministic local bundle and refuse content collisions."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_payload = canonical_bytes(samples)
    sample_path = out / "simulation.json.gz"
    compressed = gzip.compress(sample_payload, compresslevel=9, mtime=0)
    if sample_path.exists() and sample_path.read_bytes() != compressed:
        raise ValueError("frozen sample path already contains different bytes")
    sample_path.write_bytes(compressed)
    manifest = freeze_simulation(samples, model, runtime)
    manifest.update({"sample_file": sample_path.name,
                     "sample_file_sha256": hashlib.sha256(compressed).hexdigest()})
    manifest["artifact_sha256"] = content_sha256({k: v for k, v in manifest.items() if k != "artifact_sha256"})
    manifest_path = out / "manifest.json"
    manifest_bytes = canonical_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("frozen manifest path already contains different bytes")
    manifest_path.write_bytes(manifest_bytes)
    return manifest

def verify_frozen_artifact(manifest: dict[str, Any]) -> bool:
    if not isinstance(manifest, dict) or "artifact_sha256" not in manifest:
        return False
    supplied = manifest["artifact_sha256"]
    clone = dict(manifest)
    clone.pop("artifact_sha256", None)
    return supplied == content_sha256(clone)
