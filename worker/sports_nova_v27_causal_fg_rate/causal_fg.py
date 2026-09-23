"""Causal FG-rate resolution (estimator A), fail-closed.

The frozen estimator script is loaded from its file AFTER its sha256 is checked against the pinned value; the frozen drive-block table is hash-verified once
and cached.  Season/week come from the game_id, not from `as_of`.  Any problem raises CausalFGError -- there is no default and no fallback.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DRIVE_RELPATH, DRIVE_SHA256, ESTIMATOR_RELPATH, ESTIMATOR_SHA256

ROOT = Path(__file__).resolve().parents[2]
_GAME_ID = re.compile(r"^(\d{4})_(\d{1,2})_([A-Z0-9]+)_([A-Z0-9]+)$")


class CausalFGError(RuntimeError):
    """The causal FG rate could not be produced.  The simulation must not run."""


@dataclass(frozen=True)
class CausalFGRate:
    rate: float
    n_eligible: int
    made: int
    wilson95: tuple
    season: int
    week: int
    cutoff_key: int
    max_key_used: int
    window_min_season: int
    window_blocks: int
    estimator_sha256: str
    drive_sha256: str

    def evidence(self) -> dict:
        return {"fg_rate": self.rate, "fg_n_eligible": self.n_eligible, "fg_made": self.made, "fg_wilson95": list(self.wilson95), "fg_season": self.season,
                "fg_week": self.week, "fg_cutoff_key": self.cutoff_key, "fg_max_key_used": self.max_key_used, "fg_window_min_season": self.window_min_season,
                "fg_window_blocks": self.window_blocks, "fg_estimator_sha256": self.estimator_sha256, "fg_drive_sha256": self.drive_sha256}


def parse_game_key(game_id: str) -> tuple[int, int]:
    m = _GAME_ID.match(str(game_id))
    if not m:
        raise CausalFGError(f"game_id {game_id!r} is not 'SEASON_WEEK_AWAY_HOME'; cannot place the simulated game in time")
    return int(m.group(1)), int(m.group(2))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_STATE: dict = {}


def _estimator():
    if "mod" in _STATE:
        return _STATE["mod"]
    path = ROOT / ESTIMATOR_RELPATH
    try:
        got = _sha256(path)
    except OSError as e:
        raise CausalFGError(f"estimator unavailable: {path}: {e}") from e
    if got != ESTIMATOR_SHA256:
        raise CausalFGError(f"estimator hash mismatch: {got} != pinned {ESTIMATOR_SHA256}")
    spec = importlib.util.spec_from_file_location("_sports_nova_fg_causal_estimator_v1_pinned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.DRIVE_SHA != DRIVE_SHA256 or Path(mod.DRIVE).resolve() != (ROOT / DRIVE_RELPATH).resolve():
        raise CausalFGError("estimator points at a different drive table than the V27 pin")
    _STATE["mod"] = mod
    return mod


def _drives():
    if "d" not in _STATE:
        mod = _estimator()
        try:
            _STATE["d"] = mod.load_drives(verify=True)          # verifies sha256 of the parquet against the frozen value, once
        except Exception as e:  # noqa: BLE001
            raise CausalFGError(f"drive-block table unavailable or hash mismatch: {e}") from e
    return _STATE["d"]


def causal_fg_rate(season: int, week: int, game_id: str | None = None) -> CausalFGRate:
    key = (int(season), int(week), game_id)
    cache = _STATE.setdefault("rates", {})
    if key in cache:
        return cache[key]
    mod = _estimator()
    d = _drives()
    try:
        e = mod.estimate(d, int(season), int(week), game_id)    # asserts: key < cutoff, own game absent, window seasons
    except Exception as ex:  # noqa: BLE001
        raise CausalFGError(f"estimator failed for {season}_{week:02d}: {ex!r}") from ex
    cutoff = int(season) * 100 + int(week)
    p = mod.prior_window(d, int(season), int(week))
    # independent re-check of the temporal contract (the estimator asserts it too)
    if len(p) == 0 or int(p["key"].max()) >= cutoff or int(p["season"].min()) < max(1999, int(season) - mod.WINDOW_SEASONS):
        raise CausalFGError("temporally invalid estimator window")
    if game_id is not None and bool((p["game_id"] == game_id).any()):
        raise CausalFGError("simulated game present in its own estimator window")
    rate, n = float(e["rate"]), int(e["n_eligible"])
    if n <= 0 or not math.isfinite(rate) or not (0.0 < rate < 0.5):
        raise CausalFGError(f"unusable estimate rate={rate} n={n}")
    out = CausalFGRate(rate, n, int(e["made"]), tuple(e["wilson95"]), int(season), int(week), cutoff, int(e["max_key_used"]),
                       max(1999, int(season) - mod.WINDOW_SEASONS), int(e["window_blocks"]), ESTIMATOR_SHA256, DRIVE_SHA256)
    cache[key] = out
    return out


def causal_fg_rate_for_game(game_id: str) -> CausalFGRate:
    season, week = parse_game_key(game_id)
    return causal_fg_rate(season, week, game_id)
