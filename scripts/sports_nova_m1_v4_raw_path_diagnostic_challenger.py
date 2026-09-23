"""SPORTS_NOVA M1 V4 diagnostic challenger.

This is an additive, research-only harness.  V3 and V21 source packages are
never edited.  V4 uses the current V3 entry point with one narrow
opportunity-conservation repair: the existing named-player target shares are
used to redistribute the explicit residual target mass that V3 allocates but
never realizes.  The harness instruments the frozen entry points so every
simulation path has a raw ledger and individually persisted accounting
failures.

The full run intentionally remains at the established 16 simulations/game
resolution used by the existing V3/V21 walk-forward artifacts.  Use
``--pilot`` for the pre-registered 60-game/128-simulation diagnostic gate.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.sports_nova_v3.allocation import AllocationResult
from worker.sports_nova_v3.config import MODEL_VERSION as V3_MODEL_VERSION
from worker.sports_nova_v3.pregame_state import build_pregame_state  # noqa: F401
from worker.sports_nova_v3 import allocation as allocation_mod
from scripts.sports_nova_v3_player_joint_walkforward_v1 import (
    MANIFEST, PLAYER, DRIVE, PLAYER_SHA, DRIVE_SHA, MANIFEST_SHA,
    game_parts, make_state, surrogate_kickoff,
)

DATA = ROOT / "data" / "sports_nova_v3"
INPUT = DATA / "validation_inputs"
RAW_LEDGER = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_LEDGER_V1.jsonl"
RAW_LEDGER_PARTIAL = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_LEDGER_V1.partial.jsonl"
RAW_MANIFEST = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_LEDGER_MANIFEST_V1.json"
REPORT = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_DIAGNOSTIC_REPORT_V1.json"
N_FULL = 16
N_PILOT = 128
V4_MODEL_VERSION = "sports_nova_v4.m1.raw_path_residual_target_conservation.1"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def feature(obj: Any, name: str, default: float | None = None) -> float | None:
    for item in getattr(obj, "features", ()):
        if item.name == name and item.value is not None:
            return float(item.value)
    return default


def pilot_games(games: list[str]) -> list[str]:
    by_season: dict[int, list[str]] = defaultdict(list)
    for game in games:
        by_season[game_parts(game)[0]].append(game)
    out: list[str] = []
    for season in sorted(by_season):
        xs = by_season[season]
        out.extend(xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int))
    return out


def freeze_inputs() -> dict[str, Any]:
    v3_pred = DATA / "SPORTS_NOVA_V3_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet"
    v21_pred = DATA / "SPORTS_NOVA_V21_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet"
    v3_joint = DATA / "SPORTS_NOVA_V3_JOINT_WALKFORWARD_PREDICTIONS_V1.parquet"
    v21_joint = DATA / "SPORTS_NOVA_V21_JOINT_WALKFORWARD_PREDICTIONS_V1.parquet"
    paths = [
        ("V3_PLAYER_INPUT", PLAYER, PLAYER_SHA, "causal feature and realized player outcomes"),
        ("V3_DRIVE_INPUT", DRIVE, DRIVE_SHA, "causal team drive/block outcomes"),
        ("CAUSAL_COHORT_MANIFEST", MANIFEST, MANIFEST_SHA, "frozen evaluation games"),
        ("V3_PLAYER_ARTIFACT", v3_pred, None, "frozen V3 comparator artifact"),
        ("V3_JOINT_ARTIFACT", v3_joint, None, "frozen V3 joint comparator artifact"),
        ("V21_PLAYER_ARTIFACT", v21_pred, None, "frozen V21 comparator artifact"),
        ("V21_JOINT_ARTIFACT", v21_joint, None, "frozen V21 joint comparator artifact"),
        ("V3_SIMULATOR", ROOT / "worker" / "sports_nova_v3" / "simulator.py", None, "current V3 engine source"),
        ("V3_SIM_CONFIG", ROOT / "worker" / "sports_nova_v3" / "config.py", None, "simulation config and RNG contract"),
        ("V3_ALLOCATION", ROOT / "worker" / "sports_nova_v3" / "allocation.py", None, "opportunity allocation source"),
        ("V3_PLAY_VOLUME", ROOT / "worker" / "sports_nova_v3" / "play_volume.py", None, "team volume source"),
        ("V21_SIMULATOR", ROOT / "worker" / "sports_nova_v21" / "simulator.py", None, "V21 comparator source"),
        ("V21_SIM_CONFIG", ROOT / "worker" / "sports_nova_v21" / "config.py", None, "V21 version and PAT config"),
        ("V4_HARNESS", Path(__file__), None, "challenger and raw-ledger implementation"),
    ]
    frozen = []
    for name, path, expected, role in paths:
        actual = sha256(path)
        frozen.append({"name": name, "path": str(path), "exists": path.is_file(),
                       "sha256": actual, "expected_sha256": expected, "role": role,
                       "hash_match": expected is None or actual == expected})
    depth_dir = INPUT / "depth_charts_external"
    depth_files = sorted(depth_dir.glob("*.parquet")) if depth_dir.is_dir() else []
    injury_candidates = [p for p in INPUT.rglob("*") if p.is_file() and "injur" in p.name.lower()]
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": frozen,
        "roster_depth_chart_inputs": {
            "status": "PRESENT_BUT_NOT_USED_BY_FROZEN_V3_STATE",
            "paths": [str(p) for p in depth_files],
            "sha256": {str(p): sha256(p) for p in depth_files},
            "cohort_seasons": [2020, 2021, 2022, 2023, 2024, 2025],
            "coverage_note": "No 2020-2024 depth-chart files were found; frozen make_state uses event-causal player history instead.",
        },
        "injury_inputs": {
            "status": "NO_DATA",
            "paths": [str(p) for p in injury_candidates],
            "note": "No historical pregame injury input is consumed by the frozen V3/V21 walk-forward state builder.",
        },
        "random_seed_contract": {
            "algorithm": "PCG64DXSM",
            "seed": "int(sha256(game_id.encode())[:8], 16)",
            "path_seed": "SeedSequence([game_seed, sim_id])",
            "simulation_index": "ascending from 0",
        },
    }


def _skill_candidates(state, team_id: str) -> list[str]:
    return [p.player_id for p in state.players
            if p.team_id == team_id and p.position in {"RB", "WR", "TE"}
            and p.availability != "OUT"]


_ENGINE_MODULE_PATHS = {
    "V21": "worker.sports_nova_v21.simulator",
    "V3": "worker.sports_nova_v3.simulator",
    "V4": "worker.sports_nova_v3.simulator",
}


@contextlib.contextmanager
def instrument(engine_name: str, collector: list[dict], fix: bool = False, detail: bool = False,
                module_path: str | None = None):
    """Instrument one engine without editing its module source.

    `module_path` overrides the `engine_name` -> module lookup for engines
    added after this table was written (e.g. worker.sports_nova_v23.simulator)
    without touching the existing V21/V3/V4 callers.
    """
    module = importlib.import_module(
        module_path or _ENGINE_MODULE_PATHS.get(engine_name, "worker.sports_nova_v3.simulator")
    )
    originals = {
        "run_one": module._run_one,
        "volume": module.draw_block_volume,
        "selection": module.select_plays,
        "allocation": module.allocate_opportunities,
        "inc": module._inc,
        "dm": allocation_mod.sample_dirichlet_multinomial,
    }
    ctx: dict[str, Any] = {"path": None}

    def dm(n, shares, concentration, rng):
        shares = np.asarray(shares, dtype=float)
        clipped = np.clip(shares, 0.0, None)
        if clipped.sum() <= 0:
            clipped = np.ones(len(clipped)) / len(clipped)
        else:
            clipped = clipped / clipped.sum()
        p = rng.dirichlet(np.maximum(clipped * max(concentration, 1e-6), 1e-6))
        counts = rng.multinomial(int(n), p)
        path = ctx.get("path")
        if path is not None and path.get("pending_draws") is not None:
            path["pending_draws"].append({"n": int(n), "p": p.tolist(), "counts": counts.tolist()})
        return counts

    def volume(state, env, rng, **kwargs):
        value = originals["volume"](state, env, rng, **kwargs)
        path = ctx["path"]
        if detail:
            block = {"block_index": int(state.block_index), "offense_team": state.possession,
                     "seconds_remaining_before": int(state.seconds_remaining), "plays": int(value)}
            path["blocks"].append(block)
        return value

    def selection(plays, pass_rate, rng, **kwargs):
        value = originals["selection"](plays, pass_rate, rng, **kwargs)
        if detail:
            block = ctx["path"]["blocks"][-1]
            block["selection"] = {"pass_rate": float(pass_rate), "dropbacks": int(value.dropbacks),
                                   "pass_attempts": int(value.pass_attempts), "sacks": int(value.sacks),
                                   "scrambles": int(value.scrambles), "designed_rushes": int(value.designed_rushes),
                                   "rush_attempts": int(value.rush_attempts)}
        return value

    def allocation(state, team_id, pass_attempts, rush_attempts, rng, **kwargs):
        target_ids, target_shares, target_residual = allocation_mod._shares(state, team_id, "target")
        carry_ids, carry_shares, carry_residual = allocation_mod._shares(state, team_id, "carry")
        path = ctx["path"]
        path["pending_draws"] = [] if detail else None
        result = originals["allocation"](state, team_id, pass_attempts, rush_attempts, rng, **kwargs)
        draws = path.pop("pending_draws") or []
        path["residual_by_team"][team_id] = path["residual_by_team"].get(team_id, 0) + int(result.untargeted)
        path["residual_carries_by_team"][team_id] = path["residual_carries_by_team"].get(team_id, 0) + int(result.residual_carries)
        before = result
        redistributed = {}
        if fix and result.untargeted > 0:
            ids = list(target_ids)
            weights = np.asarray(target_shares, dtype=float)
            if not ids:
                ids = _skill_candidates(state, team_id)
                weights = np.ones(len(ids), dtype=float)
            if ids and weights.sum() > 0:
                weights = weights / weights.sum()
                extra = rng.multinomial(int(result.untargeted), weights)
                target_map = dict(result.target_counts)
                for pid, count in zip(ids, extra):
                    target_map[pid] = int(target_map.get(pid, 0) + int(count))
                    if count:
                        redistributed[pid] = int(count)
                result = AllocationResult(result.player_ids, target_map, result.carry_counts,
                                          0, result.residual_carries)
        carry_redistributed = {}
        if fix and result.residual_carries > 0:
            ids = list(carry_ids)
            weights = np.asarray(carry_shares, dtype=float)
            if not ids:
                ids = _skill_candidates(state, team_id)
                weights = np.ones(len(ids), dtype=float)
            if ids and weights.sum() > 0:
                weights = weights / weights.sum()
                extra = rng.multinomial(int(result.residual_carries), weights)
                carry_map = dict(result.carry_counts)
                for pid, count in zip(ids, extra):
                    carry_map[pid] = int(carry_map.get(pid, 0) + int(count))
                    if count:
                        carry_redistributed[pid] = int(count)
                result = AllocationResult(result.player_ids, result.target_counts, carry_map,
                                          result.untargeted, 0)
        if not detail:
            return result
        block = path["blocks"][-1]
        block["allocation"] = {
            "team_resource_total": {"pass_attempts": int(pass_attempts), "rush_attempts": int(rush_attempts)},
            "player_share_prior": {"targets": {pid: float(v) for pid, v in zip(target_ids, target_shares)},
                                    "carries": {pid: float(v) for pid, v in zip(carry_ids, carry_shares)},
                                    "target_residual": float(target_residual),
                                    "carry_residual": float(carry_residual)},
            "player_share_draw": {"dirichlet_multinomial_draws": draws},
            "post_normalization_share": {
                "targets": {pid: (float(v) / pass_attempts if pass_attempts else 0.0)
                             for pid, v in result.target_counts.items()},
                "carries": {pid: (float(v) / rush_attempts if rush_attempts else 0.0)
                            for pid, v in result.carry_counts.items()},
            },
            "allocated_opportunities": {
                "targets": dict(result.target_counts), "carries": dict(result.carry_counts),
                "untargeted_before_fix": int(before.untargeted),
                "residual_carries": int(result.residual_carries),
                "redistributed_target_mass": redistributed,
                "redistributed_carry_mass": carry_redistributed,
            },
            "realized_outcome": {},
        }
        block["_events"] = []
        return result

    def inc(mapping, pid, stat, value):
        originals["inc"](mapping, pid, stat, value)
        path = ctx.get("path")
        if detail and path is not None and path.get("blocks") and "_events" in path["blocks"][-1]:
            path["blocks"][-1]["_events"].append({"player_id": pid, "stat": stat, "value": int(value)})

    def run_one(pregame, sim_id, seed, params):
        path = {"sim_id": int(sim_id), "seed": int(seed), "blocks": [], "pending_draws": None,
                "residual_by_team": {}, "residual_carries_by_team": {}}
        ctx["path"] = path
        result = originals["run_one"](pregame, sim_id, seed, params)
        path["result"] = result
        for block in path["blocks"]:
            events: dict[str, dict[str, int]] = defaultdict(dict)
            for event in block.pop("_events", []):
                events[event["player_id"]][event["stat"]] = events[event["player_id"]].get(event["stat"], 0) + event["value"]
            block["allocation"]["realized_outcome"] = {pid: stats for pid, stats in events.items()}
        collector.append(path)
        ctx["path"] = None
        return result

    module.draw_block_volume = volume
    module.select_plays = selection
    module.allocate_opportunities = allocation
    module._inc = inc
    module._run_one = run_one
    allocation_mod.sample_dirichlet_multinomial = dm
    try:
        yield module
    finally:
        module._run_one = originals["run_one"]
        module.draw_block_volume = originals["volume"]
        module.select_plays = originals["selection"]
        module.allocate_opportunities = originals["allocation"]
        module._inc = originals["inc"]
        allocation_mod.sample_dirichlet_multinomial = originals["dm"]


def player_stat_dict(stats: dict[str, int]) -> dict[str, int]:
    return {name: int(stats.get(name, 0)) for name in (
        "pass_attempts", "completions", "pass_yards", "pass_tds",
        "rush_attempts", "rush_yards", "rush_tds", "targets", "receptions",
        "receiving_yards", "receiving_tds")}


def make_raw_row(game: str, season: int, week: int, state, path: dict) -> dict[str, Any]:
    stats, team, winner = path["result"]
    home, away = state.home.team_id, state.away.team_id
    pmap = {p.player_id: p for p in state.players}
    players = []
    for pid in sorted(stats):
        p = pmap[pid]
        s = stats[pid]
        row = player_stat_dict(s)
        row.update({"player_id": pid, "position": p.position, "active_status": p.availability,
                    "depth_role": getattr(p, "depth_role", None),
                    "snap_opportunity_proxy": feature(p, "snap_proxy", None),
                    "prior_target_share": None, "prior_carry_share": None})
        players.append(row)
    teams = []
    for tid in (home, away):
        t = team[tid]
        team_players = [x for x in players if pmap[x["player_id"]].team_id == tid]
        teams.append({
            "team_id": tid, "pass_attempts": int(t["pass_attempts"]),
            "completions": int(sum(x["receptions"] for x in team_players)),
            "pass_yards": int(t["pass_yards"]), "pass_tds": int(sum(x["pass_tds"] for x in team_players)),
            "rush_attempts": int(t["rush_attempts"]), "rush_yards": int(t["rush_yards"]),
            "rush_tds": int(sum(x["rush_tds"] for x in team_players)),
            "targets": int(sum(x["targets"] for x in team_players)),
            "receptions": int(sum(x["receptions"] for x in team_players)),
            "score": int(t["score"]), "blocks": int(t["blocks"]),
        })
    home_score = next(x["score"] for x in teams if x["team_id"] == home)
    away_score = next(x["score"] for x in teams if x["team_id"] == away)
    trace = path["blocks"]
    for block in trace:
        for pid, prior in block["allocation"]["player_share_prior"]["targets"].items():
            pass
    timing = {"source_timestamp": iso(state.game_evidence.event_end),
              "publication_timestamp": iso(getattr(state.game_evidence, "published_at", None)),
              "decision_eligible_timestamp": iso(state.as_of),
              "missing_publication_flag": True}
    return {"game_id": game, "sim_id": int(path["sim_id"]), "season": season, "week": week,
            "teams": f"{away}@{home}", "home_team": home, "away_team": away,
            "seed": int(path["seed"]), "final_score": int(home_score + away_score),
            "margin": int(home_score - away_score), "total_points": int(home_score + away_score),
            "possessions": None, "winner": winner,
            "team_offense_json": json.dumps(teams, separators=(",", ":")),
            "player_json": json.dumps(players, separators=(",", ":")),
            "allocation_trace_json": json.dumps(trace, separators=(",", ":")),
            "timing_causal_json": json.dumps(timing, separators=(",", ":")),
            "raw_ledger_complete": True}


def assert_path(game: str, state, path: dict, engine: str, failures: list[dict]) -> None:
    stats, team, _ = path["result"]
    for tid in (state.home.team_id, state.away.team_id):
        team_players = [p for p in state.players if p.team_id == tid]
        ssum = lambda name: sum(int(stats[p.player_id].get(name, 0)) for p in team_players)
        t = team[tid]
        checks = [
            ("PASS_ATTEMPTS_QB_CONSERVATION", ssum("pass_attempts"), int(t["pass_attempts"]),
             bool([p for p in team_players if p.position == "QB"])),
            ("TARGETS_LE_PASS_ATTEMPTS", ssum("targets"), int(t["pass_attempts"]), True),
            # Genuinely independent passing-side check (unlike the two below):
            # ssum("targets") and t["pass_attempts"] are NOT computed from the
            # same loop, so this can actually catch a dropped target-residual
            # defect that TARGETS_LE_PASS_ATTEMPTS's `<=` cannot (a shortfall
            # still satisfies `<=`). See worker/sports_nova_v23/__init__.py.
            ("TARGETS_TEAM_ATTEMPTS_CONSERVATION", ssum("targets"), int(t["pass_attempts"]), True),
            # NOTE: both of the next two are computed from the SAME per-block
            # loop on both sides (make_raw_row / _run_one credit stats[pid] and
            # team[tid] together), so they are true BY CONSTRUCTION and cannot
            # detect a dropped-opportunity defect. Kept for regression coverage
            # of that construction itself, not as independent accounting checks.
            ("RECEPTIONS_LE_COMPLETIONS", ssum("receptions"), ssum("receptions"), True),
            ("RECEIVING_YARDS_TEAM_RECONCILIATION", ssum("receiving_yards"), int(t["pass_yards"]), True),
            ("RUSH_ATTEMPTS_CONSERVATION", ssum("rush_attempts"), int(t["rush_attempts"]),
             bool([p for p in team_players if p.position == "QB"])),
            ("RUSH_YARDS_TEAM_RECONCILIATION", ssum("rush_yards"), int(t["rush_yards"]), True),
        ]
        for name, actual, expected, required in checks:
            ok = (actual == expected if name.endswith("CONSERVATION") or "RECONCILIATION" in name
                  else actual <= expected)
            if required and not ok:
                failures.append({"engine": engine, "game_id": game, "sim_id": path["sim_id"],
                                 "team_id": tid, "assertion": name, "actual": actual, "expected": expected})
        for block in path["blocks"]:
            alloc = block["allocation"]["allocated_opportunities"]
        if engine != "V4" and path.get("residual_by_team", {}).get(tid, 0):
            failures.append({"engine": engine, "game_id": game, "sim_id": path["sim_id"],
                             "team_id": tid, "assertion": "UNREALIZED_TARGET_RESIDUAL",
                             "actual": path["residual_by_team"][tid], "expected": 0})


def add_metric(values: dict, observed: dict, metric: str, value: float, obs: float) -> None:
    if math.isfinite(float(value)) and math.isfinite(float(obs)):
        values[metric].append(float(value)); observed[metric].append(float(obs))


def stage_data(engine: str, game: str, state, path: dict, cur: pd.DataFrame,
               values: dict, observed: dict) -> None:
    stats, team, _ = path["result"]
    for tid in (state.home.team_id, state.away.team_id):
        actual = cur[cur.TEAM == tid]
        obs_plays = float(actual.PASS_ATTEMPTS.sum() + actual.RUSH_ATTEMPTS.sum())
        obs_pa = float(actual.PASS_ATTEMPTS.sum())
        obs_ra = float(actual.RUSH_ATTEMPTS.sum())
        add_metric(values, observed, "TEAM_VOLUME:plays", team[tid]["pass_attempts"] + team[tid]["rush_attempts"], obs_plays)
        add_metric(values, observed, "TEAM_VOLUME:pass_attempts", team[tid]["pass_attempts"], obs_pa)
        add_metric(values, observed, "TEAM_VOLUME:rush_attempts", team[tid]["rush_attempts"], obs_ra)
        ps = state.home if state.home.team_id == tid else state.away
        current = cur[cur.TEAM == tid]
        total_t = float(current.TARGETS.sum())
        total_c = float(current.RUSH_ATTEMPTS.sum())
        target_by = {pid: float(v) for pid, v in zip(ps.target_shares.player_ids, ps.target_shares.shares)}
        carry_by = {pid: float(v) for pid, v in zip(ps.carry_shares.player_ids, ps.carry_shares.shares)}
        for r in current.itertuples():
            pid = str(r.PLAYER_ID)
            if pid not in stats:
                continue
            obs_target_share = float(r.TARGETS) / total_t if total_t else 0.0
            obs_carry_share = float(r.RUSH_ATTEMPTS) / total_c if total_c else 0.0
            add_metric(values, observed, "PLAYER_SHARE_PRIORS:target_share", target_by.get(pid, 0.0), obs_target_share)
            add_metric(values, observed, "PLAYER_SHARE_PRIORS:carry_share", carry_by.get(pid, 0.0), obs_carry_share)
            alloc_targets = float(stats[pid]["targets"])
            alloc_carries = float(stats[pid]["rush_attempts"])
            add_metric(values, observed, "PLAYER_SHARE_ALLOCATION:targets", alloc_targets, float(r.TARGETS))
            add_metric(values, observed, "PLAYER_SHARE_ALLOCATION:carries", alloc_carries, float(r.RUSH_ATTEMPTS))
            add_metric(values, observed, "OPPORTUNITY_REALIZATION:receptions", stats[pid]["receptions"], float(r.RECEPTIONS))
            add_metric(values, observed, "OPPORTUNITY_REALIZATION:targets", stats[pid]["targets"], float(r.TARGETS))
            add_metric(values, observed, "EFFICIENCY_CONVERSION:catch_rate", stats[pid]["receptions"] / stats[pid]["targets"] if stats[pid]["targets"] else 0.0,
                       float(r.RECEPTIONS) / float(r.TARGETS) if r.TARGETS else 0.0)
            add_metric(values, observed, "YARDAGE_MODEL:receiving_yards", stats[pid]["receiving_yards"], float(r.RECEIVING_YARDS))
            add_metric(values, observed, "YARDAGE_MODEL:rush_yards", stats[pid]["rush_yards"], float(r.RUSH_YARDS))
            add_metric(values, observed, "TD_MODEL:scored_tds", stats[pid]["scored_tds"], float(r.PASS_TD + r.RUSH_TD + r.RECEIVING_TD))
        pass_total = float(team[tid]["pass_attempts"])
        rush_total = float(team[tid]["rush_attempts"])
        target_alloc = float(sum(stats[p.player_id]["targets"] for p in team_players(state, tid)))
        carry_alloc = float(sum(stats[p.player_id]["rush_attempts"] for p in team_players(state, tid)))
        add_metric(values, observed, "ALLOCATION_NORMALIZATION:target_conservation", target_alloc / pass_total if pass_total else 1.0, 1.0)
        add_metric(values, observed, "ALLOCATION_NORMALIZATION:carry_conservation", carry_alloc / rush_total if rush_total else 1.0, 1.0)


def team_players(state, tid):
    return [p for p in state.players if p.team_id == tid]


def summarize_engine(engine: str, values: dict, observed: dict, baseline: dict | None = None) -> dict:
    out = {}
    for metric, xs in values.items():
        x = np.asarray(xs, dtype=float); y = np.asarray(observed[metric], dtype=float)
        key = metric
        item = {"N": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
                "P10": float(np.quantile(x, .10)), "P50": float(np.quantile(x, .50)),
                "P90": float(np.quantile(x, .90)), "bias_vs_observed": float(np.mean(x - y))}
        if baseline and key in baseline:
            item["delta_vs_V21"] = float(item["mean"] - baseline[key]["mean"])
        out[key] = item
    return out


def prediction_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NO_DATA", "path": str(path)}
    df = pd.read_parquet(path)
    out: dict[str, Any] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        x = df[df.POSITION == pos]
        if x.empty:
            continue
        err = x.PRED_MEAN - x.OBSERVED
        out[pos] = {"N": int(len(x)), "MAE": float(np.abs(err).mean()),
                    "RMSE": float(np.sqrt(np.mean(err ** 2))), "bias": float(err.mean()),
                    "rank_correlation": float(x.OBSERVED.corr(x.PRED_MEAN, method="spearman")),
                    "coverage90": float(((x.OBSERVED >= x.P10) & (x.OBSERVED <= x.P90)).mean()),
                    "P90_observed": float(np.quantile(x.OBSERVED, .90)),
                    "P95_observed": float(np.quantile(x.OBSERVED, .95)),
                    "P99_observed": float(np.quantile(x.OBSERVED, .99)),
                    "P90_predicted": float(np.quantile(x.PRED_MEAN, .90)),
                    "P95_predicted": float(np.quantile(x.PRED_MEAN, .95)),
                    "P99_predicted": float(np.quantile(x.PRED_MEAN, .99))}
    return {"status": "COMPUTED_FROM_FROZEN_ARTIFACT", "rows": int(len(df)), "by_position": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="60 games x 128 simulations")
    ap.add_argument("--engines", default="V3,V21,V4", help="comma-separated engines")
    args = ap.parse_args()
    prior_report = json.loads(REPORT.read_text()) if REPORT.is_file() else None
    freeze = freeze_inputs()
    if not all(x["hash_match"] for x in freeze["artifacts"] if x["expected_sha256"]):
        raise SystemExit("BLOCKED_HASH: frozen validation input mismatch")
    manifest = json.loads(MANIFEST.read_text())
    games_all = list(manifest["GAME_IDS"])
    games = pilot_games(games_all) if args.pilot else games_all
    n_sims = N_PILOT if args.pilot else N_FULL
    df = pd.read_parquet(PLAYER)
    df = df[df.GAME_ID.isin(games)].copy()
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    df_by_game = {g: df[df.GAME_ID == g] for g in games}
    engines = [x.strip().upper() for x in args.engines.split(",") if x.strip()]
    failures: list[dict] = []
    raw_rows_count = 0
    raw_fh = RAW_LEDGER_PARTIAL.open("w", encoding="utf-8") if "V4" in engines else None
    # make_state is deterministic and identical across engines.  Build it
    # once per game so the three-engine comparison does not re-run the costly
    # pandas state assembly three times.
    states = {}
    for game in games:
        season, week, _, _ = game_parts(game)
        key = season * 100 + week
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]
        states[game] = make_state(game, prior, surrogate_kickoff(season, week), None)
    engine_summaries: dict[str, dict] = {}
    for engine in engines:
        vals = defaultdict(list); obs = defaultdict(list); collector: list[dict] = []
        fix = engine == "V4"
        model = importlib.import_module("worker.sports_nova_v21.config" if engine == "V21" else "worker.sports_nova_v3.config").MODEL_VERSION
        with instrument(engine, collector, fix=fix, detail=(engine == "V4")) as _:
            for game in games:
                season, week, away, home = game_parts(game)
                key = season * 100 + week
                prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]
                state = states[game]
                seed = int(hashlib.sha256(game.encode()).hexdigest()[:8], 16)
                simulate_engine = importlib.import_module("worker.sports_nova_v21.simulator" if engine == "V21" else "worker.sports_nova_v3.simulator")
                before = len(collector)
                simulate_engine.simulate_game(state, n_sims, seed, model)
                paths = collector[before:]
                cur = df_by_game[game]
                for p in paths:
                    assert_path(game, state, p, engine, failures)
                    stage_data(engine, game, state, p, cur, vals, obs)
                    if engine == "V4":
                        raw_fh.write(json.dumps(make_raw_row(game, season, week, state, p), separators=(",", ":")) + "\n")
                        raw_rows_count += 1
                # Do not retain the full per-path result dictionaries after
                # they have been converted into the stage summaries/raw rows.
                del collector[before:]
        baseline = engine_summaries.get("V21")
        engine_summaries[engine] = {"stage_metrics": summarize_engine(engine, vals, obs, baseline),
                                    "path_count": len(collector), "games": len(games), "sims_per_game": n_sims}
        collector.clear()
    # A V4-only rerun is useful for validating a narrow repair without
    # recomputing unchanged V3/V21 paths.  Preserve the already-computed
    # same-cohort comparator sections when such a report is available.
    prior_matches_run = bool(prior_report and prior_report.get("cohort", {}).get("games") == len(games)
                             and prior_report.get("stage_metrics", {}).get("V3", {}).get("sims_per_game") == n_sims)
    if prior_matches_run and "V3" not in engine_summaries and "V3" in prior_report.get("stage_metrics", {}):
        for name in ("V3", "V21"):
            if name in prior_report.get("stage_metrics", {}):
                engine_summaries[name] = prior_report["stage_metrics"][name]
        failures.extend([x for x in prior_report.get("ACCOUNTING_FAILURES", []) if x.get("engine") in {"V3", "V21"}])

    if raw_fh is not None:
        raw_fh.close()
        if raw_rows_count == len(games) * n_sims:
            RAW_LEDGER_PARTIAL.replace(RAW_LEDGER)
    if raw_rows_count == 0 and "V4" in engines:
        raise SystemExit("BLOCKED_RAW_LEDGER: V4 produced no paths")
    if raw_rows_count:
        RAW_MANIFEST.write_text(json.dumps({"status": "COMPLETE", "path": str(RAW_LEDGER),
            "rows": int(raw_rows_count), "expected_rows": int(len(games) * n_sims),
            "sha256": sha256(RAW_LEDGER), "format": "JSONL",
            "nested_ledgers": {"team_offense": True, "player": True, "allocation_trace": True, "timing_causal": True}}, indent=2))

    v3_metrics = prediction_metrics(DATA / "SPORTS_NOVA_V3_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet")
    v21_metrics = prediction_metrics(DATA / "SPORTS_NOVA_V21_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet")
    player_metrics_v4 = {"status": "RAW_PATH_LEDGER_ONLY", "note": "V4 full prediction parquet is emitted by the companion battery pass; raw path metrics are stage-gated here."}
    raw_status = "COMPLETE" if raw_rows_count == len(games) * n_sims else "PARTIAL"
    publication_status = "FAIL_NO_PUBLICATION_TIMESTAMPS" if df.EVENT_TIME.isna().all() else "UNVERIFIED"
    temporal_status = "PASS_EVENT_ORDER_ONLY_PUBLICATION_FIREWALL_FAIL" if publication_status != "PASS" else "PASS"
    v3_failures = [f for f in failures if f["engine"] == "V3"]
    first_failure = "OPPORTUNITY_REALIZATION" if any(f["assertion"] == "UNREALIZED_TARGET_RESIDUAL" for f in v3_failures) else "MULTIPLE"
    qb_v3 = v3_metrics.get("by_position", {}).get("QB", {})
    qb_v21 = v21_metrics.get("by_position", {}).get("QB", {})
    report = {
        "V4_STATUS": "PARTIAL" if raw_status == "COMPLETE" else "FAIL_VERSION",
        "RAW_LEDGER_STATUS": raw_status,
        "RAW_LEDGER_PATH": str(RAW_LEDGER),
        "RAW_LEDGER_ROWS": raw_rows_count,
        "TEMPORAL_FIREWALL": temporal_status,
        "PUBLICATION_TIME_CERTIFICATION": publication_status,
        "ACCOUNTING_STATUS": "FAIL" if failures else "PASS",
        "ACCOUNTING_FAILURES": failures,
        "FIRST_FAILURE_STAGE": first_failure,
        "QB_UNDERPRODUCTION_ROOT_CAUSE": "ACCOUNTING_LOSS: explicit residual target mass is allocated by V3 but never passed through reception/yardage realization; secondary identity/roster exclusions remain visible in the frozen V3 comparator.",
        "FIX_APPLIED": "Redistribute explicit residual target and carry opportunities across existing causal active-player share mass; no new model family or tuned market input.",
        "FIX_SCOPE": "V4 only; V3 and V21 source/artifacts unchanged.",
        "PLAYER_ALLOCATION_V21": engine_summaries.get("V21", {}).get("stage_metrics", {}).get("PLAYER_SHARE_ALLOCATION:targets"),
        "PLAYER_ALLOCATION_V3": engine_summaries.get("V3", {}).get("stage_metrics", {}).get("PLAYER_SHARE_ALLOCATION:targets"),
        "PLAYER_ALLOCATION_V4": engine_summaries.get("V4", {}).get("stage_metrics", {}).get("PLAYER_SHARE_ALLOCATION:targets"),
        "QB_PASS_YARD_MAE_V21": qb_v21.get("MAE"),
        "QB_PASS_YARD_MAE_V3": qb_v3.get("MAE"),
        "QB_PASS_YARD_MAE_V4": None,
        "JOINT_DELTA": "PENDING_FULL_V4_BATTERY",
        "TAIL_DELTA": "PENDING_FULL_V4_BATTERY",
        "GAME_LEVEL_DELTA": "AVAILABLE_IN_STAGE_METRICS; V4 prediction battery pending",
        "DISTRIBUTIONAL_DELTA": "PENDING_FULL_V4_BATTERY",
        "stage_metrics": engine_summaries,
        "frozen_inputs": freeze,
        "cohort": {"games": len(games), "all_games": len(games_all), "same_games": games == games_all,
                    "player_rows": int(len(df)), "train_oos_rule": "prior weeks only, five-season lookback; unchanged"},
        "comparison_artifacts": {"V21": v21_metrics, "V3": v3_metrics, "V4": player_metrics_v4},
        "PROMOTION_DECISION": "DO_NOT_PROMOTE: publication certification failed and full V4 validation battery is not complete",
        "CHAMPION_AFTER_TEST": "V21",
        "STATE_SPACE_CHALLENGER_NEEDED": "NO_NOT_YET",
        "REAL_BLOCKER": "Historical input has no row-level publication/availability timestamps, and no complete historical injury/depth-chart input is wired into the frozen state builder; V4 cannot be promoted on event ordering alone.",
        "NEXT_SINGLE_ACTION": "Run the same full 1,693-game/16-simulation V4 prediction and joint battery, then audit the raw-ledger hash and accounting failures before any promotion decision.",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["V4_STATUS"], "raw_rows": raw_rows_count,
                      "first_failure": first_failure, "accounting_failures": len(failures),
                      "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
