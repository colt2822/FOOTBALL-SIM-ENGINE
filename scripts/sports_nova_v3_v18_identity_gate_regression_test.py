"""SPORTS_V18_IDENTITY_POLICY_SINGLE_ENFORCEMENT_POINT -- regression test.

Deliberately written as a "naive third entrypoint": imports the FROZEN
simulator's _primary_qb the same way any new capture script would, and never
imports scripts/sports_nova_v3_v18_qb_uncertainty_policy.py or calls
policy.classify_team()/enforce_fresh_identity() itself. If this script still
gets correct exclusion/blocking behavior, the canonical gate in
worker/sports_nova_v3/__init__.py + identity_gate.py is enforcing
independently of any capture script's own opt-in checks -- which is the
whole point of this mission.

Never imports or calls simulate_game, never writes to predictions/ or
predictions_live/, never touches an existing prediction artifact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
SIMULATOR_PATH = ROOT / "worker" / "sports_nova_v3" / "simulator.py"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
RESOLUTION_PATH = DATA / "prospective" / "identity" / "SPORTS_NOVA_V18_QB_IDENTITY_RESOLUTION_2026.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_walkforward():
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    hash_before = {rel: sha256_file(ROOT / rel) for rel in freeze["CODE_HASHES"]}
    spec = importlib.util.spec_from_file_location(
        "sports_nova_v3_player_joint_walkforward_v1_REGRESSION_TEST", WALKFORWARD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, freeze, hash_before


def build_prior_snapshot(all_df, season, week):
    key = season * 100 + week
    return all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]


def run():
    results = {}

    wf_mod, freeze, hash_before = load_walkforward()

    # As a naive third entrypoint would: import the simulator's private
    # helper directly. This is the exact bypass vector the mission worries
    # about -- no policy import anywhere in this script.
    from worker.sports_nova_v3.simulator import _primary_qb

    live_df = pd.read_parquet(LIVE_PANEL)
    live_df["_key"] = live_df.SEASON * 100 + live_df.WEEK
    live_df["EVENT_TIME"] = pd.to_datetime(live_df.EVENT_TIME, utc=True)

    game_id = "2026_02_SEA_ARI"
    season, week = 2026, 2
    kickoff = datetime.fromisoformat("2026-09-20T20:25:00+00:00")
    prior = build_prior_snapshot(live_df, season, week)
    state = wf_mod.make_state(game_id, prior, kickoff, None)

    # TEST 1: confirmed allowed
    qb_ari = _primary_qb(state, "ARI")
    results["CONFIRMED_TEST"] = {
        "PASS": qb_ari is not None,
        "DETAIL": f"ARI (CONFIRMED) -> {qb_ari.player_id if qb_ari else None}",
    }

    # TEST 2: uncertain excluded (never a silent guess: None, not a player)
    qb_sea = _primary_qb(state, "SEA")
    results["UNCERTAIN_TEST"] = {
        "PASS": qb_sea is None,
        "DETAIL": f"SEA (UNCERTAIN) -> {qb_sea.player_id if qb_sea else None} (expected None)",
    }

    # TEST 3: stale hard-blocked -- exercise identity_gate directly with an
    # overridden resolution_path pointing at a synthetic stale copy, so the
    # real on-disk artifact is never touched.
    from worker.sports_nova_v3 import identity_gate
    real_resolution = json.loads(RESOLUTION_PATH.read_text())
    stale_path = ROOT / "scripts" / "_regression_test_stale_resolution.json"
    stale = dict(real_resolution)
    stale["AS_OF_UTC"] = str(datetime.now(timezone.utc) - timedelta(hours=48))
    stale_path.write_text(json.dumps(stale))
    try:
        clearance = identity_gate.require_identity_clearance(
            "2026_02_SEA_ARI", "ARI", datetime.now(timezone.utc), resolution_path=stale_path)
        results["STALE_TEST"] = {
            "PASS": clearance["DECISION"] == identity_gate.DECISION_BLOCKED_STALE,
            "DETAIL": clearance,
        }
    finally:
        stale_path.unlink(missing_ok=True)

    # TEST 4: missing identity artifact fails closed
    missing_path = ROOT / "scripts" / "_regression_test_missing_resolution.json"
    if missing_path.exists():
        missing_path.unlink()
    clearance_missing = identity_gate.require_identity_clearance(
        "2026_02_SEA_ARI", "ARI", datetime.now(timezone.utc), resolution_path=missing_path)
    results["MISSING_ARTIFACT_TEST"] = {
        "PASS": clearance_missing["DECISION"] == identity_gate.DECISION_BLOCKED_MISSING,
        "DETAIL": clearance_missing,
    }

    # TEST 5: exclusion hash deterministic -- same inputs, two independent
    # calls, must match. Uses real now (not kickoff -- see __init__.py's
    # _gated_primary_qb for why) since that is what production calls use.
    now_for_gate = datetime.now(timezone.utc)
    clearance_a = identity_gate.require_identity_clearance(game_id, "SEA", now_for_gate)
    clearance_b = identity_gate.require_identity_clearance(game_id, "SEA", now_for_gate)
    results["EXCLUSION_HASH_TEST"] = {
        "PASS": (clearance_a.get("EXCLUSION_HASH") is not None
                  and clearance_a.get("EXCLUSION_HASH") == clearance_b.get("EXCLUSION_HASH")),
        "DETAIL": {"CALL_A": clearance_a.get("EXCLUSION_HASH"), "CALL_B": clearance_b.get("EXCLUSION_HASH")},
    }

    # TEST 6: bypass test -- this whole script IS the synthetic third
    # entrypoint; CONFIRMED_TEST/UNCERTAIN_TEST above already prove a script
    # that never imports the policy module still gets gated correctly.
    results["BYPASS_TEST"] = {
        "PASS": results["CONFIRMED_TEST"]["PASS"] and results["UNCERTAIN_TEST"]["PASS"],
        "DETAIL": "This script never imported sports_nova_v3_v18_qb_uncertainty_policy.py "
                  "or identity_gate.py before calling _primary_qb (identity_gate was only "
                  "imported afterward, for tests 3-5, which need direct access to build a "
                  "synthetic stale case) -- CONFIRMED/UNCERTAIN still resolved correctly.",
    }

    # TEST 7: V18 hashes unchanged -- re-verify every CODE_HASHES-listed file
    # still matches its pre-import value (nothing in this test run touched
    # frozen bytes on disk).
    hash_after = {rel: sha256_file(ROOT / rel) for rel in freeze["CODE_HASHES"]}
    mismatches = {rel: (hash_before[rel], hash_after[rel])
                  for rel in freeze["CODE_HASHES"] if hash_before[rel] != hash_after[rel]}
    frozen_manifest_ok = all(hash_after[rel] == freeze["CODE_HASHES"][rel]["sha256"]
                              for rel in freeze["CODE_HASHES"])
    results["V18_HASH_UNCHANGED_TEST"] = {
        "PASS": not mismatches and frozen_manifest_ok,
        "DETAIL": {"MISMATCHES_DURING_RUN": mismatches, "MATCHES_FREEZE_MANIFEST": frozen_manifest_ok},
    }

    all_pass = all(r["PASS"] for r in results.values())
    print(json.dumps({"ALL_PASS": all_pass, "RESULTS": results}, indent=2, default=str))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run())
