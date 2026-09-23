"""SPORTS_NOVA_V18_2026_CAUSAL_PANEL_REFRESH -- SMOKE phase.

Two jobs, run in strict order, mirroring the mission's SMOKE phase:

1. DIAGNOSTIC ONLY: rebuild pregame features for the already-captured
   2026_02_DET_BUF game under both the frozen (2025-frozen) panel and the
   refreshed (2026-week-1-inclusive) panel, and report the delta. Never
   writes to data/sports_nova_v3/prospective/predictions/ -- that write-once
   directory belongs to the original prospective_capture.py run and stays
   untouched. This block cannot rewrite an existing prospective artifact
   because it never opens PRED_DIR for writing at all.

2. NEW CAPTURE: using the refreshed panel, capture exactly one new eligible
   future game (not DET_BUF -- that identity already has frozen artifacts)
   into a SEPARATE predictions_live/ directory with its own ledger, so the
   original (frozen-panel) prospective cohort and this (refreshed-panel)
   cohort can never be mixed or mistaken for one another. Every artifact
   here carries PANEL_MODE="REFRESHED_2026" and PANEL_SOURCE pointing at the
   refresh receipt; the original capture script's artifacts (implicitly
   PANEL_MODE="FROZEN_2025") are untouched.

Same frozen-code-hash gate as the original script (CODE_HASHES for the
walkforward + simulator files) -- refreshing DATA never licenses silently
running on patched V18 code. The one deliberate difference from the original
script: this one hash-checks the LIVE panel against the refresh receipt's
recorded sha256, not against DATA_SCHEMA_HASHES (which pins the FROZEN
panel and must never be asked to match a file that was built to differ from
it by design).

Temporal firewall for the new capture: panel_max_event_time (max EVENT_TIME
among the refreshed panel's 2026 rows) must be strictly before the target
game's kickoff. Enforced explicitly below and recorded in the artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sports_nova_v3_v18_qb_uncertainty_policy as policy

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3 import identity_gate  # noqa: E402 -- canonical_exclusion_hash
DATA = ROOT / "data" / "sports_nova_v3"
PROSPECTIVE = DATA / "prospective"
CLASSIFICATION_PATH = PROSPECTIVE / "classification" / "SPORTS_NOVA_V18_PROSPECTIVE_CLASSIFICATION.json"
PRED_DIR_FROZEN = PROSPECTIVE / "predictions"
PRED_DIR_LIVE = PROSPECTIVE / "predictions_live"
LEDGER_LIVE_PATH = PROSPECTIVE / "ledger" / "SPORTS_NOVA_V18_PROSPECTIVE_LEDGER_LIVE.json"
DIAGNOSTIC_PATH = PROSPECTIVE / "SPORTS_NOVA_V18_STALE_FEATURE_DELTA_DIAGNOSTIC.json"

FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
SIMULATOR_PATH = ROOT / "worker" / "sports_nova_v3" / "simulator.py"

FROZEN_PANEL = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_RECEIPT = DATA / "validation_inputs_live" / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"

THRESHOLD_GRID = [149.5, 174.5, 199.5, 224.5, 249.5, 274.5, 299.5, 324.5]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def load_frozen_walkforward_module():
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    code_hashes = freeze["CODE_HASHES"]
    checks = {
        "scripts/sports_nova_v3_player_joint_walkforward_v1.py": WALKFORWARD_SCRIPT,
        "worker/sports_nova_v3/simulator.py": SIMULATOR_PATH,
    }
    mismatches = []
    for rel, path in checks.items():
        expected = code_hashes[rel]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(f"{rel}: expected {expected} got {actual}")
    if mismatches:
        raise SystemExit("BLOCKED_FREEZE_HASH_MISMATCH:\n" + "\n".join(mismatches))
    spec = importlib.util.spec_from_file_location(
        "sports_nova_v3_player_joint_walkforward_v1_FROZEN_LIVE", WALKFORWARD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, freeze


def build_prior_snapshot(all_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    key = season * 100 + week
    return all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]


def state_feature_summary(state) -> dict:
    """Flatten the PregameState's numeric Feature values for a diff.

    Keyed on stable identity (player_id / feature name), never on list
    position: `players` is a tuple whose ORDER can differ between two
    PregameStates built from different prior snapshots (a player's rank in
    the window shifts once new games enter it), so positional keys like
    `players[23].features[1]` silently compare two different players across
    runs and inflate the diff count. Team-level `home`/`away` blocks have no
    such reordering risk (fixed two entries) and use their own name.
    """
    out = {}
    payload = state.model_dump()

    def feature_dict(features) -> dict:
        return {f["name"]: f["value"] for f in features}

    for side in ("home", "away"):
        for name, value in feature_dict(payload[side]["features"]).items():
            out[f"{side}.{name}"] = value

    for p in payload.get("players") or ():
        pid = p["player_id"]
        for name, value in feature_dict(p["features"]).items():
            out[f"players.{pid}.{name}"] = value

    return out


def diagnostic_stale_vs_refreshed(wf_mod, freeze, frozen_df, live_df) -> dict:
    game_id = "2026_02_DET_BUF"
    season, week = 2026, 2
    kickoff = datetime.fromisoformat("2026-09-18T00:15:00+00:00")

    frozen_df = frozen_df.copy(); frozen_df["_key"] = frozen_df.SEASON * 100 + frozen_df.WEEK
    live_df = live_df.copy(); live_df["_key"] = live_df.SEASON * 100 + live_df.WEEK

    prior_stale = build_prior_snapshot(frozen_df, season, week)
    prior_refreshed = build_prior_snapshot(live_df, season, week)

    state_stale = wf_mod.make_state(game_id, prior_stale, kickoff, None)
    state_refreshed = wf_mod.make_state(game_id, prior_refreshed, kickoff, None)

    f_stale = state_feature_summary(state_stale)
    f_refreshed = state_feature_summary(state_refreshed)

    deltas = {}
    for k in sorted(set(f_stale) | set(f_refreshed)):
        a, b = f_stale.get(k), f_refreshed.get(k)
        if a != b:
            deltas[k] = {"STALE_2025_FROZEN": a, "REFRESHED_WITH_2026_WK1": b}

    return {
        "GAME_ID": game_id,
        "N_PRIOR_ROWS_STALE": int(len(prior_stale)),
        "N_PRIOR_ROWS_REFRESHED": int(len(prior_refreshed)),
        "N_FEATURES_COMPARED": len(set(f_stale) | set(f_refreshed)),
        "N_FEATURES_CHANGED": len(deltas),
        "SAMPLE_DELTAS": dict(list(deltas.items())[:20]),
        "NOTE": "Diagnostic only; frozen prospective artifacts for this game "
                "were not read, opened, or rewritten by this comparison.",
    }


def parse_kickoff(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def latest_name(prior: pd.DataFrame, player_id: str):
    q = prior[prior.PLAYER_ID == player_id].sort_values(["SEASON", "WEEK"])
    if q.empty:
        return None
    return str(q.iloc[-1]["PLAYER_NAME"])


def capture_new_game_with_live_panel(wf_mod, freeze, live_df, live_sha, live_receipt,
                                      game_row, n_sims, now, identity_resolution,
                                      resolution_sha256):
    from worker.sports_nova_v3.simulator import simulate_game, _primary_qb
    from worker.sports_nova_v3.config import MODEL_VERSION

    game_id = game_row["GAME_ID"]
    season, week = game_row["SEASON"], game_row["WEEK"]
    home, away = game_row["HOME_TEAM"], game_row["AWAY_TEAM"]
    kickoff = parse_kickoff(game_row["KICKOFF_UTC"])
    if now >= kickoff:
        return [], "KICKOFF_HAS_PASSED_SINCE_CLASSIFICATION"

    panel_max_2026_event_time = live_receipt["MAX_2026_EVENT_TIME"]
    if panel_max_2026_event_time is not None:
        panel_max_dt = datetime.fromisoformat(panel_max_2026_event_time.replace("Z", "+00:00"))
        if not (panel_max_dt < kickoff):
            return [], f"TEMPORAL_FIREWALL_VIOLATION: panel_max_event_time {panel_max_2026_event_time} not before kickoff {kickoff.isoformat()}"

    prior = build_prior_snapshot(live_df, season, week)
    state = wf_mod.make_state(game_id, prior, kickoff, None)
    seed = int(hashlib.sha256((game_id + "|LIVE").encode()).hexdigest()[:8], 16)
    sim = simulate_game(state, n_sims, seed, MODEL_VERSION)

    input_payload = state.model_dump()
    input_hash = sha256_obj(input_payload)
    code_hash = {
        "scripts/sports_nova_v3_player_joint_walkforward_v1.py":
            freeze["CODE_HASHES"]["scripts/sports_nova_v3_player_joint_walkforward_v1.py"]["sha256"],
        "worker/sports_nova_v3/simulator.py":
            freeze["CODE_HASHES"]["worker/sports_nova_v3/simulator.py"]["sha256"],
        "scripts/sports_nova_v3_v18_prospective_capture_live.py": sha256_file(Path(__file__).resolve()),
    }

    written = []
    for tid, opp in ((home, away), (away, home)):
        out_path = PRED_DIR_LIVE / f"{game_id}__{tid}.json"
        if out_path.exists():
            continue

        cohort, identity = policy.classify_team(tid, identity_resolution)
        if cohort == "EXCLUDED_UNCERTAIN_IDENTITY":
            marker = {
                "GAME_ID": game_id, "TIMESTAMP": kickoff.isoformat(), "TEAM": tid, "OPPONENT": opp,
                "COHORT": "EXCLUDED_UNCERTAIN_IDENTITY",
                "POLICY": "SPORTS_V18_QB_UNCERTAINTY_POLICY_AND_WEEKLY_REFRESH: "
                          "UNCERTAIN_QB_POLICY=EXCLUDE_FROM_PRIMARY",
                "IDENTITY_RESOLUTION": identity,
                "CREATED_AT": datetime.now(timezone.utc).isoformat(),
                "NOTE": "No point prediction written -- identity UNCERTAIN, not silently "
                        "guessed. Marker file, not a prediction: no EXPECTED_PASS_YARDS.",
            }
            marker["EXCLUSION_HASH"] = identity_gate.canonical_exclusion_hash(
                game_id, tid, identity, resolution_sha256)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(marker, indent=2, default=str))
            written.append((tid, opp, marker))
            continue

        qb_state = _primary_qb(state, tid)
        if qb_state is None:
            continue
        qb_id = qb_state.player_id
        arr = np.asarray(sim.player(qb_id, "pass_yards"), dtype=float)
        dist = {
            "MEAN": float(arr.mean()), "SD": float(arr.std()),
            "P10": float(np.quantile(arr, .10)), "P25": float(np.quantile(arr, .25)),
            "P50": float(np.quantile(arr, .50)), "P75": float(np.quantile(arr, .75)),
            "P90": float(np.quantile(arr, .90)),
        }
        thresholds = {str(t): float(np.mean(arr > t)) for t in THRESHOLD_GRID}
        created_at = datetime.now(timezone.utc)
        artifact = {
            "GAME_ID": game_id,
            "TIMESTAMP": kickoff.isoformat(),
            "MODEL_VERSION": MODEL_VERSION,
            "FEATURE_STORE_VERSION": freeze["FEATURE_STORE_VERSION"],
            "CODE_HASH": code_hash,
            "INPUT_HASH": input_hash,
            "QB": qb_id,
            "QB_NAME": latest_name(prior, qb_id),
            "TEAM": tid,
            "OPPONENT": opp,
            "PREDICTIVE_DISTRIBUTION": dist,
            "EXPECTED_PASS_YARDS": dist["MEAN"],
            "THRESHOLD_PROBABILITIES": thresholds,
            "CREATED_AT": created_at.isoformat(),
            "N_SIMS": n_sims,
            "SIM_COUNT_LABEL": "SMOKE_ONLY",
            "SEED": seed,
            "RNG_ALGORITHM": "PCG64DXSM",
            "CAUSALITY_MODE": "EVENT_CAUSAL_ONLY",
            "PANEL_MODE": "REFRESHED_2026",
            "PLAYER_PANEL_SHA256": live_sha,
            "PLAYER_PANEL_MAX_SEASON": int(live_df.SEASON.max()),
            "PLAYER_PANEL_MAX_2026_EVENT_TIME": panel_max_2026_event_time,
            "PANEL_REFRESH_RECEIPT": str(LIVE_RECEIPT.relative_to(ROOT)),
            "TARGET_GAME_FEATURE_CUTOFF": f"season={season} week<{week} (2026 rows through week "
                                           f"{max((int(w) for w in live_receipt['WEEKS_ELIGIBLE_THIS_RUN']), default=None)} only)",
            "KNOWN_LIMITATION": (
                "recent_form/pass_rate/shares now reflect completed 2026 weeks "
                "through the refresh receipt's WEEKS_ELIGIBLE_THIS_RUN, in "
                "addition to all seasons through 2025; any 2026 week not yet "
                "fully complete at refresh time is still absent, same "
                "data-currency-gap shape as before, just a smaller gap."
            ),
        }
        artifact["PREDICTION_HASH"] = sha256_obj(artifact)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        written.append((tid, opp, artifact))
    return written, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=1000,
                     help="SMOKE_ONLY default; production N_SIMS policy not yet frozen")
    ap.add_argument("--target-game-id", type=str, default=None,
                     help="override which ELIGIBLE future game (other than "
                          "2026_02_DET_BUF) to capture with the refreshed panel")
    args = ap.parse_args()

    identity_gate.assert_prospective_identity_gate_installed()

    wf_mod, freeze = load_frozen_walkforward_module()
    identity_resolution = policy.enforce_fresh_identity()
    resolution_sha256 = sha256_file(policy.RESOLUTION_PATH)

    frozen_df = pd.read_parquet(FROZEN_PANEL)
    live_df = pd.read_parquet(LIVE_PANEL)
    live_sha = sha256_file(LIVE_PANEL)
    live_receipt = json.loads(LIVE_RECEIPT.read_text())
    if live_sha != live_receipt["LIVE_PANEL_SHA256"]:
        raise SystemExit(f"BLOCKED_LIVE_PANEL_HASH_MISMATCH_VS_RECEIPT: expected "
                          f"{live_receipt['LIVE_PANEL_SHA256']} got {live_sha}")

    diag = diagnostic_stale_vs_refreshed(wf_mod, freeze, frozen_df, live_df)
    DIAGNOSTIC_PATH.write_text(json.dumps(diag, indent=2))

    live_df["_key"] = live_df.SEASON * 100 + live_df.WEEK

    classification = json.loads(CLASSIFICATION_PATH.read_text())["GAMES"]
    now = datetime.now(timezone.utc)
    eligible_future = [g for g in classification
                        if g["STATUS"] == "ELIGIBLE"
                        and g["KICKOFF_UTC"] is not None
                        and datetime.fromisoformat(g["KICKOFF_UTC"]) > now
                        and g["GAME_ID"] != "2026_02_DET_BUF"]
    eligible_future.sort(key=lambda g: g["KICKOFF_UTC"])
    if args.target_game_id:
        eligible_future = [g for g in eligible_future if g["GAME_ID"] == args.target_game_id]
    if not eligible_future:
        raise SystemExit("NO_ELIGIBLE_FUTURE_GAME_FOUND_FOR_NEW_CAPTURE")
    target = eligible_future[0]

    ledger = json.loads(LEDGER_LIVE_PATH.read_text()) if LEDGER_LIVE_PATH.exists() else {}
    written, skip_reason = capture_new_game_with_live_panel(
        wf_mod, freeze, live_df, live_sha, live_receipt, target, args.n_sims, now,
        identity_resolution, resolution_sha256)

    new_hashes = {}
    if skip_reason:
        summary = {"CAPTURED": False, "SKIP_REASON": skip_reason, "TARGET": target}
    else:
        for tid, opp, artifact in written:
            key = f"{target['GAME_ID']}__{tid}"
            phash = artifact.get("PREDICTION_HASH") or artifact.get("EXCLUSION_HASH")
            ledger[key] = {
                "GAME_ID": target["GAME_ID"], "TEAM": tid, "OPPONENT": opp,
                "QB": artifact.get("QB"), "COHORT": artifact.get("COHORT", "PRIMARY"),
                "PANEL_MODE": "REFRESHED_2026",
                "IDENTITY_STATUS_AT_PREDICTION_TIME": artifact.get("IDENTITY_RESOLUTION", {}).get("STATUS"),
                "PREDICTION_HASH": phash,
                "PREDICTION_TIME": artifact["CREATED_AT"],
                "KICKOFF_UTC": target["KICKOFF_UTC"],
                "SETTLED": False,
            }
            new_hashes[key] = phash
        LEDGER_LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEDGER_LIVE_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))
        summary = {"CAPTURED": True, "TARGET": target, "NEW_PREDICTION_HASHES": new_hashes}

    summary["STALE_FEATURE_DELTA_DIAGNOSTIC"] = diag
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
