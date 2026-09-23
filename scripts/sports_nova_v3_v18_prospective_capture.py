"""SPORTS_NOVA_V18_PROSPECTIVE_CAPTURE_GATE.

Builds a genuinely prospective (never-touched, pre-outcome) validation stream
for the frozen V18 engine. Two responsibilities, kept in this one script
because they share the same classification pass, but strictly ordered so
capture can never see an outcome:

  1. CLASSIFY every scheduled game after PROSPECTIVE_START as ELIGIBLE or
     EXCLUDED_WITH_REASON (item 3 of the mission).
  2. CAPTURE an immutable, hashed, write-once prediction artifact for each
     ELIGIBLE game whose kickoff is still strictly in the future relative to
     wall-clock "now" at run time (items 4-5).

Outcome joining is a SEPARATE script
(sports_nova_v3_v18_prospective_outcome_join.py) and is never imported or
called from here, by design -- this script must not be able to see a result.

This script does not modify anything under worker/sports_nova_v3/ or the V18
frozen walkforward script. It imports the frozen walkforward module by exact
file path and re-verifies its sha256 (and the simulator's) against
SPORTS_NOVA_V18_FREEZE_MANIFEST.json before using either, and refuses to run
(BLOCKED_FREEZE_HASH_MISMATCH) if either has drifted -- silently predicting
off a patched engine under the V18 label is exactly what the mission's item
11 forbids.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
PROSPECTIVE = DATA / "prospective"
SCHEDULE_DIR = PROSPECTIVE / "schedule_snapshots"
PRED_DIR = PROSPECTIVE / "predictions"
LEDGER_DIR = PROSPECTIVE / "ledger"
CLASSIFICATION_DIR = PROSPECTIVE / "classification"
CAPTURE_MANIFEST = PROSPECTIVE / "SPORTS_NOVA_V18_PROSPECTIVE_CAPTURE_MANIFEST.json"
LEDGER_PATH = LEDGER_DIR / "SPORTS_NOVA_V18_PROSPECTIVE_LEDGER.json"
CLASSIFICATION_PATH = CLASSIFICATION_DIR / "SPORTS_NOVA_V18_PROSPECTIVE_CLASSIFICATION.json"

FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
SIMULATOR_PATH = ROOT / "worker" / "sports_nova_v3" / "simulator.py"
PLAYER_PANEL = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
KICKOFF_TZ_ASSUMPTION = "America/New_York (ASSUMED; nflverse `gametime` is not explicitly tz-labeled)"

# Generic, round-number reporting grid -- not sourced from or tuned to any
# market line (sportsbook data stays banned from model construction per
# project convention; this is only for later comparison, per mission item 10).
THRESHOLD_GRID = [149.5, 174.5, 199.5, 224.5, 249.5, 274.5, 299.5, 324.5]

THIS_SCRIPT = Path(__file__).resolve()


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
    """Import the frozen V18 feature-store script by path and hash-verify it."""
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
        "sports_nova_v3_player_joint_walkforward_v1_FROZEN", WALKFORWARD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, freeze


def fetch_schedule_snapshot(offline_path: str | None) -> tuple[pd.DataFrame, Path, str]:
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    if offline_path:
        raw = Path(offline_path).read_bytes()
    else:
        with urllib.request.urlopen(SCHEDULE_URL, timeout=30) as resp:
            raw = resp.read()
    fetched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = SCHEDULE_DIR / f"SPORTS_NOVA_V18_SCHEDULE_SNAPSHOT_{fetched_at}.csv"
    snap_path.write_bytes(raw)
    df = pd.read_csv(snap_path)
    return df, snap_path, hashlib.sha256(raw).hexdigest()


def parse_kickoff_utc(gameday: str, gametime) -> datetime | None:
    if pd.isna(gametime):
        return None
    d = datetime.strptime(gameday, "%Y-%m-%d")
    hh, mm = str(gametime).split(":")
    local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def classify_schedule(sched: pd.DataFrame, prospective_start: datetime, season: int):
    rows = []
    s = sched[sched.season == season].copy()
    for r in s.itertuples():
        kickoff = parse_kickoff_utc(r.gameday, r.gametime)
        played = pd.notna(r.home_score)
        reason = None
        status = "ELIGIBLE"
        if played:
            status, reason = "EXCLUDED", "ALREADY_PLAYED"
        elif r.game_type != "REG":
            status, reason = "EXCLUDED", f"GAME_TYPE_NOT_IN_SCOPE:{r.game_type}"
        elif kickoff is None:
            status, reason = "EXCLUDED", "KICKOFF_TIME_NOT_YET_PUBLISHED"
        elif kickoff <= prospective_start:
            status, reason = "EXCLUDED", "KICKOFF_NOT_AFTER_PROSPECTIVE_START"
        rows.append({
            "GAME_ID": r.game_id, "SEASON": int(r.season), "WEEK": int(r.week),
            "GAME_TYPE": r.game_type, "HOME_TEAM": r.home_team, "AWAY_TEAM": r.away_team,
            "KICKOFF_UTC": kickoff.isoformat() if kickoff else None,
            "STATUS": status, "REASON": reason,
        })
    return rows


def build_prior_snapshot(all_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    key = season * 100 + week
    return all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]


def latest_name(prior: pd.DataFrame, player_id: str) -> str | None:
    q = prior[prior.PLAYER_ID == player_id].sort_values(["SEASON", "WEEK"])
    if q.empty:
        return None
    return str(q.iloc[-1]["PLAYER_NAME"])


def capture_one_game(wf_mod, freeze, all_df, panel_sha, game_row, run_id, n_sims, now):
    game_id = game_row["GAME_ID"]
    season, week = game_row["SEASON"], game_row["WEEK"]
    home, away = game_row["HOME_TEAM"], game_row["AWAY_TEAM"]
    kickoff = datetime.fromisoformat(game_row["KICKOFF_UTC"])
    if now >= kickoff:
        return [], "KICKOFF_HAS_PASSED_SINCE_CLASSIFICATION"

    from worker.sports_nova_v3.simulator import simulate_game, _primary_qb
    from worker.sports_nova_v3.config import MODEL_VERSION

    prior = build_prior_snapshot(all_df, season, week)
    state = wf_mod.make_state(game_id, prior, kickoff, None)
    seed = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
    sim = simulate_game(state, n_sims, seed, MODEL_VERSION)

    input_payload = state.model_dump()
    input_hash = sha256_obj(input_payload)
    code_hash = {
        "scripts/sports_nova_v3_player_joint_walkforward_v1.py":
            freeze["CODE_HASHES"]["scripts/sports_nova_v3_player_joint_walkforward_v1.py"]["sha256"],
        "worker/sports_nova_v3/simulator.py":
            freeze["CODE_HASHES"]["worker/sports_nova_v3/simulator.py"]["sha256"],
        "scripts/sports_nova_v3_v18_prospective_capture.py": sha256_file(THIS_SCRIPT),
    }

    written = []
    for tid, opp in ((home, away), (away, home)):
        qb_state = _primary_qb(state, tid)
        if qb_state is None:
            continue
        qb_id = qb_state.player_id
        out_path = PRED_DIR / f"{game_id}__{tid}.json"
        if out_path.exists():
            continue  # write-once guard
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
            "SEED": seed,
            "RNG_ALGORITHM": "PCG64DXSM",
            "CAUSALITY_MODE": "EVENT_CAUSAL_ONLY",
            "PLAYER_PANEL_SHA256": panel_sha,
            "PLAYER_PANEL_MAX_SEASON": int(all_df.SEASON.max()),
            "KNOWN_LIMITATION": (
                "Prior-window features are frozen at the end of the panel's most "
                "recent ingested season; any 2026-season games not yet ingested into "
                "the causal panel are NOT reflected in recent_form/pass_rate/shares "
                "for this prediction, even if they occurred before this game's "
                "kickoff. This is a data-currency gap, not lookahead: no "
                "information from AFTER this prediction's CREATED_AT was used."
            ),
        }
        artifact["PREDICTION_HASH"] = sha256_obj(artifact)
        out_path.write_text(json.dumps(artifact, indent=2))
        written.append((tid, opp, artifact))
    return written, None


def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {}


def save_ledger(ledger: dict):
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--n-sims", type=int, default=None,
                     help="default: worker.sports_nova_v3.config.TARGET_N_SIMS")
    ap.add_argument("--max-capture", type=int, default=2,
                     help="max team-perspective artifacts to write this run (smoke-test bound)")
    ap.add_argument("--offline-schedule", type=str, default=None,
                     help="path to a pre-fetched games.csv, for offline/test runs")
    args = ap.parse_args()

    from worker.sports_nova_v3 import identity_gate
    identity_gate.assert_prospective_identity_gate_installed()

    wf_mod, freeze = load_frozen_walkforward_module()
    from worker.sports_nova_v3.config import TARGET_N_SIMS
    n_sims = args.n_sims or TARGET_N_SIMS

    panel_sha = sha256_file(PLAYER_PANEL)
    expected_panel_sha = freeze["DATA_SCHEMA_HASHES"]["NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"]
    if panel_sha != expected_panel_sha:
        raise SystemExit(f"BLOCKED_PANEL_HASH_MISMATCH: expected {expected_panel_sha} got {panel_sha}")

    prospective_start = datetime.fromisoformat(freeze["TIMESTAMP_UTC"].replace("Z", "+00:00"))

    sched, snap_path, snap_sha = fetch_schedule_snapshot(args.offline_schedule)
    now = datetime.now(timezone.utc)

    classification = classify_schedule(sched, prospective_start, args.season)
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_PATH.write_text(json.dumps({
        "PROSPECTIVE_START_UTC": prospective_start.isoformat(),
        "CLASSIFIED_AT_UTC": now.isoformat(),
        "SCHEDULE_SNAPSHOT": str(snap_path.relative_to(ROOT)),
        "SCHEDULE_SNAPSHOT_SHA256": snap_sha,
        "KICKOFF_TZ_ASSUMPTION": KICKOFF_TZ_ASSUMPTION,
        "GAMES": classification,
    }, indent=2))

    all_df = pd.read_parquet(PLAYER_PANEL)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK

    eligible_future = [g for g in classification
                       if g["STATUS"] == "ELIGIBLE"
                       and g["KICKOFF_UTC"] is not None
                       and datetime.fromisoformat(g["KICKOFF_UTC"]) > now]
    eligible_future.sort(key=lambda g: g["KICKOFF_UTC"])

    ledger = load_ledger()
    captured_games = []
    remaining_budget = args.max_capture
    for g in eligible_future:
        if remaining_budget <= 0:
            break
        already = f"{g['GAME_ID']}__{g['HOME_TEAM']}" in ledger and f"{g['GAME_ID']}__{g['AWAY_TEAM']}" in ledger
        if already:
            continue
        written, skip_reason = capture_one_game(wf_mod, freeze, all_df, panel_sha, g, args.season, n_sims, now)
        if skip_reason:
            continue
        for tid, opp, artifact in written:
            ledger[f"{g['GAME_ID']}__{tid}"] = {
                "GAME_ID": g["GAME_ID"], "TEAM": tid, "OPPONENT": opp,
                "QB": artifact["QB"],
                "PREDICTION_HASH": artifact["PREDICTION_HASH"],
                "PREDICTION_TIME": artifact["CREATED_AT"],
                "KICKOFF_UTC": g["KICKOFF_UTC"],
                "OUTCOME_TIME": None, "PASS_YARDS_ACTUAL": None,
                "ATTEMPT_ACTUAL": None, "COMPLETION_ACTUAL": None,
                "MODEL_METRICS": None, "MARKET_PRICES_IF_CAUSALLY_CAPTURED": None,
                "SETTLED": False,
            }
            remaining_budget -= 1
        if written:
            captured_games.append(g["GAME_ID"])
        if remaining_budget <= 0:
            break
    save_ledger(ledger)

    n_eligible_total = sum(1 for g in classification if g["STATUS"] == "ELIGIBLE")
    n_excluded = sum(1 for g in classification if g["STATUS"] == "EXCLUDED")
    summary = {
        "PROSPECTIVE_START_UTC": prospective_start.isoformat(),
        "SEASON": args.season,
        "N_SCHEDULED_GAMES": len(classification),
        "N_ELIGIBLE": n_eligible_total,
        "N_EXCLUDED": n_excluded,
        "N_ELIGIBLE_AND_FUTURE_AT_RUN_TIME": len(eligible_future),
        "N_TEAM_GAME_PREDICTIONS_IN_LEDGER": len(ledger),
        "CAPTURED_THIS_RUN": captured_games,
        "CLASSIFICATION_FILE": str(CLASSIFICATION_PATH.relative_to(ROOT)),
        "LEDGER_FILE": str(LEDGER_PATH.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
