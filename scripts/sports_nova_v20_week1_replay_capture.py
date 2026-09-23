"""SPORTS_NOVA_V20 -- 2026 Week 1 test run (rule frozen after DEV validation).

Mirrors scripts/sports_nova_v19_week1_replay_capture.py exactly (same live
panel, same kickoff cutoffs, same N_SIMS, same 16 games, same seeds, same
CONFIRMED resolutions already on disk from the V19 mission -- no new
resolver run, no re-tuning) except roster construction goes through
make_state_v20 (scripts/sports_nova_v20_player_joint_walkforward_v1.py)
instead of the unmodified V3 make_state, and the simulator is
worker.sports_nova_v19 (UNCHANGED -- V20's only diff surface is roster
construction, not the simulator).

Two-phase run:
  1. FIDELITY CHECK: 2 games, injection disabled ({}), diffed against the
     already-hash-verified frozen V18 replay artifacts -- proves the V20
     harness reproduces V18 byte-for-byte when the one permitted change is
     inert (same discipline V19 applied to itself).
  2. REAL RUN: all 16 games, CONFIRMED resolutions installed for BOTH the
     roster-injection step and V19's own `_qb_shares` override (same
     resolutions object serves both -- there is exactly one identity
     authority, not two).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "sports_nova_v3"
FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_RECEIPT = DATA / "validation_inputs_live" / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"
SCHEDULE_CSV = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"
V19_RESOLUTION_PATH = DATA / "prospective" / "identity" / "SPORTS_NOVA_V19_WEEK1_QB_IDENTITY_RESOLUTION.json"

V18_REPLAY_DIR = DATA / "replay" / "week1_2026"
V18_PRED_DIR = V18_REPLAY_DIR / "predictions"

V19_REPLAY_DIR = DATA / "replay_v19" / "week1_2026"
V19_PRED_DIR = V19_REPLAY_DIR / "predictions"

REPLAY_DIR = DATA / "replay_v20" / "week1_2026"
PRED_DIR = REPLAY_DIR / "predictions"
DRAWS_DIR = REPLAY_DIR / "draws"
FIDELITY_GAME_IDS = ("2026_01_DEN_KC", "2026_01_ARI_LAC")
SEASON = 2026
WEEK = 1
N_SIMS = 2000


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


def verify_freeze_and_load():
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    mismatches = []
    for rel, meta in freeze["CODE_HASHES"].items():
        actual = sha256_file(ROOT / rel)
        if actual != meta["sha256"]:
            mismatches.append(f"{rel}: expected {meta['sha256']} got {actual}")
    if mismatches:
        raise SystemExit("BLOCKED_FREEZE_HASH_MISMATCH (V18 must stay untouched):\n" + "\n".join(mismatches))

    live_receipt = json.loads(LIVE_RECEIPT.read_text())
    live_actual = sha256_file(LIVE_PANEL)
    if live_actual != live_receipt["LIVE_PANEL_SHA256"]:
        raise SystemExit(
            f"BLOCKED_LIVE_PANEL_HASH_MISMATCH: expected {live_receipt['LIVE_PANEL_SHA256']} got {live_actual}")

    spec = importlib.util.spec_from_file_location(
        "sports_nova_v20_player_joint_walkforward_v1_REPLAY",
        ROOT / "scripts" / "sports_nova_v20_player_joint_walkforward_v1.py")
    v20wf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v20wf)
    return v20wf, freeze


def parse_kickoff_utc(gameday: str, gametime) -> datetime | None:
    if pd.isna(gametime):
        return None
    d = datetime.strptime(str(gameday), "%Y-%m-%d")
    hh, mm = str(gametime).split(":")
    local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def build_resolution_lookup():
    payload = json.loads(V19_RESOLUTION_PATH.read_text())
    lookup = {}
    for key, rec in payload["RESOLUTIONS"].items():
        game_id, team = key.split("__")
        lookup[(game_id, team)] = rec
    return lookup


def run_one_game(v20wf, v19_sim, panel, r, resolutions):
    away, home = r.away_team, r.home_team
    game_id = f"{SEASON}_{WEEK:02d}_{away}_{home}"
    kickoff = parse_kickoff_utc(r.gameday, r.gametime)
    if kickoff is None:
        return game_id, None, "INVALID_FOR_REPLAY: no gametime"

    prior = panel[(panel._event_time < kickoff) & (panel.SEASON >= SEASON - 5)].copy()
    firewall_ok = prior.empty or prior._event_time.max() < kickoff
    if not firewall_ok:
        return game_id, None, "INVALID_FOR_REPLAY: temporal firewall assertion failed"
    prior_for_state = prior.drop(columns=["_event_time"])
    prior_sha256 = sha256_obj(prior_for_state.to_dict(orient="list"))

    state = v20wf.make_state_v20(game_id, prior_for_state, kickoff, None, resolutions)
    seed = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)

    v19_sim.set_identity_resolutions(resolutions)
    sim = v19_sim.simulate_game(state, N_SIMS, seed, v19_sim.MODEL_VERSION)

    home_qb = v19_sim._primary_qb(state, home)
    away_qb = v19_sim._primary_qb(state, away)

    def qb_block(qb):
        if qb is None:
            return None
        att = sim.player(qb.player_id, "pass_attempts")
        yds = sim.player(qb.player_id, "pass_yards")
        tds = sim.player(qb.player_id, "pass_tds")
        return {
            "PLAYER_ID": qb.player_id,
            "PASS_ATTEMPTS": {"mean": float(att.mean()), "sd": float(att.std()),
                               "p10": float(np.quantile(att, .1)), "p50": float(np.quantile(att, .5)),
                               "p90": float(np.quantile(att, .9))},
            "PASS_YARDS": {"mean": float(yds.mean()), "sd": float(yds.std()),
                            "p10": float(np.quantile(yds, .1)), "p50": float(np.quantile(yds, .5)),
                            "p90": float(np.quantile(yds, .9))},
            "PASS_TDS": {"mean": float(tds.mean())},
        }

    home_score = sim.team(home, "score")
    away_score = sim.team(away, "score")
    draws = {"home_score": home_score, "away_score": away_score, "winner": sim.winner.astype("U4")}
    if home_qb is not None:
        draws["home_qb_pass_attempts"] = sim.player(home_qb.player_id, "pass_attempts")
        draws["home_qb_pass_yards"] = sim.player(home_qb.player_id, "pass_yards")
    if away_qb is not None:
        draws["away_qb_pass_attempts"] = sim.player(away_qb.player_id, "pass_attempts")
        draws["away_qb_pass_yards"] = sim.player(away_qb.player_id, "pass_yards")

    win_prob_home = float(np.mean(sim.winner == "HOME"))
    win_prob_away = float(np.mean(sim.winner == "AWAY"))
    win_prob_tie = float(np.mean(sim.winner == "TIE"))
    model_spread_home = float(np.mean(home_score - away_score))
    model_total = float(np.mean(home_score + away_score))

    v18_pred_path = V18_PRED_DIR / f"{game_id}.json"
    v18_pred = json.loads(v18_pred_path.read_text()) if v18_pred_path.exists() else None
    v19_pred_path = V19_PRED_DIR / f"{game_id}.json"
    v19_pred = json.loads(v19_pred_path.read_text()) if v19_pred_path.exists() else None

    def identity_info(team, qb):
        rec = resolutions.get((game_id, team))
        v18_qb_id = (v18_pred.get("HOME_QB" if team == home else "AWAY_QB") or {}).get("PLAYER_ID") if v18_pred else None
        v19_qb_id = (v19_pred.get("HOME_QB" if team == home else "AWAY_QB") or {}).get("PLAYER_ID") if v19_pred else None
        return {
            "RESOLUTION": rec,
            "V20_PICKED_QB_ID": qb.player_id if qb is not None else None,
            "V19_PICKED_QB_ID": v19_qb_id,
            "V18_PICKED_QB_ID": v18_qb_id,
            "V20_CHANGED_PICK_VS_V19": bool(
                qb is not None and v19_qb_id is not None and qb.player_id != v19_qb_id),
        }

    market = {
        "SOURCE": "nflverse games.csv (schedule_snapshot_2026.csv) -- aggregated line field, not attributed to a named sportsbook",
        "SPREAD_LINE": None if pd.isna(r.spread_line) else float(r.spread_line),
        "TOTAL_LINE": None if pd.isna(r.total_line) else float(r.total_line),
        "HOME_MONEYLINE": None if pd.isna(r.home_moneyline) else float(r.home_moneyline),
        "AWAY_MONEYLINE": None if pd.isna(r.away_moneyline) else float(r.away_moneyline),
    }

    artifact = {
        "SCHEMA": "SPORTS_NOVA_V20_WEEK1_REPLAY_PREDICTION",
        "GAME_ID": game_id, "SEASON": SEASON, "WEEK": WEEK,
        "AWAY_TEAM": away, "HOME_TEAM": home,
        "KICKOFF_UTC": kickoff.isoformat(),
        "MODEL_VERSION": v19_sim.MODEL_VERSION + ".v20_roster_injection",
        "N_SIMS": N_SIMS, "SEED": seed,
        "STATE_HASH": sim.state_hash,
        "PRIOR_ROW_COUNT": int(len(prior_for_state)),
        "PRIOR_SHA256": prior_sha256,
        "MODEL_SCORE_DIST": {
            "HOME": {"mean": float(home_score.mean()), "sd": float(home_score.std()),
                      "p10": float(np.quantile(home_score, .1)), "p50": float(np.quantile(home_score, .5)),
                      "p90": float(np.quantile(home_score, .9))},
            "AWAY": {"mean": float(away_score.mean()), "sd": float(away_score.std()),
                      "p10": float(np.quantile(away_score, .1)), "p50": float(np.quantile(away_score, .5)),
                      "p90": float(np.quantile(away_score, .9))},
        },
        "MODEL_WIN_PROB": {"HOME": win_prob_home, "AWAY": win_prob_away, "TIE": win_prob_tie},
        "MODEL_SPREAD_HOME": model_spread_home,
        "MODEL_TOTAL": model_total,
        "HOME_QB": qb_block(home_qb),
        "AWAY_QB": qb_block(away_qb),
        "HOME_QB_IDENTITY": identity_info(home, home_qb),
        "AWAY_QB_IDENTITY": identity_info(away, away_qb),
        "MARKET": market,
        "PREDICTION_TIME_UTC": datetime.now(timezone.utc).isoformat(),
    }
    return game_id, (artifact, draws), None


def main():
    v20wf, freeze = verify_freeze_and_load()
    from worker.sports_nova_v19 import simulator as v19_sim

    sched = pd.read_csv(SCHEDULE_CSV)
    week1 = sched[(sched.season == SEASON) & (sched.week == WEEK) & (sched.game_type == "REG")].copy()
    if len(week1) != 16:
        raise SystemExit(f"EXPECTED_16_WEEK1_GAMES_GOT_{len(week1)}")

    panel = pd.read_parquet(LIVE_PANEL)
    panel["_event_time"] = pd.to_datetime(panel["EVENT_TIME"], utc=True, errors="coerce")

    resolutions = build_resolution_lookup()
    confirmed_resolutions = {k: v for k, v in resolutions.items() if v.get("STATUS") == "CONFIRMED"}

    # ---- PHASE 1: fidelity check (injection + override disabled) ----
    fidelity_log = []
    for r in week1.itertuples():
        game_id = f"{SEASON}_{WEEK:02d}_{r.away_team}_{r.home_team}"
        if game_id not in FIDELITY_GAME_IDS:
            continue
        gid, result, err = run_one_game(v20wf, v19_sim, panel, r, {})
        if err:
            fidelity_log.append({"GAME_ID": gid, "STATUS": "ERROR", "REASON": err})
            continue
        artifact, draws = result
        v18_pred = json.loads((V18_PRED_DIR / f"{gid}.json").read_text())
        v18_draws_path = ROOT / v18_pred["RAW_DRAWS_PATH"]
        v18_draws_npz = np.load(v18_draws_path)
        arrays_match = all(
            k in v18_draws_npz.files and np.array_equal(draws[k], v18_draws_npz[k]) for k in draws
        ) and set(draws.keys()) == set(v18_draws_npz.files)
        fidelity_log.append({
            "GAME_ID": gid, "STATUS": "CHECKED",
            "STATE_HASH_MATCH": artifact["STATE_HASH"] == v18_pred["STATE_HASH"],
            "MODEL_SCORE_DIST_MATCH": artifact["MODEL_SCORE_DIST"] == v18_pred["MODEL_SCORE_DIST"],
            "RAW_DRAWS_ARRAYS_MATCH": bool(arrays_match),
        })

    fidelity_all_pass = (
        len(fidelity_log) == len(FIDELITY_GAME_IDS)
        and all(f["STATUS"] == "CHECKED" for f in fidelity_log)
        and all(f["STATE_HASH_MATCH"] and f["MODEL_SCORE_DIST_MATCH"] and f["RAW_DRAWS_ARRAYS_MATCH"] for f in fidelity_log)
    )
    print(json.dumps({"FIDELITY_CHECK": fidelity_log, "FIDELITY_ALL_PASS": fidelity_all_pass}, indent=2))
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    (REPLAY_DIR / "SPORTS_NOVA_V20_FIDELITY_CHECK.json").write_text(json.dumps(
        {"FIDELITY_CHECK": fidelity_log, "FIDELITY_ALL_PASS": fidelity_all_pass,
         "CHECKED_AT_UTC": datetime.now(timezone.utc).isoformat()}, indent=2))
    if not fidelity_all_pass:
        raise SystemExit("BLOCKED: V20 harness does not reproduce frozen V18 byte-for-byte with injection/override disabled.")

    # ---- PHASE 2: real run ----
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    capture_log = []
    for r in week1.itertuples():
        gid, result, err = run_one_game(v20wf, v19_sim, panel, r, confirmed_resolutions)
        if err:
            capture_log.append({"GAME_ID": gid, "STATUS": "INVALID_FOR_REPLAY", "REASON": err})
            continue
        artifact, draws = result
        draws_path = DRAWS_DIR / f"{gid}.npz"
        np.savez_compressed(draws_path, **draws)
        artifact["RAW_DRAWS_PATH"] = str(draws_path.relative_to(ROOT))
        artifact["RAW_DRAWS_SHA256"] = sha256_file(draws_path)
        artifact["PREDICTION_HASH"] = sha256_obj({k: v for k, v in artifact.items() if k != "PREDICTION_HASH"})
        (PRED_DIR / f"{gid}.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
        capture_log.append({"GAME_ID": gid, "STATUS": "CAPTURED", "PREDICTION_HASH": artifact["PREDICTION_HASH"]})
        print(f"CAPTURED {gid}", flush=True)

    (REPLAY_DIR / "SPORTS_NOVA_V20_WEEK1_CAPTURE_LOG.json").write_text(json.dumps(capture_log, indent=2))
    print(json.dumps({
        "CAPTURED": sum(1 for c in capture_log if c["STATUS"] == "CAPTURED"),
        "INVALID": sum(1 for c in capture_log if c["STATUS"] == "INVALID_FOR_REPLAY"),
        "TOTAL": len(capture_log), "FIDELITY_ALL_PASS": fidelity_all_pass,
        "N_CONFIRMED_RESOLUTIONS_AVAILABLE": len(confirmed_resolutions),
    }, indent=2))


if __name__ == "__main__":
    main()
