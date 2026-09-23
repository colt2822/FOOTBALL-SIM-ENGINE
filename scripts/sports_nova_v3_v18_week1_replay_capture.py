"""SPORTS_NOVA_V18_WEEK1_HISTORICAL_REPLAY -- capture phase (predictions only).

Builds a leakage-free, event-causal pregame snapshot for every 2026 Week 1
REG-season game and runs the FROZEN V18 engine exactly once per game,
freezing the raw simulation output to disk BEFORE any outcome is read.

Never imports the outcome-join script. Never reads home_score/away_score,
the "result" column, or any player-panel row whose EVENT_TIME >= that game's
own kickoff. This is the entire integrity claim of this script; the join
step is a separate process (sports_nova_v3_v18_week1_replay_join.py) that
can only append outcome fields, never regenerate a prediction.

Causal construction, per game:
  - `prior` = live causal panel rows with EVENT_TIME strictly before this
    game's own kickoff, SEASON >= season-5 (same lookback window convention
    as the frozen walkforward script's own `prior_snapshot`, generalized
    from week-granularity to real per-game timestamps so a Monday game
    legitimately sees the same week's earlier, already-final games while a
    Thursday opener sees none of them).
  - QB team assignment inside make_state() comes from each player's own
    last prior-game TEAM in `prior` (the frozen script's `latest_team`
    logic) -- NOT from any depth chart or roster file. This is an inherent
    property of the hash-pinned make_state()/team_state() functions this
    script calls unmodified; an offseason trade that has not yet produced a
    2026 game for the new team is invisible to it. That is a known,
    honestly-disclosed V18 property, not something this replay can or
    should patch (mission: DO NOT RETUNE V18).
  - The QB identity-uncertainty gate (worker/sports_nova_v3/__init__.py)
    self-detects kickoff <= wall-clock-now and defers to the engine's own
    unwrapped _primary_qb for every Week 1 game (they already happened) --
    verified by reading that file; not re-implemented here.

Market comparison: schedule_snapshot_2026.csv (nflverse games.csv) carries
spread_line/total_line/moneyline/over_under columns for this slate already.
Recorded as MARKET_SOURCE="nflverse games.csv (aggregated line field, not a
named sportsbook)" -- never fabricated, never fetched post-hoc and
mislabeled as pregame.
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
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
SIMULATOR_PATH = ROOT / "worker" / "sports_nova_v3" / "simulator.py"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_RECEIPT = DATA / "validation_inputs_live" / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"
SCHEDULE_CSV = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"

REPLAY_DIR = DATA / "replay" / "week1_2026"
PRED_DIR = REPLAY_DIR / "predictions"
DRAWS_DIR = REPLAY_DIR / "draws"
CANARY_GAME_AWAY, CANARY_GAME_HOME = "DEN", "KC"
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
    code_hashes = freeze["CODE_HASHES"]
    mismatches = []
    for rel, meta in code_hashes.items():
        actual = sha256_file(ROOT / rel)
        if actual != meta["sha256"]:
            mismatches.append(f"{rel}: expected {meta['sha256']} got {actual}")
    if mismatches:
        raise SystemExit("BLOCKED_FREEZE_HASH_MISMATCH:\n" + "\n".join(mismatches))

    live_receipt = json.loads(LIVE_RECEIPT.read_text())
    live_actual = sha256_file(LIVE_PANEL)
    if live_actual != live_receipt["LIVE_PANEL_SHA256"]:
        raise SystemExit(
            f"BLOCKED_LIVE_PANEL_HASH_MISMATCH: expected {live_receipt['LIVE_PANEL_SHA256']} "
            f"got {live_actual}")

    spec = importlib.util.spec_from_file_location(
        "sports_nova_v3_player_joint_walkforward_v1_FROZEN_REPLAY", WALKFORWARD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, freeze, live_receipt


def parse_kickoff_utc(gameday: str, gametime) -> datetime | None:
    if pd.isna(gametime):
        return None
    d = datetime.strptime(str(gameday), "%Y-%m-%d")
    hh, mm = str(gametime).split(":")
    local = d.replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def main():
    walkforward_mod, freeze, live_receipt = verify_freeze_and_load()
    from worker.sports_nova_v3 import simulator as canonical_simulator

    sched = pd.read_csv(SCHEDULE_CSV)
    week1 = sched[(sched.season == SEASON) & (sched.week == WEEK) & (sched.game_type == "REG")].copy()
    if len(week1) != 16:
        raise SystemExit(f"EXPECTED_16_WEEK1_GAMES_GOT_{len(week1)}")

    panel = pd.read_parquet(LIVE_PANEL)
    panel["_event_time"] = pd.to_datetime(panel["EVENT_TIME"], utc=True, errors="coerce")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)

    capture_log = []
    for r in week1.itertuples():
        away, home = r.away_team, r.home_team
        game_id = f"{SEASON}_{WEEK:02d}_{away}_{home}"
        kickoff = parse_kickoff_utc(r.gameday, r.gametime)
        if kickoff is None:
            capture_log.append({"GAME_ID": game_id, "STATUS": "INVALID_FOR_REPLAY",
                                 "REASON": "no gametime in schedule snapshot"})
            continue

        prior = panel[(panel._event_time < kickoff) & (panel.SEASON >= SEASON - 5)].copy()
        firewall_ok = prior.empty or prior._event_time.max() < kickoff
        if not firewall_ok:
            capture_log.append({"GAME_ID": game_id, "STATUS": "INVALID_FOR_REPLAY",
                                 "REASON": "temporal firewall assertion failed"})
            continue
        prior_for_state = prior.drop(columns=["_event_time"])
        prior_sha256 = sha256_obj(prior_for_state.to_dict(orient="list"))

        try:
            state = walkforward_mod.make_state(game_id, prior_for_state, kickoff, None)
        except Exception as exc:  # noqa: BLE001
            capture_log.append({"GAME_ID": game_id, "STATUS": "INVALID_FOR_REPLAY",
                                 "REASON": f"make_state failed: {exc!r}"})
            continue

        seed = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
        sim = walkforward_mod.simulate_game(state, N_SIMS, seed, walkforward_mod.MODEL_VERSION)

        home_qb = canonical_simulator._primary_qb(state, home)
        away_qb = canonical_simulator._primary_qb(state, away)

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
                "COMPLETIONS": "NOT_MODELED_BY_V18_ENGINE -- no completions stat exists in "
                                "simulator.STAT_NAMES; not fabricated here.",
            }

        home_score = sim.team(home, "score")
        away_score = sim.team(away, "score")

        draws = {"home_score": home_score, "away_score": away_score,
                 "winner": sim.winner.astype("U4")}
        if home_qb is not None:
            draws["home_qb_pass_attempts"] = sim.player(home_qb.player_id, "pass_attempts")
            draws["home_qb_pass_yards"] = sim.player(home_qb.player_id, "pass_yards")
        if away_qb is not None:
            draws["away_qb_pass_attempts"] = sim.player(away_qb.player_id, "pass_attempts")
            draws["away_qb_pass_yards"] = sim.player(away_qb.player_id, "pass_yards")
        draws_path = DRAWS_DIR / f"{game_id}.npz"
        np.savez_compressed(draws_path, **draws)
        draws_sha256 = sha256_file(draws_path)

        win_prob_home = float(np.mean(sim.winner == "HOME"))
        win_prob_away = float(np.mean(sim.winner == "AWAY"))
        win_prob_tie = float(np.mean(sim.winner == "TIE"))
        model_spread_home = float(np.mean(home_score - away_score))  # positive = home favored
        model_total = float(np.mean(home_score + away_score))

        market = {
            "SOURCE": "nflverse games.csv (schedule_snapshot_2026.csv) -- aggregated line "
                       "field, not attributed to a named sportsbook",
            "SPREAD_LINE": None if pd.isna(r.spread_line) else float(r.spread_line),
            "TOTAL_LINE": None if pd.isna(r.total_line) else float(r.total_line),
            "HOME_MONEYLINE": None if pd.isna(r.home_moneyline) else float(r.home_moneyline),
            "AWAY_MONEYLINE": None if pd.isna(r.away_moneyline) else float(r.away_moneyline),
        }

        artifact = {
            "SCHEMA": "SPORTS_NOVA_V18_WEEK1_REPLAY_PREDICTION",
            "GAME_ID": game_id,
            "SEASON": SEASON, "WEEK": WEEK,
            "AWAY_TEAM": away, "HOME_TEAM": home,
            "KICKOFF_UTC": kickoff.isoformat(),
            "MODEL_VERSION": walkforward_mod.MODEL_VERSION,
            "N_SIMS": N_SIMS,
            "SEED": seed,
            "STATE_HASH": sim.state_hash,
            "PRIOR_ROW_COUNT": int(len(prior_for_state)),
            "PRIOR_SHA256": prior_sha256,
            "PRIOR_MAX_EVENT_TIME": None if prior.empty else prior._event_time.max().isoformat(),
            "TEMPORAL_FIREWALL_PASS": bool(firewall_ok),
            "LIVE_PANEL_SHA256": live_receipt["LIVE_PANEL_SHA256"],
            "FREEZE_MANIFEST_TIMESTAMP": freeze["TIMESTAMP_UTC"],
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
            "MARKET": market,
            "RAW_DRAWS_PATH": str(draws_path.relative_to(ROOT)),
            "RAW_DRAWS_SHA256": draws_sha256,
            "PREDICTION_TIME_UTC": datetime.now(timezone.utc).isoformat(),
        }
        artifact["PREDICTION_HASH"] = sha256_obj({k: v for k, v in artifact.items() if k != "PREDICTION_HASH"})

        out_path = PRED_DIR / f"{game_id}.json"
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        capture_log.append({"GAME_ID": game_id, "STATUS": "CAPTURED", "PREDICTION_HASH": artifact["PREDICTION_HASH"]})
        print(f"CAPTURED {game_id} kickoff={kickoff.isoformat()} prior_rows={len(prior_for_state)}", flush=True)

    (REPLAY_DIR / "SPORTS_NOVA_V18_WEEK1_CAPTURE_LOG.json").write_text(json.dumps(capture_log, indent=2))
    print(json.dumps({"CAPTURED": sum(1 for c in capture_log if c["STATUS"] == "CAPTURED"),
                       "INVALID": sum(1 for c in capture_log if c["STATUS"] == "INVALID_FOR_REPLAY"),
                       "TOTAL": len(capture_log)}, indent=2))


if __name__ == "__main__":
    main()
