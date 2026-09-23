"""SPORTS_NOVA M1 Phase 5/6: upcoming-slate production run on V23.

ONE COMMAND to (re)run the current upcoming NFL slate on the adopted V23
engine, using ONLY information available before kickoff:

    python scripts/sports_nova_m1_v23_upcoming_slate.py

M1 reality output only -- no market/odds columns from the schedule snapshot
(moneyline/spread/total) are read or used anywhere in this script, per
doctrine's DO_NOT list.

Slate selection: schedule_snapshot_2026.csv, season 2026, the lowest week
number that has any game with kickoff strictly after NOW. Games in that week
whose kickoff has already passed (e.g. a Thursday game once Sunday's slate is
being run) are excluded -- this is a pregame-only tool.

Injury/inactive handling: data/sports_nova_v3/validation_inputs_live/injuries_2026.parquet
(nflverse injury report release -- fetch/refresh with:
  curl -sL https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.parquet
    -o data/sports_nova_v3/validation_inputs_live/injuries_2026.parquet
this was previously assumed NO_DATA in this project; it exists and is free).
Only `report_status == "Out"` rows flip a player's PregameState.availability
to "OUT" (which allocate_opportunities already filters on -- no engine
change). "Questionable"/"Doubtful" are left at "UNKNOWN" rather than
resolved either way, per doctrine's "represent uncertainty explicitly,
do not silently assume" -- this engine has no partial-availability
mechanism, and forcing a binary guess on a genuinely uncertain status would
be exactly that silent assumption.

QB starters: schedule_snapshot's away_qb_id/home_qb_id, installed as V23's
identity resolution for each game with SOURCE_TIMESTAMP = the schedule
snapshot's own fetch time (the file's SOURCES.schedule.fetched_at_utc,
recorded in SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json) -- this is
before every game's kickoff in the current slate, so it is a valid pre-
kickoff identity source under set_identity_resolutions()'s own provenance
check. CONFIDENCE is marked HIGH, not CONFIRMED-inactive-list-grade: this is
the schedule provider's several-days-out starter designation, not a 90-
minute pregame inactive list. Represent that distinction explicitly in the
manifest rather than overstating certainty.

Team/player distributions come directly from worker.sports_nova_v23's own
SimulationBatch arrays (no instrumentation overhead). TD combo probabilities
also come from those arrays (scored_tds > 0 per sim is sufficient for
P(player), P(A+B), correlations). The exportable per-event TD ledger
(scorer_id/name/TD_type/quarter) requires the heavier per-block
instrumentation and is run separately, at a smaller per-game sim count, via
worker.sports_nova_v3.td_ledger.extract_td_events.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sports_nova_v3_player_joint_walkforward_v1 import make_state, game_parts, PLAYER
from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import instrument
from worker.sports_nova_v23.simulator import simulate_game, MODEL_VERSION, set_identity_resolutions
from worker.sports_nova_v3.td_ledger import extract_td_events, TDComboQuery
from worker.sports_nova_v3.schemas import Evidence

DATA = ROOT / "data" / "sports_nova_v3"
SCHEDULE = DATA / "validation_inputs_live" / "schedule_snapshot_2026.csv"
RECEIPT = DATA / "validation_inputs_live" / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"
INJURIES = DATA / "validation_inputs_live" / "injuries_2026.parquet"
OUT_DIR = DATA / "upcoming_slate"
N_SIMS = 10000  # SATURDAY_ACCEPTANCE target; measured ~44 sims/sec single-threaded on this
                # machine, so a 15-game slate at this setting takes roughly (10000*15)/44 =~ 57min.
N_TD_LEDGER_SIMS = 300
STAT_NAMES_BY_POSITION = {"QB": "pass_yards", "RB": "rush_yards", "WR": "receiving_yards", "TE": "receiving_yards"}
ET = ZoneInfo("America/New_York")


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.splitlines()
    except Exception as exc:
        return {"commit": None, "error": str(exc)}
    return {"commit": commit, "working_tree_dirty_file_count": len(dirty),
            "note": "This repo has a large pre-existing count of unrelated dirty/untracked files "
                    "(crypto/leadlag research, etc.) not committed by this pipeline; the commit hash "
                    "pins the base, not a guaranteed-clean tree. Not committed automatically."}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pctl(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def summary(arr: np.ndarray) -> dict:
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "P10": pctl(arr, 10), "P25": pctl(arr, 25), "P75": pctl(arr, 75), "P90": pctl(arr, 90)}


def select_slate(now_utc: datetime) -> tuple[pd.DataFrame, int]:
    sched = pd.read_csv(SCHEDULE)
    sched = sched[sched.season == 2026].copy()

    def kickoff_utc(row) -> datetime | None:
        if pd.isna(row.gameday) or pd.isna(row.gametime):
            return None
        local = datetime.strptime(f"{row.gameday} {row.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        return local.astimezone(timezone.utc)

    sched["kickoff_utc"] = sched.apply(kickoff_utc, axis=1)
    future = sched[sched.kickoff_utc > now_utc]
    if future.empty:
        raise SystemExit("BLOCKED_NO_UPCOMING_GAMES: no scheduled 2026 game has a kickoff after now")
    week = int(future.week.min())
    slate = future[future.week == week].sort_values("kickoff_utc")
    return slate, week


def build_identity_resolutions(slate: pd.DataFrame, source_timestamp: str) -> dict[tuple[str, str], dict]:
    resolutions: dict[tuple[str, str], dict] = {}
    for row in slate.itertuples():
        kickoff_iso = row.kickoff_utc.isoformat()
        for side, team, qb_id in (("away", row.away_team, row.away_qb_id), ("home", row.home_team, row.home_qb_id)):
            if pd.isna(qb_id):
                continue
            resolutions[(row.game_id, team)] = {
                "GAME_ID": row.game_id, "TEAM": team, "QB_ID": str(qb_id),
                "STATUS": "CONFIRMED", "CONFIDENCE": "HIGH",
                "SOURCE": "nflverse_schedule_games_csv_scheduled_starter",
                "SOURCE_TIMESTAMP": source_timestamp, "KICKOFF_TIME": kickoff_iso,
            }
    return resolutions


def load_out_players(week: int) -> tuple[set[str], dict]:
    """gsis_id set of players with report_status=='Out' for this week, plus a
    small evidence dict describing the source (used to build one Evidence
    object per flip, since PlayerState requires evidence for non-UNKNOWN
    availability)."""
    if not INJURIES.is_file():
        return set(), {}
    df = pd.read_parquet(INJURIES)
    wk = df[(df.week == week) & (df.report_status == "Out")]
    out_ids = set(wk.gsis_id.astype(str))
    fetched_at = datetime.fromtimestamp(INJURIES.stat().st_mtime, tz=timezone.utc)
    return out_ids, {"raw_sha256": sha256_file(INJURIES), "fetched_at": fetched_at}


def apply_injury_report(state, out_ids: set[str], injury_meta: dict):
    if not out_ids or not injury_meta:
        return state
    ev = Evidence(source_id="NFLVERSE_INJURY_REPORT_2026", raw_sha256=injury_meta["raw_sha256"],
                  event_end=injury_meta["fetched_at"], available_at=injury_meta["fetched_at"],
                  retrieved_at=injury_meta["fetched_at"], availability_basis="LIVE_RECEIPT")
    new_players = tuple(
        p.model_copy(update={"availability": "OUT", "availability_evidence": ev})
        if p.player_id in out_ids else p
        for p in state.players
    )
    if new_players == state.players:
        return state
    return state.model_copy(update={"players": new_players})


def run_game(game_id: str, prior: pd.DataFrame, kickoff_utc: datetime, causal_positions: pd.DataFrame,
             out_ids: set[str], injury_meta: dict) -> dict:
    season, week, away, home = game_parts(game_id)
    state = make_state(game_id, prior, kickoff_utc, None)
    state = apply_injury_report(state, out_ids, injury_meta)
    players_ruled_out = sorted(p.player_id for p in state.players if p.availability == "OUT")
    seed = int(hashlib.sha256((game_id + "_UPCOMING_SLATE").encode()).hexdigest()[:8], 16)
    batch = simulate_game(state, N_SIMS, seed, MODEL_VERSION)

    home_score = batch.team_stats["score"][:, batch.team_ids.index(home)]
    away_score = batch.team_stats["score"][:, batch.team_ids.index(away)]
    margin = home_score - away_score
    total = home_score + away_score
    win_home = float(np.mean(home_score > away_score))

    team_out = {}
    for tid in (home, away):
        j = batch.team_ids.index(tid)
        team_out[tid] = {
            "pass_attempts": summary(batch.team_stats["pass_attempts"][:, j]),
            "rush_attempts": summary(batch.team_stats["rush_attempts"][:, j]),
            "pass_yards": summary(batch.team_stats["pass_yards"][:, j]),
            "rush_yards": summary(batch.team_stats["rush_yards"][:, j]),
        }

    pos_by_pid = dict(zip(causal_positions.PLAYER_ID.astype(str), causal_positions.POSITION))
    name_by_pid = dict(zip(causal_positions.PLAYER_ID.astype(str), causal_positions.PLAYER_NAME))
    player_out: dict[str, dict] = {}
    scored_tds_by_pid: dict[str, np.ndarray] = {}
    for j, pid in enumerate(batch.player_ids):
        position = pos_by_pid.get(pid)
        stat = STAT_NAMES_BY_POSITION.get(position)
        if not stat:
            continue
        arr = batch.player_stats[stat][:, j]
        if arr.sum() == 0:
            continue
        player_out[pid] = {
            "name": name_by_pid.get(pid), "position": position, "team": batch.player_team[pid],
            **summary(arr),
        }
        scored_tds_by_pid[pid] = batch.player_stats["scored_tds"][:, j]

    n = N_SIMS
    td_probs = {pid: float(np.mean(arr > 0)) for pid, arr in scored_tds_by_pid.items() if np.mean(arr > 0) >= 0.02}
    top = sorted(td_probs.items(), key=lambda kv: kv[1], reverse=True)[:6]
    combo_matrix = []
    for i in range(len(top)):
        for k in range(i + 1, len(top)):
            pid_a, pid_b = top[i][0], top[k][0]
            both = float(np.mean((scored_tds_by_pid[pid_a] > 0) & (scored_tds_by_pid[pid_b] > 0)))
            combo_matrix.append({"player_a": pid_a, "player_b": pid_b,
                                 "p_a": top[i][1], "p_b": top[k][1], "p_both": both})

    return {
        "game_id": game_id, "home_team": home, "away_team": away,
        "win_probability_home": win_home, "win_probability_away": 1.0 - win_home,
        "score": {"home": summary(home_score), "away": summary(away_score)},
        "margin_distribution": summary(margin), "total_points_distribution": summary(total),
        "team": team_out,
        "player_td_probability": dict(sorted(td_probs.items(), key=lambda kv: kv[1], reverse=True)),
        "top_td_combo_matrix_same_or_cross_team": combo_matrix,
        "players_ruled_out": players_ruled_out,
        "players": player_out,
        "n_sims": n,
    }


def run_td_ledger(game_id: str, prior: pd.DataFrame, kickoff_utc: datetime, name_map: dict,
                   out_ids: set[str], injury_meta: dict) -> list[dict]:
    state = make_state(game_id, prior, kickoff_utc, None)
    state = apply_injury_report(state, out_ids, injury_meta)
    seed = int(hashlib.sha256((game_id + "_UPCOMING_SLATE_TD_LEDGER").encode()).hexdigest()[:8], 16)
    collector: list[dict] = []
    with instrument("V23", collector, fix=False, detail=True,
                     module_path="worker.sports_nova_v23.simulator") as module:
        module.simulate_game(state, N_TD_LEDGER_SIMS, seed, module.MODEL_VERSION)
    rows = []
    for p in collector:
        rows.extend(extract_td_events(game_id, p["sim_id"], p["blocks"], name_map))
    return rows


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    slate, week = select_slate(now_utc)
    receipt = json.loads(RECEIPT.read_text())
    source_timestamp = receipt["SOURCES"]["schedule"]["fetched_at_utc"]
    resolutions = build_identity_resolutions(slate, source_timestamp)
    set_identity_resolutions(resolutions)
    out_ids, injury_meta = load_out_players(week)

    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    causal = pd.read_parquet(ROOT / "data" / "sports_nova_v3" / "validation_inputs" /
                              "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet", columns=["PLAYER_ID", "PLAYER_NAME", "POSITION"])
    causal_positions = causal.drop_duplicates("PLAYER_ID", keep="last")
    name_map = dict(zip(causal_positions.PLAYER_ID.astype(str), causal_positions.PLAYER_NAME))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games_out = []
    td_ledger_all: list[dict] = []
    for row in slate.itertuples():
        key = int(f"{row.season}{row.week:02d}")
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, row.season - 5))]
        game_result = run_game(row.game_id, prior, row.kickoff_utc, causal_positions, out_ids, injury_meta)
        games_out.append(game_result)
        td_ledger_all.extend(run_td_ledger(row.game_id, prior, row.kickoff_utc, name_map, out_ids, injury_meta))
        print(f"done: {row.game_id}", file=sys.stderr)

    manifest = {
        "SCHEMA": "SPORTS_NOVA_M1_V23_UPCOMING_SLATE",
        "GENERATED_AT_UTC": now_utc.isoformat(),
        "MODEL_VERSION": MODEL_VERSION,
        "GIT_STATE": git_state(),
        "WEEK": week,
        "N_GAMES": len(games_out),
        "SIMS_PER_GAME": N_SIMS,
        "TD_LEDGER_SIMS_PER_GAME": N_TD_LEDGER_SIMS,
        "DATA_CUTOFF_NOTE": "prior = all seasons>=season-5 with (SEASON*100+WEEK) < this game's own week; "
                            "no in-week or postgame information used.",
        "QB_IDENTITY_SOURCE": "nflverse_schedule_games_csv_scheduled_starter (several-days-out designation, "
                               "NOT a 90-minute pregame inactive list -- see script docstring)",
        "INJURY_INPUT_STATUS": ("PRESENT" if injury_meta else "NO_DATA"),
        "INJURY_INPUT_SOURCE": "nflverse injuries_2026.parquet (report_status=='Out' only; "
                                "Questionable/Doubtful left UNKNOWN, not resolved -- see script docstring)",
        "INJURY_INPUT_FETCHED_AT": injury_meta.get("fetched_at").isoformat() if injury_meta else None,
        "PLAYERS_RULED_OUT_TOTAL": sum(len(g["players_ruled_out"]) for g in games_out),
        "QB_IDENTITY_SOURCE_TIMESTAMP": source_timestamp,
        "NO_MARKET_INPUTS": True,
        "RANDOM_SEED_POLICY": "sha256(game_id + '_UPCOMING_SLATE')[:8] as int, PCG64DXSM via SeedSequence([seed, sim_id])",
        "INPUT_MANIFEST": {
            "causal_panel_sha256": sha256_file(PLAYER),
            "schedule_snapshot_sha256": sha256_file(SCHEDULE),
            "injuries_sha256": sha256_file(INJURIES) if INJURIES.is_file() else None,
        },
        "ENGINE_SOURCE_SHA256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in sorted((ROOT / "worker" / "sports_nova_v23").glob("*.py"))
        },
        "GAME_IDS": [g["game_id"] for g in games_out],
    }
    (OUT_DIR / "SPORTS_NOVA_M1_V23_UPCOMING_SLATE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "SPORTS_NOVA_M1_V23_UPCOMING_SLATE_GAMES.json").write_text(json.dumps(games_out, indent=2))
    with (OUT_DIR / "SPORTS_NOVA_M1_V23_UPCOMING_SLATE_TD_LEDGER.jsonl").open("w") as fh:
        for row in td_ledger_all:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps({"WEEK": week, "N_GAMES": len(games_out), "TD_LEDGER_ROWS": len(td_ledger_all),
                      "OUT_DIR": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
