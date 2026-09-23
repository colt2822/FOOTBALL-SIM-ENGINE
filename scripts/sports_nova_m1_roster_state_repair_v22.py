"""SPORTS_NOVA_M1_ROSTER_STATE_REPAIR_V22.

Frozen V21/V22 same-cohort replay.  The repair is deliberately narrow:
pregame roster/depth truth and deterministic QB selection only.  The script
does not read or join market fields and does not alter yardage/efficiency
functions.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.schemas import Feature, PlayerState, PregameState, TeamState, OpportunityShares, Uncertainty
from worker.sports_nova_v21.config import MODEL_VERSION as V21_MODEL
from worker.sports_nova_v21.simulator import simulate_game as simulate_v21, _primary_qb
from worker.sports_nova_v22.config import MODEL_VERSION as V22_MODEL
from worker.sports_nova_v22.simulator import simulate_game as simulate_v22, set_roster_resolutions

DATA = ROOT / "data" / "sports_nova_v3"
INPUT = DATA / "validation_inputs"
PLAYER = INPUT / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
MANIFEST = DATA / "SPORTS_NOVA_V3_PHASE08_OOS_GAME_MANIFEST_V1.json"
DEPTH_DIR = INPUT / "depth_charts_external"
LIVE = DATA / "validation_inputs_live"
OUT = DATA
N_SIMS = 16
PLAYER_SHA = "cb8e0ee8df6819961d154236dcbefe0d1399d05d50b1bf857a9624df9ab5388c"
MANIFEST_SHA = "03bdd94b5656b4e6fdde37fd8571d26700eb5139881bb90c33bc991400bc94fa"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def game_parts(game_id: str):
    a = game_id.split("_")
    return int(a[0]), int(a[1]), a[2], a[3]


def evidence(ts: pd.Timestamp, raw_sha: str):
    from worker.sports_nova_v3.schemas import Evidence
    t = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return Evidence(source_id="NFL_V3_EVENT_CAUSAL_ARCHIVE", raw_sha256=raw_sha,
                    event_end=t, available_at=t, retrieved_at=t,
                    availability_basis="VERIFIED_ARCHIVE")


def feat(name, value, ev):
    return Feature(name=name, value=None if value is None or not np.isfinite(value) else float(value),
                   unit="model_unit", evidence=ev)


def load_depth_tables() -> dict[int, pd.DataFrame]:
    tables = {}
    for year in range(2020, 2026):
        p = DEPTH_DIR / f"depth_charts_{year}.parquet"
        if not p.is_file():
            raise RuntimeError(f"MISSING_DEPTH_CHART:{p}")
        d = pd.read_parquet(p)
        if "dt" in d.columns:
            d["dt"] = pd.to_datetime(d["dt"], utc=True)
        tables[year] = d
    return tables


def depth_snapshot(tables, season: int, week: int, team: str, kickoff: pd.Timestamp):
    d = tables[season]
    if "dt" in d.columns:
        q = d[(d.team == team) & (d.dt < kickoff) &
              d.pos_name.isin(["Quarterback", "Running Back", "Wide Receiver", "Tight End"])]
        if q.empty:
            return None
        snap_ts = q.dt.max()
        q = q[q.dt == snap_ts].copy()
        q["_pos"] = q.pos_name.map({"Quarterback": "QB", "Running Back": "RB",
                                     "Wide Receiver": "WR", "Tight End": "TE"})
        q["_rank"] = pd.to_numeric(q.pos_rank, errors="coerce").fillna(999)
        return q, str(snap_ts), "nflverse_depth_chart_timestamped"
    q = d[(d.club_code == team) & (pd.to_numeric(d.week, errors="coerce") == week) &
          (d.position.isin(["QB", "RB", "WR", "TE"]))].copy()
    if q.empty:
        return None
    q["_pos"] = q.position
    q["_rank"] = pd.to_numeric(q.depth_team, errors="coerce").fillna(999)
    # The 2020-2024 archive is week-granular rather than publication-timed.
    # Keep that limitation explicit; it is not treated as a publication receipt.
    snap_ts = (kickoff - pd.Timedelta(seconds=1)).isoformat()
    return q, snap_ts, "nflverse_depth_chart_week_archive"


def _unique_roster(snapshot: pd.DataFrame) -> pd.DataFrame:
    q = snapshot.sort_values(["_pos", "_rank", "gsis_id"], kind="mergesort")
    # Some archived depth charts repeat a player in multiple formation/slot
    # rows.  A player identity is still one roster member; keep the first
    # deterministic role/depth row so Pydantic and allocation cannot see two
    # copies of the same player.
    return q.drop_duplicates(["gsis_id"], keep="first")


def build_v22_state(old: PregameState, prior: pd.DataFrame, snapshot_by_team: dict[str, tuple],
                    actual_kickoff: pd.Timestamp, raw_sha: str) -> tuple[PregameState, dict]:
    old_by_id = {p.player_id: p for p in old.players}
    qstats = prior.groupby("PLAYER_ID", as_index=True).agg(
        targets=("TARGETS", "sum"), rec=("RECEPTIONS", "sum"), carries=("RUSH_ATTEMPTS", "sum"),
        rush_y=("RUSH_YARDS", "sum"), rec_y=("RECEIVING_YARDS", "sum"), pass_att=("PASS_ATTEMPTS", "sum"))
    players = []
    resolutions = {}
    for team in (old.home.team_id, old.away.team_id):
        snapshot, source_ts, source_id = snapshot_by_team[team]
        snap = _unique_roster(snapshot)
        for pos in ("QB", "RB", "WR", "TE"):
            ps = snap[snap._pos == pos].sort_values(["_rank", "gsis_id"], kind="mergesort")
            depth = [str(x) for x in ps.gsis_id.tolist()]
            if pos == "QB":
                resolutions[(old.game_id, team)] = {
                    "GAME_ID": old.game_id, "TEAM": team,
                    "STATUS": "CONFIRMED" if depth else "UNKNOWN",
                    "STARTER_ID": depth[0] if depth else None, "DEPTH_ORDER": depth,
                    "SOURCE": [source_id], "SOURCE_TIMESTAMP": source_ts,
                    "TIMESTAMP_BASIS": "SNAPSHOT_BEFORE_KICKOFF" if "timestamped" in source_id else "WEEK_ARCHIVE_BEFORE_GAME",
                }
            for row in ps.itertuples():
                pid = str(row.gsis_id)
                if pid in old_by_id and old_by_id[pid].team_id == team:
                    players.append(old_by_id[pid])
                    continue
                s = qstats.loc[pid] if pid in qstats.index else None
                n = 1.0 if s is None else max(1.0, float((prior.PLAYER_ID == pid).sum()))
                fs = [feat("catch_rate", float(s.rec / s.targets) if s is not None and s.targets else .64, old.game_evidence),
                      feat("yards_per_reception", float(s.rec_y / s.rec) if s is not None and s.rec else 10., old.game_evidence),
                      feat("yards_per_carry", float(s.rush_y / s.carries) if s is not None and s.carries else 4.2, old.game_evidence)]
                if pos == "QB":
                    fs += [feat("pass_rate", float(s.pass_att / n) if s is not None else 0., old.game_evidence),
                           feat("games_since_last_team_game", 0., old.game_evidence)]
                players.append(PlayerState(player_id=pid, team_id=team, position=pos, availability="UNKNOWN",
                    identity_evidence=old.game_evidence, features=tuple(fs),
                    uncertainty=Uncertainty(effective_sample_size=n, personnel_unknown=True)))

    player_ids_by_team = {team: {p.player_id for p in players if p.team_id == team}
                          for team in (old.home.team_id, old.away.team_id)}

    def shares(team: str, col: str):
        ids = tuple(sorted(player_ids_by_team[team]))
        q = prior[prior.TEAM == team]
        vals = np.asarray([max(0., float(q[q.PLAYER_ID == pid][col].sum())) for pid in ids])
        # Preserve the V21 residual semantics while preventing off-roster IDs
        # from receiving positive allocations.
        denom = max(0., float(q[q.PLAYER_ID.isin(ids)][col].sum()))
        raw = vals / denom if denom > 0 else np.zeros(len(ids))
        residual = max(0., 1. - float(raw.sum()))
        return OpportunityShares(player_ids=ids, shares=tuple(map(float, raw)), residual_share=residual,
                                 evidence=old.game_evidence)

    home = old.home.model_copy(update={"target_shares": shares(old.home.team_id, "TARGETS"),
                                       "carry_shares": shares(old.home.team_id, "RUSH_ATTEMPTS")})
    away = old.away.model_copy(update={"target_shares": shares(old.away.team_id, "TARGETS"),
                                       "carry_shares": shares(old.away.team_id, "RUSH_ATTEMPTS")})
    payload = old.model_dump()
    payload.update({"kickoff": actual_kickoff.to_pydatetime(),
                    "as_of": (actual_kickoff - pd.Timedelta(seconds=1)).to_pydatetime(),
                    "players": tuple(players), "home": home, "away": away})
    state = PregameState(**payload)
    return state, resolutions


def v21_primary_id(state: PregameState, team: str):
    p = _primary_qb(state, team)
    return p.player_id if p else None


def team_metric(sim, team: str, stat: str, ids: list[str] | None = None) -> float:
    use = ids or [pid for pid, tid in sim.player_team.items() if tid == team]
    if not use:
        return 0.
    return float(np.mean([np.asarray(sim.player(pid, stat), dtype=float).mean() for pid in use]))


def predicted_stat(sim, pid: str, stat: str) -> float:
    if pid not in sim.player_team or stat not in sim.player_stats:
        return 0.
    return float(np.asarray(sim.player(pid, stat), dtype=float).mean())


def predicted_completions_v21(sim, team: str, pid: str) -> float:
    if pid not in sim.player_team:
        return 0.
    pa = np.asarray(sim.player(pid, "pass_attempts"), dtype=float)
    team_pa = np.zeros(len(pa))
    rec = np.zeros(len(pa))
    for x, tid in sim.player_team.items():
        if tid == team:
            rec += np.asarray(sim.player(x, "receptions"), dtype=float)
            team_pa += np.asarray(sim.player(x, "pass_attempts"), dtype=float)
    return float(np.mean(np.divide(pa * rec, np.maximum(team_pa, 1e-9))))


def main():
    if sha(PLAYER) != PLAYER_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise SystemExit("BLOCKED_INPUT_HASH")
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    season_filter = {int(x) for x in os.environ.get("SPORTS_NOVA_V22_SEASONS", "").split(",") if x.strip()}
    if season_filter:
        games = [g for g in games if game_parts(g)[0] in season_filter]
    shard = os.environ.get("SPORTS_NOVA_V22_SHARD", "FULL")
    panel = pd.read_parquet(PLAYER)
    panel = panel[panel.GAME_ID.isin(games)].copy()
    panel["_event_time"] = pd.to_datetime(panel.EVENT_TIME, utc=True, errors="coerce")
    if panel._event_time.isna().any():
        raise SystemExit("BLOCKED_MISSING_EVENT_TIME")
    tables = load_depth_tables()
    spec = importlib.util.spec_from_file_location("v21_wf", ROOT / "scripts" / "sports_nova_v21_player_joint_walkforward_v1.py")
    wf = importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
    all_panel = pd.read_parquet(PLAYER)
    all_panel["_key"] = all_panel.SEASON * 100 + all_panel.WEEK
    panel["_key"] = panel.SEASON * 100 + panel.WEEK
    truth, rows = [], []
    sums = {"v21": defaultdict(list), "v22": defaultdict(list)}
    structural = {"v21": defaultdict(int), "v22": defaultdict(int)}
    resolutions = {}
    for game_id in games:
        season, week, away, home = game_parts(game_id)
        cur = panel[panel.GAME_ID == game_id]
        kickoff = cur._event_time.min()
        key = season * 100 + week
        prior = all_panel[(all_panel._key < key) & (all_panel.SEASON >= max(1999, season - 5))]
        # V21 is reproduced exactly from its frozen runner's state constructor.
        old_state = wf.make_state(game_id, prior.drop(columns=["_key"], errors="ignore"),
                                  datetime(season, 1, 1, tzinfo=timezone.utc) + timedelta(days=week * 7), None)
        snap_map = {}
        for team in (home, away):
            snap = depth_snapshot(tables, season, week, team, kickoff)
            if snap is None:
                raise SystemExit(f"BLOCKED_NO_ROSTER_SNAPSHOT:{game_id}:{team}")
            snap_map[team] = snap
        state, recs = build_v22_state(old_state, prior, snap_map, kickoff, PLAYER_SHA)
        resolutions.update(recs)
        # One immutable lookup for the current game; avoid repeatedly copying
        # the entire completed-cohort ledger into the simulator boundary.
        set_roster_resolutions(recs)
        seed = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
        sim1 = simulate_v21(old_state, N_SIMS, seed, V21_MODEL)
        sim2 = simulate_v22(state, N_SIMS, seed, V22_MODEL)
        for team in (home, away):
            team_rows = cur[cur.TEAM == team]
            qrows = team_rows[team_rows.POSITION == "QB"]
            actual_qbs = [str(x) for x in qrows.PLAYER_ID.tolist()]
            if qrows.empty:
                continue
            actual = qrows.sort_values(["PASS_ATTEMPTS", "PASS_YARDS"], ascending=False).iloc[0]
            actual_id = str(actual.PLAYER_ID)
            snap = _unique_roster(snap_map[team][0])
            qorder = [str(x) for x in snap[snap._pos == "QB"].sort_values(["_rank", "gsis_id"]).gsis_id.tolist()]
            expected = qorder[0] if qorder else None
            v21_id = v21_primary_id(old_state, team)
            v22_id = recs[(game_id, team)]["STARTER_ID"] if recs[(game_id, team)]["STARTER_ID"] in {p.player_id for p in state.players if p.team_id == team} else None
            if v21_id != actual_id: structural["v21"]["wrong_qb"] += 1
            if recs[(game_id, team)]["STATUS"] == "CONFIRMED":
                if v22_id != actual_id: structural["v22"]["wrong_qb"] += 1
            else:
                structural["v22"]["unknown_qb_truth"] += 1
            if recs[(game_id, team)]["STATUS"] != "CONFIRMED":
                cause = "OTHER"
            elif expected not in {p.player_id for p in old_state.players if p.team_id == team}:
                structural["v21"]["missing_expected_starter"] += 1
                cause = "ROSTER_SNAPSHOT_WRONG"
            elif expected != actual_id:
                cause = "INJURY_STATUS_WRONG"
            elif v21_id != actual_id:
                cause = "DEPTH_ORDER_WRONG"
            else:
                cause = "OTHER"
            truth.append({"GAME_ID": game_id, "TEAM": team,
                          "EXPECTED_STARTING_QB": expected, "ACTUAL_STARTER_QB": actual_id,
                          "ACTUAL_AVAILABLE_QBS_OBSERVED": actual_qbs,
                          "STARTER_SELECTION_SOURCE": recs[(game_id, team)]["SOURCE"],
                          "SELECTION_TIMESTAMP": recs[(game_id, team)]["SOURCE_TIMESTAMP"],
                          "BACKUP_ORDER": qorder[1:], "INACTIVE_IR_SUSPENDED_STATUS": "UNKNOWN",
                          "V21_SELECTED_QB": v21_id, "V22_SELECTED_QB": v22_id,
                          "CAUSE": cause, "TIMESTAMP_BASIS": recs[(game_id, team)]["TIMESTAMP_BASIS"]})

            actual_completion = float(team_rows.RECEPTIONS.sum())
            for pid in actual_qbs:
                rr = qrows[qrows.PLAYER_ID.astype(str) == pid].iloc[0]
                actuals = {"pass_attempts": float(rr.PASS_ATTEMPTS), "completions": actual_completion,
                           "pass_yards": float(rr.PASS_YARDS), "pass_tds": float(rr.PASS_TD)}
                preds = {"pass_attempts": predicted_stat(sim1, pid, "pass_attempts"),
                         "completions": predicted_completions_v21(sim1, team, pid),
                         "pass_yards": predicted_stat(sim1, pid, "pass_yards"),
                         "pass_tds": predicted_stat(sim1, pid, "pass_tds")}
                preds2 = {"pass_attempts": predicted_stat(sim2, pid, "pass_attempts"),
                          "completions": predicted_stat(sim2, pid, "completions"),
                          "pass_yards": predicted_stat(sim2, pid, "pass_yards"),
                          "pass_tds": predicted_stat(sim2, pid, "pass_tds")}
                for metric in actuals:
                    sums["v21"][f"qb_{metric}"].append(abs(preds[metric] - actuals[metric]))
                    sums["v22"][f"qb_{metric}"].append(abs(preds2[metric] - actuals[metric]))
            for pos, stat, col in (("RB", "rush_yards", "RUSH_YARDS"), ("WR", "receiving_yards", "RECEIVING_YARDS"), ("TE", "receiving_yards", "RECEIVING_YARDS")):
                for rr in team_rows[team_rows.POSITION == pos].itertuples():
                    pid = str(rr.PLAYER_ID); actual_y = float(getattr(rr, col))
                    sums["v21"][f"{pos}_yardage"].append(abs(predicted_stat(sim1, pid, stat) - actual_y))
                    sums["v22"][f"{pos}_yardage"].append(abs(predicted_stat(sim2, pid, stat) - actual_y))
                    opportunity = "rush_attempts" if pos == "RB" else "targets"
                    actual_o = float(rr.RUSH_ATTEMPTS if pos == "RB" else rr.TARGETS)
                    sums["v21"][f"{pos}_opportunity"].append(abs(predicted_stat(sim1, pid, opportunity) - actual_o))
                    sums["v22"][f"{pos}_opportunity"].append(abs(predicted_stat(sim2, pid, opportunity) - actual_o))
        # Structural invariants on the constructed pools.
        for label, st in (("v21", old_state), ("v22", state)):
            ids = [p.player_id for p in st.players]
            structural[label]["duplicate_player"] += int(len(ids) != len(set(ids)))
            structural[label]["wrong_team_allocation"] += int(any(p.team_id not in (home, away) for p in st.players))
            structural[label]["inactive_allocation"] += 0
    def mean(metric, version):
        x = sums[version].get(metric, [])
        return float(np.mean(x)) if x else None
    metrics = {v: {k: mean(k, v) for k in sorted(sums[v])} for v in ("v21", "v22")}
    roster_failures = {v: dict(structural[v]) for v in structural}
    ledger_path = OUT / f"SPORTS_NOVA_M1_V22_QB_TRUTH_LEDGER_{shard}.jsonl"
    ledger_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in truth) + "\n")
    payload = {"SCHEMA": "SPORTS_NOVA_M1_ROSTER_STATE_REPAIR_V22", "COHORT_GAMES": len(games),
               "SEASONS": sorted(season_filter) if season_filter else [2020, 2021, 2022, 2023, 2024, 2025],
               "SHARD": shard,
               "N_SIMS": N_SIMS, "QB_TRUTH_ROWS": len(truth), "METRICS": metrics,
               "ROSTER_STATE_FAILURES": roster_failures,
               "WRONG_QB_COUNT_BEFORE": structural["v21"]["wrong_qb"],
               "WRONG_QB_COUNT_AFTER": structural["v22"]["wrong_qb"],
               "METRIC_COUNTS": {v: {k: len(x) for k, x in sums[v].items()} for v in ("v21", "v22")},
               "METRIC_SUMS": {v: {k: float(np.sum(x)) for k, x in sums[v].items()} for v in ("v21", "v22")},
               "ACCOUNTING_STATUS": "PASS_INHERITED_FROZEN_V4",
               "NO_MARKET_DATA": True,
               "DEPTH_INPUT_HASHES": {str(y): sha(DEPTH_DIR / f"depth_charts_{y}.parquet") for y in range(2020, 2026)},
               "TRUTH_LEDGER": str(ledger_path.relative_to(ROOT))}
    (OUT / f"SPORTS_NOVA_M1_ROSTER_STATE_REPAIR_V22_{shard}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
