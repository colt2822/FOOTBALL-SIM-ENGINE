"""Diagnostic-only: MAE/RMSE of QB pass_yards broken down by the REASON
classification from sports_nova_v3_qb_allocation_diagnostic_v1.py, on the
SAME 60-game/128-sim cohort as SPORTS_NOVA_V3_ENGINE_REPAIR_VALIDATION_V2.
Makes NO engine changes. Answers: if identity were perfectly repaired, where
would MAE land?
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game, _primary_qb, _feature
from sports_nova_v3_player_joint_walkforward_v1 import (PLAYER, MANIFEST, game_parts,
    surrogate_kickoff, make_state)

OUT = ROOT / "data" / "sports_nova_v3"
N = 128

def main():
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    oos = all_df[all_df.GAME_ID.isin(selected)].copy()

    rows = []
    for g in selected:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        sim = simulate_game(state, N, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16), MODEL_VERSION)
        cur = oos[oos.GAME_ID == g]
        qcur = cur[cur.POSITION == "QB"]

        state_players = {p.player_id: p for p in state.players}
        primary = {}
        for tid in (home, away):
            pq = _primary_qb(state, tid)
            primary[tid] = pq.player_id if pq is not None else None

        for tid in (home, away):
            for r in qcur[qcur.TEAM == tid].itertuples():
                pid = str(r.PLAYER_ID)
                in_roster = pid in state_players
                resolved_pos = state_players[pid].position if in_roster else None
                in_sim = pid in sim.player_team
                sim_att = float(np.mean(sim.player(pid, "pass_attempts"))) if in_sim else 0.0
                sim_yds = float(np.mean(sim.player(pid, "pass_yards"))) if in_sim else 0.0
                is_primary = (pid == primary[tid])

                if not in_roster:
                    reason = "ROSTER_STATE_FAILURE"
                elif resolved_pos != "QB":
                    reason = "PLAYER_ID_MAPPING_FAILURE"
                elif primary[tid] is None:
                    reason = "NO_QB_SELECTED"
                elif not is_primary:
                    reason = "WRONG_QB_SELECTED"
                elif sim_att == 0:
                    reason = "TEAM_PASS_ATTEMPTS_NOT_ASSIGNED"
                else:
                    reason = "OK"

                rows.append({
                    "GAME_ID": g, "TEAM": tid, "EVAL_QB_ID": pid,
                    "REALIZED_ATTEMPTS": float(r.PASS_ATTEMPTS), "REALIZED_YARDS": float(r.PASS_YARDS),
                    "SIM_QB_ATTEMPTS": sim_att, "SIM_QB_YARDS": sim_yds,
                    "ABS_ERR_YARDS": abs(sim_yds - float(r.PASS_YARDS)),
                    "REASON": reason,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "SPORTS_NOVA_V3_QB_MAE_BY_REASON_ROWS_V1.csv", index=False)

    def summarize(g):
        return {
            "N": int(len(g)),
            "MAE": float(g.ABS_ERR_YARDS.mean()) if len(g) else None,
            "RMSE": float(np.sqrt((g.ABS_ERR_YARDS ** 2).mean())) if len(g) else None,
            "mean_realized_yards": float(g.REALIZED_YARDS.mean()) if len(g) else None,
            "mean_sim_yards": float(g.SIM_QB_YARDS.mean()) if len(g) else None,
            "mean_realized_att": float(g.REALIZED_ATTEMPTS.mean()) if len(g) else None,
            "mean_sim_att": float(g.SIM_QB_ATTEMPTS.mean()) if len(g) else None,
        }

    result = {
        "n_eval_rows": len(df),
        "overall": summarize(df),
        "by_reason": {reason: summarize(sub) for reason, sub in df.groupby("REASON")},
    }
    (OUT / "SPORTS_NOVA_V3_QB_MAE_BY_REASON_SUMMARY_V1.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
