"""Diagnostic-only (Y1): decompose matched-QB pass-yards error into attempts,
completion rate, and yards-per-completion, on the SAME 60-game/128-sim cohort
and REASON classification as the QB identity repair. Makes NO engine changes.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game, _primary_qb
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
                is_primary = in_roster and resolved_pos == "QB" and primary[tid] is not None and pid == primary[tid]
                if not (in_roster and resolved_pos == "QB" and in_sim and is_primary):
                    continue  # OK-only per Y1 scope
                sim_att = float(np.mean(sim.player(pid, "pass_attempts")))
                if sim_att == 0:
                    continue
                sim_yds = float(np.mean(sim.player(pid, "pass_yards")))
                # team receivers' total receptions on this QB's attempts approximate his completions
                teammates = [q for q in sim.player_ids if sim.player_team[q] == tid]
                sim_comp = float(np.mean(sum(sim.player(q, "receptions") for q in teammates)))
                real_att = float(r.PASS_ATTEMPTS)
                real_comp = float(r.COMPLETIONS)
                real_yds = float(r.PASS_YARDS)
                rows.append({
                    "GAME_ID": g, "TEAM": tid, "QB_ID": pid, "SEASON": s,
                    "REAL_ATT": real_att, "SIM_ATT": sim_att,
                    "REAL_COMP": real_comp, "SIM_COMP": sim_comp,
                    "REAL_COMP_RATE": real_comp / real_att if real_att else np.nan,
                    "SIM_COMP_RATE": sim_comp / sim_att if sim_att else np.nan,
                    "REAL_YDS": real_yds, "SIM_YDS": sim_yds,
                    "REAL_YPA": real_yds / real_att if real_att else np.nan,
                    "SIM_YPA": sim_yds / sim_att if sim_att else np.nan,
                    "REAL_YPC": real_yds / real_comp if real_comp else np.nan,
                    "SIM_YPC": sim_yds / sim_comp if sim_comp else np.nan,
                    "ABS_ERR_YDS": abs(sim_yds - real_yds),
                    "ATT_BUCKET": ("<20" if real_att < 20 else "20-30" if real_att < 30 else
                                   "30-40" if real_att < 40 else "40+"),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "SPORTS_NOVA_V3_QB_YARDS_DECOMP_ROWS_V1.csv", index=False)

    def summ(g):
        return {
            "N": int(len(g)),
            "MAE_yards": float(g.ABS_ERR_YDS.mean()),
            "bias_yards_sim_minus_real": float((g.SIM_YDS - g.REAL_YDS).mean()),
            "sd_err_yards": float((g.SIM_YDS - g.REAL_YDS).std(ddof=1)) if len(g) > 1 else None,
            "real_att_mean": float(g.REAL_ATT.mean()), "sim_att_mean": float(g.SIM_ATT.mean()),
            "real_comp_rate_mean": float(g.REAL_COMP_RATE.mean()), "sim_comp_rate_mean": float(g.SIM_COMP_RATE.mean()),
            "real_ypa_mean": float(g.REAL_YPA.mean()), "sim_ypa_mean": float(g.SIM_YPA.mean()),
            "real_ypc_mean": float(g.REAL_YPC.mean()), "sim_ypc_mean": float(g.SIM_YPC.mean()),
            "real_yds_sd": float(g.REAL_YDS.std(ddof=1)) if len(g) > 1 else None,
            "sim_yds_sd": float(g.SIM_YDS.std(ddof=1)) if len(g) > 1 else None,
        }

    result = {
        "n_matched_qb_rows": len(df),
        "overall": summ(df),
        "by_season": {int(s): summ(sub) for s, sub in df.groupby("SEASON")},
        "by_att_bucket": {b: summ(sub) for b, sub in df.groupby("ATT_BUCKET")},
    }
    (OUT / "SPORTS_NOVA_V3_QB_YARDS_DECOMP_SUMMARY_V1.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print(json.dumps(result["by_att_bucket"], indent=2))

if __name__ == "__main__":
    main()
