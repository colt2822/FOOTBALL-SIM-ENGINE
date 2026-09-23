"""SPORTS_NOVA M1 Phase-3 full certification: V23 team/player realism across
the SAME full 1,693-game cohort and 16-sims/game resolution as V21's existing
comparison_artifacts.by_position numbers (data/sports_nova_v3/SPORTS_NOVA_M1_V4_RAW_PATH_DIAGNOSTIC_REPORT_V1.json),
so V23's numbers are directly comparable to V21's already-published ones,
not just to the 60-game pilot in scripts/sports_nova_m1_v23_realism_check.py.

Uses plain (uninstrumented) simulate_game -- no detail tracking, no
accounting re-check (already done at both pilot and full-cohort scale, see
[[project-sports-nova-v21-residual-fix-ablation]]) -- purely for realism
throughput. Read-only / additive.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sports_nova_v3_player_joint_walkforward_v1 import make_state, game_parts, PLAYER, MANIFEST
from worker.sports_nova_v23.simulator import simulate_game, MODEL_VERSION

DATA = ROOT / "data" / "sports_nova_v3"
OUT = DATA / "SPORTS_NOVA_M1_V23_FULL_COHORT_REALISM_V1.json"
N_SIMS = 16
POSITION_STAT = {"QB": "PASS_YARDS", "RB": "RUSH_YARDS", "WR": "RECEIVING_YARDS", "TE": "RECEIVING_YARDS"}
STAT_KEY = {"QB": "pass_yards", "RB": "rush_yards", "WR": "receiving_yards", "TE": "receiving_yards"}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    games = list(manifest["GAME_IDS"])
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK

    team_biases = defaultdict(list)
    pos_errs = defaultdict(list)
    pos_biases = defaultdict(list)
    n_done = 0
    for game_id in games:
        season, week, away, home = game_parts(game_id)
        key = season * 100 + week
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]
        cur = all_df[all_df.GAME_ID == game_id]
        if cur.empty or prior.empty:
            continue
        kickoff = datetime(season, 1, 1, tzinfo=timezone.utc) + timedelta(days=week * 7)
        state = make_state(game_id, prior, kickoff, None)
        seed = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
        batch = simulate_game(state, N_SIMS, seed, MODEL_VERSION)

        for tid in (home, away):
            if tid not in batch.team_ids:
                continue
            j = batch.team_ids.index(tid)
            obs = cur[cur.TEAM == tid]
            for k, obs_col in (("pass_attempts", "PASS_ATTEMPTS"), ("rush_attempts", "RUSH_ATTEMPTS"),
                               ("pass_yards", "PASS_YARDS"), ("rush_yards", "RUSH_YARDS")):
                obs_val = float(obs[obs_col].sum())
                pred_val = float(np.mean(batch.team_stats[k][:, j]))
                team_biases[k].append(pred_val - obs_val)

        pos_by_pid = dict(zip(cur.PLAYER_ID.astype(str), cur.POSITION))
        for j, pid in enumerate(batch.player_ids):
            position = pos_by_pid.get(pid)
            stat = STAT_KEY.get(position)
            if not stat:
                continue
            arr = batch.player_stats[stat][:, j]
            if arr.sum() == 0:
                continue
            obs_row = cur[cur.PLAYER_ID.astype(str) == pid][POSITION_STAT[position]]
            if obs_row.empty:
                continue
            pred_mean = float(np.mean(arr))
            obs_val = float(obs_row.iloc[0])
            pos_errs[position].append(abs(pred_mean - obs_val))
            pos_biases[position].append(pred_mean - obs_val)
        n_done += 1
        if n_done % 200 == 0:
            print(f"{n_done}/{len(games)} games done", file=sys.stderr)

    out = {
        "games_processed": n_done, "games_total": len(games), "sims_per_game": N_SIMS,
        "TEAM_REALISM": {
            k: {"N": len(v), "bias_vs_observed": float(np.mean(v)), "MAE_vs_observed": float(np.mean(np.abs(v)))}
            for k, v in team_biases.items()
        },
        "PLAYER_REALISM_BY_POSITION": {
            pos: {"N": len(pos_errs[pos]), "MAE": float(np.mean(pos_errs[pos])),
                  "RMSE": float(np.sqrt(np.mean(np.square(pos_biases[pos])))),
                  "bias": float(np.mean(pos_biases[pos]))}
            for pos in POSITION_STAT if pos_errs[pos]
        },
        "note": "Full 1,693-game cohort, 16 sims/game -- directly comparable to V21's existing "
                "comparison_artifacts.by_position numbers in SPORTS_NOVA_M1_V4_RAW_PATH_DIAGNOSTIC_REPORT_V1.json "
                "(QB MAE=81.06, RB MAE=24.95, WR MAE=26.46, TE MAE=19.23).",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
