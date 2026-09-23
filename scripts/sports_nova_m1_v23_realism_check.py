"""SPORTS_NOVA M1 Phase-3 (partial): V23 vs V21 player/team realism, same
pilot cohort/seeds as scripts/sports_nova_m1_v23_vs_v21_ablation.py (which
already showed team-level pass/rush attempts+yards shift under 1.5 yards and
QB_PASS_YARD_MAE within noise, p=0.58). This script extends that check to
RB/WR/TE positions and reports bias/MAE/RMSE vs OBSERVED for both arms,
directly paired on the same cohort -- not a full 1,693-game battery (that is
the honest gap left for full certification; see NEXT_SINGLE_ACTION in the
mission status), but enough to catch a large realism regression from the
fix, which is what this check is for.

Read-only / additive, same cohort machinery as the other V4-family scripts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import (
    instrument, pilot_games, game_parts, make_state, surrogate_kickoff,
    PLAYER, MANIFEST, DATA,
)

OUT = DATA / "SPORTS_NOVA_M1_V23_REALISM_CHECK_V1.json"
N_PILOT = 128
ARMS = {"V21": "worker.sports_nova_v21.simulator", "V23": "worker.sports_nova_v23.simulator"}
POSITION_STAT = {"QB": "pass_yards", "RB": "rush_yards", "WR": "receiving_yards", "TE": "receiving_yards"}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    games_all = list(manifest["GAME_IDS"])
    games = pilot_games(games_all)
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    df = all_df[all_df.GAME_ID.isin(games)].copy()
    df_by_game = {g: df[df.GAME_ID == g] for g in games}

    pred: dict[str, dict[str, dict[tuple[str, str], list[int]]]] = {
        a: {pos: defaultdict(list) for pos in POSITION_STAT} for a in ARMS}
    team_pred: dict[str, dict[tuple[str, str], dict]] = {a: {} for a in ARMS}

    for arm, module_path in ARMS.items():
        collector: list[dict] = []
        with instrument(arm, collector, fix=False, detail=False, module_path=module_path) as module:
            for game in games:
                season, week, away, home = game_parts(game)
                key = season * 100 + week
                prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]
                state = make_state(game, prior, surrogate_kickoff(season, week), None)
                seed = int(hashlib.sha256(game.encode()).hexdigest()[:8], 16)
                before = len(collector)
                module.simulate_game(state, N_PILOT, seed, module.MODEL_VERSION)
                paths = collector[before:]
                cur = df_by_game[game]
                pos_by_pid = {str(r.PLAYER_ID): r.POSITION for r in cur.itertuples()}
                team_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
                for p in paths:
                    stats, team, _ = p["result"]
                    for pid, position in pos_by_pid.items():
                        stat = POSITION_STAT.get(position)
                        if stat and pid in stats:
                            pred[arm][position][(game, pid)].append(int(stats[pid][stat]))
                    for tid, t in team.items():
                        for k in ("pass_attempts", "rush_attempts", "pass_yards", "rush_yards"):
                            team_acc[tid][k].append(float(t[k]))
                for tid, vals in team_acc.items():
                    team_pred[arm][(game, tid)] = {k: float(np.mean(v)) for k, v in vals.items()}
                del collector[before:]

    def position_metrics(arm: str, position: str) -> dict:
        stat = POSITION_STAT[position]
        obs_col = {"pass_yards": "PASS_YARDS", "rush_yards": "RUSH_YARDS",
                   "receiving_yards": "RECEIVING_YARDS"}[stat]
        errs, biases = [], []
        for (game, pid), vals in pred[arm][position].items():
            obs_row = df_by_game[game]
            obs = obs_row[obs_row.PLAYER_ID.astype(str) == pid][obs_col]
            if obs.empty:
                continue
            pred_mean = float(np.mean(vals))
            obs_val = float(obs.iloc[0])
            errs.append(abs(pred_mean - obs_val))
            biases.append(pred_mean - obs_val)
        return {"N": len(errs), "MAE": float(np.mean(errs)) if errs else None,
                "RMSE": float(np.sqrt(np.mean(np.square(biases)))) if biases else None,
                "bias": float(np.mean(biases)) if biases else None}

    def team_metrics(arm: str) -> dict:
        out = {}
        for k, obs_cols in (("pass_attempts", "PASS_ATTEMPTS"), ("rush_attempts", "RUSH_ATTEMPTS"),
                            ("pass_yards", "PASS_YARDS"), ("rush_yards", "RUSH_YARDS")):
            biases = []
            for (game, tid), v in team_pred[arm].items():
                obs_row = df_by_game[game]
                obs_val = float(obs_row[obs_row.TEAM == tid][obs_cols].sum())
                biases.append(v[k] - obs_val)
            out[k] = {"N": len(biases), "bias_vs_observed": float(np.mean(biases)) if biases else None,
                      "MAE_vs_observed": float(np.mean(np.abs(biases))) if biases else None}
        return out

    out = {
        "cohort": {"games": len(games), "all_games": len(games_all), "sims_per_game": N_PILOT},
        "note": "Pilot-scale (60 games/128 sims), paired V21/V23 on same cohort. "
                "NOT the full 1,693-game battery -- see mission NEXT_SINGLE_ACTION for full certification.",
        "TEAM_REALISM_V21": team_metrics("V21"),
        "TEAM_REALISM_V23": team_metrics("V23"),
        "PLAYER_REALISM_V21": {pos: position_metrics("V21", pos) for pos in POSITION_STAT},
        "PLAYER_REALISM_V23": {pos: position_metrics("V23", pos) for pos in POSITION_STAT},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
