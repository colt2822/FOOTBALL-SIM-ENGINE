"""SPORTS_NOVA M1 V23 validation: V21 (champion) vs V23 (V21 + the in-engine
residual-allocation fix, worker/sports_nova_v23/allocation.py) on the same
pilot cohort, same seeds.

Unlike scripts/sports_nova_m1_v4_v21_residual_ablation.py (which validates a
POST-HOC monkeypatch that redistributes residual mass with an EXTRA
rng.multinomial call after the original draw), this script runs V23's actual
shipped allocate_opportunities directly -- a single joint dirichlet-multinomial
draw over a reshaped (residual-bucket-dropped) probability vector, no extra
draw. The two mechanisms are different distributions with the same intent;
this script is the one that validates what actually ships. See
worker/sports_nova_v23/__init__.py for the full rationale.

Because the probability vector shape differs from V21's whenever a residual
would have existed, byte-identical replay between arms at the same seed is
NOT expected here (unlike the monkeypatch ablation, which is byte-identical
on no-residual-event paths by construction). A near-zero byte_identical_rate
in this script's output is the correct, expected result, not a regression.

Read-only / additive: worker/sports_nova_v21/ and worker/sports_nova_v23/ are
never edited by this script. Writes only to its own output path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import (
    instrument, pilot_games, game_parts, make_state, surrogate_kickoff, assert_path,
    PLAYER, MANIFEST, DATA,
)

OUT = DATA / "SPORTS_NOVA_M1_V23_VS_V21_ABLATION_V1.json"
N_PILOT = 128

ARMS = {"V21": "worker.sports_nova_v21.simulator", "V23": "worker.sports_nova_v23.simulator"}


def _tally(failures: list[dict]) -> dict:
    by_assertion: dict[str, int] = defaultdict(int)
    for f in failures:
        by_assertion[f["assertion"]] += 1
    return {"total": len(failures), "by_assertion": dict(by_assertion)}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    games_all = list(manifest["GAME_IDS"])
    games = pilot_games(games_all)
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    df = all_df[all_df.GAME_ID.isin(games)].copy()
    df_by_game = {g: df[df.GAME_ID == g] for g in games}

    qb_pred: dict[str, dict[tuple[str, str], list[int]]] = {a: defaultdict(list) for a in ARMS}
    team_stage: dict[str, dict[str, list[float]]] = {
        a: {"pass_attempts": [], "pass_yards": [], "rush_attempts": [], "rush_yards": []} for a in ARMS}
    failures: dict[str, list[dict]] = {a: [] for a in ARMS}
    arm_hash: dict[str, dict[tuple[str, int], str]] = {a: {} for a in ARMS}

    for arm, module_path in ARMS.items():
        collector: list[dict] = []
        with instrument(arm, collector, fix=False, detail=True, module_path=module_path) as module:
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
                qb_ids = set(str(x) for x in cur[cur.POSITION == "QB"].PLAYER_ID)
                for p in paths:
                    stats, team, winner_ = p["result"]
                    for pid in qb_ids:
                        if pid in stats:
                            qb_pred[arm][(game, pid)].append(int(stats[pid]["pass_yards"]))
                    assert_path(game, state, p, "V21", failures[arm])
                    for tid in team:
                        team_stage[arm]["pass_attempts"].append(float(team[tid]["pass_attempts"]))
                        team_stage[arm]["pass_yards"].append(float(team[tid]["pass_yards"]))
                        team_stage[arm]["rush_attempts"].append(float(team[tid]["rush_attempts"]))
                        team_stage[arm]["rush_yards"].append(float(team[tid]["rush_yards"]))
                    sim_key = (game, int(p["sim_id"]))
                    arm_hash[arm][sim_key] = hashlib.sha256(
                        json.dumps(stats, sort_keys=True).encode()).hexdigest()
                del collector[before:]

    def qb_errors(arm: str) -> dict[tuple[str, str], float]:
        errs = {}
        for (game, pid), vals in qb_pred[arm].items():
            obs_row = df_by_game[game]
            obs = obs_row[obs_row.PLAYER_ID.astype(str) == pid]["PASS_YARDS"]
            if obs.empty:
                continue
            errs[(game, pid)] = abs(float(np.mean(vals)) - float(obs.iloc[0]))
        return errs

    def qb_mae(arm: str) -> dict:
        vals = list(qb_errors(arm).values())
        return {"N": len(vals), "MAE": float(np.mean(vals)) if vals else None}

    e_base, e_fix = qb_errors("V21"), qb_errors("V23")
    common = sorted(set(e_base) & set(e_fix))
    deltas = [e_fix[k] - e_base[k] for k in common]
    n_fix_better = sum(1 for d in deltas if d < 0)
    n_base_better = sum(1 for d in deltas if d > 0)
    sign_p = (float(_stats.binomtest(min(n_fix_better, n_base_better),
                                      n_fix_better + n_base_better, 0.5).pvalue)
              if n_fix_better + n_base_better else None)

    def team_summary(arm: str) -> dict:
        return {k: {"mean": float(np.mean(v)), "N": len(v)} for k, v in team_stage[arm].items()}

    matches = sum(1 for k in arm_hash["V21"] if arm_hash["V21"].get(k) == arm_hash["V23"].get(k))

    out = {
        "cohort": {"games": len(games), "all_games": len(games_all), "sims_per_game": N_PILOT},
        "QB_PASS_YARD_MAE_V21": qb_mae("V21"),
        "QB_PASS_YARD_MAE_V23": qb_mae("V23"),
        "QB_PASS_YARD_PAIRED_SIGN_TEST": {
            "N_PAIRS": len(common), "N_V23_BETTER": n_fix_better, "N_V21_BETTER": n_base_better,
            "N_TIED": len(common) - n_fix_better - n_base_better,
            "MEAN_DELTA_V23_MINUS_V21": float(np.mean(deltas)) if deltas else None,
            "SIGN_TEST_TWO_SIDED_P": sign_p,
        },
        "TEAM_LEVEL_MEANS_V21": team_summary("V21"),
        "TEAM_LEVEL_MEANS_V23": team_summary("V23"),
        "TEAM_LEVEL_SHIFT_V23_MINUS_V21": {
            k: team_summary("V23")[k]["mean"] - team_summary("V21")[k]["mean"]
            for k in team_stage["V21"]
        },
        "ACCOUNTING_FAILURES_V21": _tally(failures["V21"]),
        "ACCOUNTING_FAILURES_V23": _tally(failures["V23"]),
        "byte_identical_rate_note": (
            "Expected near-zero, NOT a target of 1.0: V23's probability vector "
            "drops the residual bucket, so its dirichlet draw has a different "
            "shape than V21's whenever a residual would have existed. This is "
            "not the RNG-coupling integrity check from the monkeypatch ablation."
        ),
        "byte_identical_rate": matches / len(arm_hash["V21"]) if arm_hash["V21"] else None,
        "note": "Pilot-scale (60 games/128 sims). Validates the actual worker/sports_nova_v23 "
                "module directly, not the harness's post-hoc redistribution monkeypatch "
                "(see scripts/sports_nova_m1_v4_v21_residual_ablation.py for that, separate, result).",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
