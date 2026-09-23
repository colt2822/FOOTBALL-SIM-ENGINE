"""Build the V4 marginal/joint battery from the completed raw path ledger."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
RAW = DATA / "SPORTS_NOVA_M1_V4_RAW_PATH_LEDGER_V1.jsonl"
PLAYER = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
V21_JOINT = DATA / "SPORTS_NOVA_V21_JOINT_WALKFORWARD_PREDICTIONS_V1.parquet"
PRED_OUT = DATA / "SPORTS_NOVA_V4_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet"
JOINT_OUT = DATA / "SPORTS_NOVA_V4_JOINT_WALKFORWARD_PREDICTIONS_V1.parquet"
BATTERY_OUT = DATA / "SPORTS_NOVA_M1_V4_VALIDATION_BATTERY_V1.json"
N_SIMS = 16


def crps(x: np.ndarray, y: float) -> float:
    if not len(x):
        return float("nan")
    return float(np.mean(np.abs(x - y)) - .5 * np.mean(np.abs(x[:, None] - x[None, :])))


def pct(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else float("nan")


def flush_game(game: str, paths: list[dict], cur: pd.DataFrame, pred_rows: list, joint_rows: list, game_rows: list) -> None:
    if len(paths) != N_SIMS:
        raise RuntimeError(f"RAW_PATH_COUNT:{game}:{len(paths)}")
    pids = set().union(*(set(p["players"]) for p in paths))
    for r in cur.itertuples():
        pid = str(r.PLAYER_ID)
        stat = {"QB": "pass_yards", "RB": "rush_yards", "WR": "receiving_yards", "TE": "receiving_yards"}.get(r.POSITION)
        if not stat:
            continue
        vals = np.asarray([float(p["players"].get(pid, {}).get(stat, 0)) for p in paths], dtype=float)
        observed_col = {"pass_yards": "PASS_YARDS", "rush_yards": "RUSH_YARDS", "receiving_yards": "RECEIVING_YARDS"}[stat]
        y = float(getattr(r, observed_col))
        pred_rows.append({"GAME_ID": game, "PLAYER_ID": pid, "POSITION": r.POSITION,
                          "TARGET": stat.upper(), "OBSERVED": y, "PRED_MEAN": float(vals.mean()),
                          "PRED_SD": float(vals.std()), "P10": pct(vals, .10), "P25": pct(vals, .25),
                          "P05": pct(vals, .05), "P10": pct(vals, .10), "P25": pct(vals, .25),
                          "P50": pct(vals, .50), "P75": pct(vals, .75), "P90": pct(vals, .90),
                          "P95": pct(vals, .95),
                          "P_OVER_P50": float(np.mean(vals > np.median(vals))), "N_SIMS": N_SIMS,
                          "CRPS": crps(vals, y),
                          "PIT": float((np.sum(vals < y) + .5 * np.sum(vals == y)) / len(vals)),
                          "ROSTER_PRESENT_IN_ANY_PATH": pid in pids, "CAUSALITY_MODE": "EVENT_CAUSAL_ONLY"})
    qbs = cur[cur.POSITION == "QB"]
    rbs = cur[cur.POSITION == "RB"]
    receivers = cur[cur.POSITION.isin(["WR", "TE"])]
    for qr in qbs.itertuples():
        qid = str(qr.PLAYER_ID)
        qa = np.asarray([float(p["players"].get(qid, {}).get("pass_yards", 0)) for p in paths])
        qmed = float(np.median(qa))
        team = str(qr.TEAM)
        win = np.asarray([p["winner"] == ("HOME" if p["home_team"] == team else "AWAY") for p in paths])
        joint_rows.append({"GAME_ID": game, "FAMILY": "QB_PASS_YARDS_PLUS_TEAM_WIN", "PLAYER_ID": qid,
                           "N": N_SIMS, "V4_JOINT_P": float(np.mean((qa > qmed) & win)),
                           "A_MEDIAN": qmed, "REALIZED": int(float(qr.PASS_YARDS) > qmed)})
        for wr in receivers.itertuples():
            wid = str(wr.PLAYER_ID)
            wa = np.asarray([float(p["players"].get(wid, {}).get("receiving_yards", 0)) for p in paths])
            wm = float(np.median(wa))
            joint_rows.append({"GAME_ID": game, "FAMILY": "QB_PASS_YARDS_PLUS_WR_RECEIVING_YARDS",
                               "PLAYER_ID": qid + "|" + wid, "N": N_SIMS,
                               "V4_JOINT_P": float(np.mean((qa > qmed) & (wa > wm))),
                               "A_MEDIAN": qmed, "B_MEDIAN": wm,
                               "REALIZED": int(float(qr.PASS_YARDS) > qmed and float(wr.RECEIVING_YARDS) > wm)})
    for rr in rbs.itertuples():
        rid = str(rr.PLAYER_ID)
        ra = np.asarray([float(p["players"].get(rid, {}).get("rush_yards", 0)) for p in paths])
        rm = float(np.median(ra)); team = str(rr.TEAM)
        win = np.asarray([p["winner"] == ("HOME" if p["home_team"] == team else "AWAY") for p in paths])
        joint_rows.append({"GAME_ID": game, "FAMILY": "RB_RUSH_YARDS_PLUS_TEAM_WIN", "PLAYER_ID": rid,
                           "N": N_SIMS, "V4_JOINT_P": float(np.mean((ra > rm) & win)),
                           "A_MEDIAN": rm, "REALIZED": int(float(rr.RUSH_YARDS) > rm)})
    # Game-level simulated distributions are preserved for score/volume audit;
    # the supplied player artifact has no final-score columns, so score fields
    # remain explicitly unavailable rather than reconstructed from touchdowns.
    game_rows.append({"GAME_ID": game,
                      "sim_total_plays": [sum(json.loads(p["team_offense_json"])[i]["pass_attempts"] +
                                               json.loads(p["team_offense_json"])[i]["rush_attempts"]
                                               for i in range(2)) for p in paths],
                      "sim_pass_attempts": [sum(x["pass_attempts"] for x in json.loads(p["team_offense_json"])) for p in paths],
                      "sim_qb_yards": [sum(x["pass_yards"] for x in json.loads(p["team_offense_json"])) for p in paths]})


def metric_table(pred: pd.DataFrame) -> dict:
    out = {}
    for pos in ("QB", "RB", "WR", "TE"):
        x = pred[pred.POSITION == pos]
        if x.empty:
            continue
        err = x.PRED_MEAN.to_numpy() - x.OBSERVED.to_numpy()
        cov = {"50": float(((x.OBSERVED >= x.P25) & (x.OBSERVED <= x.P75)).mean()),
               "80": float(((x.OBSERVED >= x.P10) & (x.OBSERVED <= x.P90)).mean()),
               "90": float(((x.OBSERVED >= x.P05) & (x.OBSERVED <= x.P95)).mean())}
        out[pos] = {"N": int(len(x)), "MAE": float(np.abs(err).mean()),
                    "RMSE": float(np.sqrt(np.mean(err * err))), "bias": float(err.mean()),
                    "rank_correlation": float(x.OBSERVED.corr(x.PRED_MEAN, method="spearman")),
                    "coverage_50_80_90": cov,
                    "P90_observed": float(np.quantile(x.OBSERVED, .90)),
                    "P95_observed": float(np.quantile(x.OBSERVED, .95)),
                    "P99_observed": float(np.quantile(x.OBSERVED, .99)),
                    "P90_predicted": float(np.quantile(x.PRED_MEAN, .90)),
                    "P95_predicted": float(np.quantile(x.PRED_MEAN, .95)),
                    "P99_predicted": float(np.quantile(x.PRED_MEAN, .99)),
                    "CRPS": float(x.CRPS.mean()), "PIT_mean": float(x.PIT.mean()),
                    "PIT_uniformity_abs_bias": float(abs(x.PIT.mean() - .5))}
    return out


def main() -> None:
    obs = pd.read_parquet(PLAYER)
    obs = obs[obs.GAME_ID.str.match(r"^(2020|2021|2022|2023|2024|2025)_")].copy()
    by_game = {g: x for g, x in obs.groupby("GAME_ID", sort=False)}
    pred_rows: list[dict] = []; joint_rows: list[dict] = []; game_rows: list[dict] = []
    current_game = None; paths: list[dict] = []
    with RAW.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            row = json.loads(line)
            game = row["game_id"]
            if current_game is None:
                current_game = game
            if game != current_game:
                flush_game(current_game, paths, by_game[current_game], pred_rows, joint_rows, game_rows)
                paths = []; current_game = game
            paths.append({"players": {x["player_id"]: x for x in json.loads(row["player_json"])},
                          "winner": row["winner"], "home_team": row["home_team"],
                          "team_offense_json": row["team_offense_json"]})
    if current_game is not None:
        flush_game(current_game, paths, by_game[current_game], pred_rows, joint_rows, game_rows)
    pred = pd.DataFrame(pred_rows); joint = pd.DataFrame(joint_rows)
    pred.to_parquet(PRED_OUT, index=False); joint.to_parquet(JOINT_OUT, index=False)
    v21_joint = pd.read_parquet(V21_JOINT) if V21_JOINT.is_file() else pd.DataFrame()
    joint_delta = None
    if len(v21_joint) and len(joint):
        a = joint.groupby("FAMILY").V4_JOINT_P.mean()
        b = v21_joint.groupby("FAMILY").V21_JOINT_P.mean()
        joint_delta = {k: float(a[k] - b[k]) for k in a.index.intersection(b.index)}
    result = {"status": "COMPUTED_FROM_COMPLETE_RAW_LEDGER", "raw_rows": int(sum(1 for _ in RAW.open("r", encoding="utf-8"))),
              "games": int(pred.GAME_ID.nunique()), "player_prediction_rows": int(len(pred)),
              "joint_prediction_rows": int(len(joint)), "player_metrics": metric_table(pred),
              "joint_delta_vs_V21": joint_delta,
              "tail_delta_vs_V21": "COMPARE_V4_PREDICTION_PARQUET_TO_FROZEN_V21_ARTIFACT",
              "game_level": {"status": "NO_FINAL_SCORE_IN_PLAYER_INPUT; volume distributions emitted", "games": len(game_rows)},
              "PIT": "COMPUTED_FROM_16_RAW_PATHS_PER_PLAYER",
              "raw_ledger_sha256": None}
    import hashlib
    h = hashlib.sha256()
    with RAW.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""): h.update(block)
    result["raw_ledger_sha256"] = h.hexdigest()
    BATTERY_OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
