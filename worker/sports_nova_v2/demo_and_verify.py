"""Produce the 10,000-path simulation artifact and verify V1 is untouched."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from . import distribution as D
from . import features as F
from . import montecarlo as MC
from . import panel as P

NOW = datetime.now(timezone.utc).isoformat()
SCRATCH_V1 = os.environ.get("SPORTS_NOVA_V1_SOURCE")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_v1_preserved() -> dict:
    """Byte-compare the preserved V1 copies against the originals."""
    if not SCRATCH_V1:
        return {
            "files_compared": 0,
            "ALL_IDENTICAL": False,
            "reason": "SPORTS_NOVA_V1_SOURCE_NOT_CONFIGURED",
        }
    rows = {}
    for name in sorted(os.listdir(C.V1_DIR)):
        src = os.path.join(SCRATCH_V1, name)
        dst = C.V1_DIR / name
        if not os.path.isfile(src) or not dst.is_file():
            continue
        a, b = sha256_file(src), sha256_file(dst)
        rows[name] = {"original_sha256": a, "preserved_sha256": b, "identical": a == b}
    all_ok = all(r["identical"] for r in rows.values())
    return {"files_compared": len(rows), "ALL_IDENTICAL": all_ok, "files": rows}


def build_demo() -> dict:
    with open(C.ARTIFACTS / "_final_state.pkl", "rb") as fh:
        st = pickle.load(fh)
    el, v2_oos = st["el"], st["v2_oos"]

    # Most recent 2025 QB row with a healthy sample, used as the worked example.
    last = el[el.season == 2025].sort_values("kickoff_timestamp_utc").iloc[-1]
    row_oos = v2_oos.loc[last.name]
    pred, sigma = float(row_oos.pred), float(row_oos.sigma)

    zpool = np.load(C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_ZPOOL.npy")
    sample = sigma * zpool
    curve = D.dense_curve(pred, sample)
    stats = D.summary_stats(pred, sample)

    keys = sorted(float(k) for k in curve)
    vals = [curve[str(round(k, 1))] for k in keys]
    mono_viol = sum(1 for i in range(len(vals) - 1) if vals[i] < vals[i + 1] - 1e-12)

    # ---- correlated 10,000-path game simulation ----
    corr = json.loads(
        (C.ARTIFACTS / "SPORTS_NOVA_V2_JOINT_CORRELATIONS.json").read_text()
    )["CORRELATIONS"]

    full = P.load_event_causal()
    game_id = last.get("m_canonical_game_id")
    team, opp = last.team_canon, last.opp_canon

    def trailing_mean(pid, col, n=5):
        h = full[(full.player_id == pid)].sort_values("m_kickoff_timestamp_utc")
        h = h[h.m_kickoff_timestamp_utc < last.m_kickoff_timestamp_utc]
        return float(h[col].tail(n).mean()) if len(h) else np.nan

    mates = full[(full[MC.GAME_ID_COL] == game_id) &
                 (full.recent_team == last.recent_team)]
    wrs = mates[mates.position.isin(["WR", "TE"])].nlargest(2, "targets")
    rbs = mates[mates.position == "RB"].nlargest(1, "carries")

    players = [MC.PlayerSpec(str(last.player_id), "QB", "home", pred, sigma)]
    roster_note = []
    for _, w in wrs.iterrows():
        m = trailing_mean(w.player_id, "receiving_yards")
        if np.isfinite(m):
            players.append(MC.PlayerSpec(str(w.player_id), str(w.position),
                                         "home", m, max(m * 0.65, 20.0)))
            roster_note.append({"player_id": str(w.player_id),
                                "name": str(w.player_display_name),
                                "position": str(w.position),
                                "exp_yards_source": "trailing-5 mean (placeholder)"})
    for _, r in rbs.iterrows():
        m = trailing_mean(r.player_id, "rushing_yards")
        if np.isfinite(m):
            players.append(MC.PlayerSpec(str(r.player_id), "RB", "home", m,
                                         max(m * 0.7, 15.0)))
            roster_note.append({"player_id": str(r.player_id),
                                "name": str(r.player_display_name),
                                "position": "RB",
                                "exp_yards_source": "trailing-5 mean (placeholder)"})

    opp_qb = full[(full[MC.GAME_ID_COL] == game_id) & (full.position == "QB") &
                  (full.recent_team != last.recent_team)].nlargest(1, "attempts")
    for _, q in opp_qb.iterrows():
        m = trailing_mean(q.player_id, "passing_yards")
        if np.isfinite(m):
            players.append(MC.PlayerSpec(str(q.player_id), "QB", "away", m,
                                         max(m * 0.35, 45.0)))
            roster_note.append({"player_id": str(q.player_id),
                                "name": str(q.player_display_name),
                                "position": "QB (opponent)",
                                "exp_yards_source": "trailing-5 mean (placeholder)"})

    spec = MC.GameSpec(
        home_team=str(team), away_team=str(opp),
        exp_plays_home=float(last.get("off_plays_w5", 63) or 63),
        exp_plays_away=float(last.get("def_plays_w5", 63) or 63),
        exp_pass_rate_home=float(last.get("off_pass_rate_w5", 0.57) or 0.57),
        exp_pass_rate_away=float(last.get("def_pass_rate_w5", 0.57) or 0.57),
        players=players)

    sim = MC.GameSimulator(corr).simulate(spec, n_paths=10_000, seed=C.SEED)

    qb_id = str(last.player_id)
    joint = {}
    if len(players) > 1:
        wr_id = players[1].player_id
        legs = [(qb_id, 250.0), (wr_id, 60.0)]
        joint["QB_over_250_AND_WR1_over_60"] = MC.joint_p_over(sim, legs)
    if len(players) >= 4:
        joint["QB_over_250_AND_OPP_QB_over_225"] = MC.joint_p_over(
            sim, [(qb_id, 250.0), (players[-1].player_id, 225.0)])

    marg = {p.player_id: {
        "position": p.position, "team_side": p.team,
        "mean": float(np.mean(sim["draws"][p.player_id])),
        "p50": float(np.median(sim["draws"][p.player_id])),
        "std": float(np.std(sim["draws"][p.player_id])),
    } for p in players}

    return {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_MONTE_CARLO_DEMO",
        "GENERATED_AT": NOW,
        "N_PATHS": 10_000,
        "SEED": C.SEED,
        "WORKED_EXAMPLE_QB": {
            "player_id": str(last.player_id),
            "player": str(last.get("player_display_name", "")),
            "season": int(last.season), "week": int(last.week),
            "team": str(team), "opponent": str(opp),
            "MEAN": stats["MEAN"], "MEDIAN": stats["MEDIAN"], "STD": stats["STD"],
            "P10": stats["P10"], "P25": stats["P25"], "P50": stats["P50"],
            "P75": stats["P75"], "P90": stats["P90"],
            "P_OVER_MAIN": {f"P_OVER_{t}": float((pred + sample > t).mean())
                            for t in C.THRESH_LIST_MAIN},
            "ACTUAL_RESULT": float(last[C.TARGET]),
        },
        "DENSE_CURVE_MONOTONIC": {"grid_points": len(vals),
                                  "violations": mono_viol, "PASS": mono_viol == 0},
        "DENSE_CURVE_P_OVER_HALF_UNIT": curve,
        "SIMULATION_INPUTS": {
            "exp_plays_home": spec.exp_plays_home,
            "exp_plays_away": spec.exp_plays_away,
            "exp_pass_rate_home": spec.exp_pass_rate_home,
            "exp_pass_rate_away": spec.exp_pass_rate_away,
            "correlations_used": corr,
            "roster": roster_note,
            "CAVEAT": ("only the QB marginal comes from the fitted V2 model; "
                       "WR/TE/RB/opponent-QB marginals are trailing-5 placeholders "
                       "because only QB_PASS_YARDS is a validated V2 target. The "
                       "correlation structure is real and measured; the non-QB "
                       "marginals are not yet model-quality."),
        },
        "SIMULATED_MARGINALS": marg,
        "JOINT_PROBABILITIES": joint,
        "WHY_THIS_MATTERS": ("CORRELATION_MULTIPLIER is the ratio of the true "
                             "joint probability to the independence product. A "
                             "value above 1 means an independence assumption "
                             "underprices that parlay; below 1 means it overprices."),
    }


def main():
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    (C.ARTIFACTS / "SPORTS_NOVA_V2_MONTE_CARLO_DEMO.json").write_text(
        json.dumps(demo, indent=2, default=str))

    v1 = verify_v1_preserved()
    (C.ARTIFACTS / "SPORTS_NOVA_V2_V1_INTEGRITY_CHECK.json").write_text(
        json.dumps({"ARTIFACT_ID": "SPORTS_NOVA_V2_V1_INTEGRITY_CHECK",
                    "GENERATED_AT": NOW,
                    "V1_SOURCE": SCRATCH_V1 or "NOT_CONFIGURED",
                    "V1_PRESERVED_AT": str(C.V1_DIR),
                    **v1}, indent=2, default=str))

    print(json.dumps({
        "V1_INTEGRITY": {"files_compared": v1["files_compared"],
                         "ALL_IDENTICAL": v1["ALL_IDENTICAL"]},
        "MC_PATHS": demo["N_PATHS"],
        "MONOTONIC": demo["DENSE_CURVE_MONOTONIC"],
        "EXAMPLE": {k: demo["WORKED_EXAMPLE_QB"][k] for k in
                    ("player", "season", "week", "MEAN", "STD", "P10", "P90",
                     "ACTUAL_RESULT")},
        "JOINT": demo["JOINT_PROBABILITIES"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
