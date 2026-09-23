"""SPORTS_NOVA V28 HOLDOUT (out of development sample): V27 vs V28.12 on games the pilot never touched.

  python scripts/sports_nova_m1_v28_holdout.py freeze          # writes V28_4/HOLDOUT_PREREG.json (game list + numeric gates) BEFORE any holdout simulation
  python scripts/sports_nova_m1_v28_holdout.py run [--workers 4]
  python scripts/sports_nova_m1_v28_holdout.py analyze

The 60-game pilot was used four times (V28.1, .4/.6, .7/.8, .11): its results are DEVELOPMENT results.  This holdout is every 5th game of the frozen causal cohort manifest that is NOT in the pilot;
the V28.12 package is frozen (V28_4 prereg hash) and cannot change between freeze and analysis.  The V28.12 spec was fixed before any holdout game was simulated.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.sports_nova_m1_fg_causal_ablation_v1 as H  # noqa: E402
import scripts.sports_nova_m1_v28_game_state_scoring as R  # noqa: E402
from scripts.sports_nova_m1_fg_causal_ablation_v1_analysis import load_drives, observed_scores  # noqa: E402
from scripts.sports_nova_m1_v28_game_state_scoring_analysis import boot_delta, corr, err_stats, observed_final, shape  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import config as c27  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import simulator as v27  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import config as cfg  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import simulator as s28  # noqa: E402

TAG = "V28_4"
VARIANT = 12
OUT = R.out_dir(TAG) / "holdout"
PREREG = R.out_dir(TAG) / "HOLDOUT_PREREG.json"
N_SIMS = 128
STEP = 5
GATES = {
    "J_bias": "|total bias vs regulation actuals| <= 3.0", "K_tails": "mean |P_sim - P_hist| over the 12 ladder points (pooled sims vs 2014-2025 regulation history) <= V27's + 0.5pp",
    "L_discrimination": "total corr >= V27 - 0.03 AND margin corr >= V27 - 0.05 AND total RMSE <= V27 + 0.25",
    "M_dispersion": "team-score SD and total SD each within 12% of the 2014-2025 regulation history", "N_winner": "mean winner-Brier delta vs V27 <= +0.02 AND paired-bootstrap 95% CI lower bound <= 0",
    "TOTAL_YELLOW": "structural + J,K,L,M,N + total RMSE <= league-mean RMSE + 0.35 AND total corr >= V27 - 0.02 AND total RMSE <= V27 + 0.10",
    "CHAMPION": "GAME_SIM_CHAMPION iff the DEV structural gates + dev coherence hold AND J,K,L,M,N all hold on this holdout.  Otherwise no champion; V27 stays the engine of record.",
    "GREEN": "never assigned in this sprint (no calibration, no forward evidence)"}


def holdout_games() -> list[str]:
    pilot = set(R.pilot_games())
    allg = list(json.loads(H.MANIFEST.read_text())["GAME_IDS"])
    rest = [g for g in allg if g not in pilot]
    return rest[::STEP]


def cmd_freeze() -> None:
    if PREREG.exists() or OUT.exists():
        raise SystemExit("REFUSING: holdout prereg/outputs exist")
    pre = R.load_freeze(TAG)
    games = holdout_games()
    assert not (set(games) & set(pre["COHORT"]["game_ids"]))
    rec = {"SCHEMA": "SPORTS_NOVA_V28_HOLDOUT_PREREG", "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "V28_HASH": R.v28_hash(), "PREREG_TAG_SHA256": R.sha256_file(R.prereg_path(TAG)),
           "VARIANT": VARIANT, "MODEL_VERSION": cfg.VERSIONS[VARIANT], "SIMS_PER_GAME": N_SIMS, "SEED_RULE": "sha256(game_id)[:8] (H.game_seed)", "N_GAMES": len(games), "GAME_IDS": games,
           "SELECTION": f"manifest GAME_IDS minus the 60 pilot games, every {STEP}th in manifest order", "GATES": GATES, "BOOTSTRAP": "2000 resamples of games, seed 20260921",
           "BASELINE": "V27 = unpatched worker.sports_nova_v27_causal_fg_rate.simulator.simulate_scoring (scoring boundary) on the same state/seed/sims",
           "LEAGUE_MEAN_BASELINE": "causal trailing-5-season mean regulation total/margin (strictly earlier games)", "LIVE_CAPITAL_AUTHORIZED": False}
    OUT.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print("holdout frozen:", len(games), "games; V28_HASH", rec["V28_HASH"])


def run_game(args) -> dict:
    game, n = args
    state, home, away = R.game_state(game)
    seed = H.game_seed(game)
    t0 = time.time()
    b7 = v27.simulate_scoring(state, n, seed, c27.MODEL_VERSION)
    b8 = s28.simulate_scoring(state, n, seed, cfg.VERSIONS[VARIANT])
    np.savez_compressed(OUT / "arrays" / f"{game}.npz", v27_score=np.asarray(b7.team_stats["score"], np.int16), v27_winner=np.asarray(b7.winner),
                        v28_final=np.asarray(b8.team_stats["score"], np.int16), v28_reg=np.asarray(b8.team_stats["reg_score"], np.int16), v28_winner=np.asarray(b8.winner),
                        v28_ot=np.asarray(b8.team_stats["went_ot"], np.int8)[:, 0], v28_tds=np.asarray(b8.team_stats["tds"], np.int16))
    return {"game": game, "seconds": round(time.time() - t0, 1), "fg_rate": b8.runtime["fg_rate"], "slope": b8.runtime["recent_form_slope"]}


def cmd_run(workers: int) -> None:
    pre = json.loads(PREREG.read_text())
    assert pre["V28_HASH"] == R.v28_hash(), "V28 package changed after the holdout freeze"
    R.load_freeze(TAG)
    (OUT / "arrays").mkdir(parents=True, exist_ok=True)
    before = R.guarded_hashes()
    t0 = time.time()
    res = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_game, [(g, pre["SIMS_PER_GAME"]) for g in pre["GAME_IDS"]]):
            res.append(r)
            if len(res) % 25 == 0:
                print(len(res), "games", round(time.time() - t0), "s", flush=True)
    meta = {"WALL_SEC": time.time() - t0, "GUARDED_UNCHANGED": before == R.guarded_hashes(), "V28_HASH": R.v28_hash(), "RESULTS": sorted(res, key=lambda r: r["game"])}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print("HOLDOUT RUN DONE", round(time.time() - t0), "guarded_unchanged", meta["GUARDED_UNCHANGED"])


def load_wide() -> pd.DataFrame:
    cache = R.OUT / "wide_scores.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = load_drives()
    ids = d[d.season.between(2014, 2025)].game_id.unique().tolist()
    rows = []
    for g in ids:
        try:
            rows.append(observed_scores(d, [g]))
        except KeyError:
            pass
    w = pd.concat(rows, ignore_index=True)
    w = w[np.isfinite(w.obs_home + w.obs_away)].reset_index(drop=True)
    w["season"] = w.game.str.split("_").str[0].astype(int)
    w["week"] = w.game.str.split("_").str[1].astype(int)
    w["key"] = w.season * 100 + w.week
    w.to_parquet(cache)
    return w


def cmd_analyze() -> None:
    pre = json.loads(PREREG.read_text())
    meta = json.loads((OUT / "run_meta.json").read_text())
    games = pre["GAME_IDS"]
    d = load_drives()
    wide = load_wide()
    obs = {}
    for g in games:
        try:
            r = observed_scores(d, [g]).iloc[0]
            f = observed_final(d, [g]).iloc[0]
        except KeyError:
            continue
        if not np.isfinite(r.obs_home + r.obs_away + f.final_home + f.final_away):
            continue
        obs[g] = (r.obs_home, r.obs_away, f.final_home, f.final_away)
    games = [g for g in games if g in obs]
    Z = {g: np.load(OUT / "arrays" / f"{g}.npz") for g in games}
    n = len(games)
    oh = np.array([obs[g][0] for g in games]); oa = np.array([obs[g][1] for g in games]); fh = np.array([obs[g][2] for g in games]); fa = np.array([obs[g][3] for g in games])
    o_tot, o_mar = oh + oa, oh - oa
    home_won = (fh > fa).astype(float); dec = fh != fa
    base_tot, base_mar = [], []
    for g in games:
        s_, w_ = int(g.split("_")[0]), int(g.split("_")[1])
        prev = wide[(wide.key < s_ * 100 + w_) & (wide.season >= s_ - 5)]
        base_tot.append(float((prev.obs_home + prev.obs_away).mean())); base_mar.append(float((prev.obs_home - prev.obs_away).mean()))
    base_tot, base_mar = np.array(base_tot), np.array(base_mar)

    def arm(kind):
        sc = [Z[g]["v27_score"].astype(float) if kind == "v27" else Z[g]["v28_reg"].astype(float) for g in games]
        fin = [Z[g]["v27_score"] if kind == "v27" else Z[g]["v28_final"] for g in games]
        win = [Z[g]["v27_winner"] if kind == "v27" else Z[g]["v28_winner"] for g in games]
        pt = np.array([s.sum(axis=1).mean() for s in sc]); pm = np.array([(s[:, 0] - s[:, 1]).mean() for s in sc])
        ph = np.array([(w == "HOME").mean() + 0.5 * (w == "TIE").mean() for w in win])
        return sc, fin, win, pt, pm, ph

    S = {"V27": arm("v27"), "V28": arm("v28")}
    hs = {"team": np.r_[wide.obs_home, wide.obs_away], "tot": (wide.obs_home + wide.obs_away).to_numpy()}
    lad = lambda arr, xs, up: [float(((arr >= x) if up else (arr <= x)).mean()) for x in xs]
    TU, TOU, TL, TOL = [35, 40, 45, 50], [55, 60, 65, 70], [7, 10], [25, 30]
    hist_lad = lad(hs["team"], TU, True) + lad(hs["tot"], TOU, True) + lad(hs["team"], TL, False) + lad(hs["tot"], TOL, False)
    out: dict = {"SCHEMA": "SPORTS_NOVA_V28_HOLDOUT_ANALYSIS", "N_GAMES": n, "SIMS": pre["SIMS_PER_GAME"], "V28_HASH": R.v28_hash(), "MODEL_VERSION": pre["MODEL_VERSION"],
                 "GUARDED_UNCHANGED": meta["GUARDED_UNCHANGED"], "HISTORY": {"team_sd": float(hs["team"].std(ddof=1)), "total_sd": float(hs["tot"].std(ddof=1)), "team_mean": float(hs["team"].mean())}}
    for name, (sc, fin, win, pt, pm, ph) in S.items():
        team = np.concatenate([s.ravel() for s in sc]); tot = np.concatenate([s.sum(axis=1) for s in sc])
        ladder = lad(team, TU, True) + lad(tot, TOU, True) + lad(team, TL, False) + lad(tot, TOL, False)
        p, y = ph[dec], home_won[dec]
        out[name] = {"total": err_stats(pt, o_tot), "margin": err_stats(pm, o_mar), "winner": {"n_decisive": int(dec.sum()), "accuracy": float(((p > .5) == (y > .5)).mean()), "Brier": float(((p - y) ** 2).mean()),
                                                                                                     "logloss": float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean())},
                     "pts_per_team": float(team.mean()), "team_sd": float(team.std(ddof=1)), "total_sd": float(tot.std(ddof=1)), "ladder_MAE_pp": float(100 * np.mean(np.abs(np.array(ladder) - np.array(hist_lad)))),
                     "ladder": ladder, "total_shape": shape(tot), "tie_rate_final": float(np.mean(np.concatenate(win) == "TIE"))}
    out["HISTORY"]["ladder"] = hist_lad
    out["V28"]["overtime_rate"] = float(np.mean([Z[g]["v28_ot"].mean() for g in games]))
    out["LEAGUE_MEAN"] = {"total": err_stats(base_tot, o_tot), "margin": err_stats(base_mar, o_mar)}
    e7, e8 = S["V27"][3] - o_tot, S["V28"][3] - o_tot
    p7, p8 = S["V27"][5], S["V28"][5]
    B = lambda f: boot_delta(f, n)
    out["BOOT"] = {"total_RMSE_V28_minus_V27": B(lambda i: math.sqrt(np.mean(e8[i] ** 2)) - math.sqrt(np.mean(e7[i] ** 2))), "total_MAE_V28_minus_V27": B(lambda i: np.mean(np.abs(e8[i])) - np.mean(np.abs(e7[i]))),
                   "total_corr_V28_minus_V27": B(lambda i: corr(S["V28"][3][i], o_tot[i]) - corr(S["V27"][3][i], o_tot[i])),
                   "margin_corr_V28_minus_V27": B(lambda i: corr(S["V28"][4][i], o_mar[i]) - corr(S["V27"][4][i], o_mar[i])),
                   "total_RMSE_V28_minus_league_mean": B(lambda i: math.sqrt(np.mean(e8[i] ** 2)) - math.sqrt(np.mean((base_tot[i] - o_tot[i]) ** 2))),
                   "winner_Brier_V28_minus_V27": B(lambda i: float(np.mean(((p8[i] - home_won[i]) ** 2)[dec[i]]) - np.mean(((p7[i] - home_won[i]) ** 2)[dec[i]])))}
    out["PROB_MOVEMENT"] = {"mean_abs_dP_home": float(np.abs(p8 - p7).mean()), "flips": int(((p8 > .5) != (p7 > .5)).sum()), "mean_dP": float((p8 - p7).mean())}
    v7, v8, lm = out["V27"], out["V28"], out["LEAGUE_MEAN"]
    G = {"J": abs(v8["total"]["bias"]) <= 3.0, "K": v8["ladder_MAE_pp"] <= v7["ladder_MAE_pp"] + 0.5,
         "L": v8["total"]["corr"] >= v7["total"]["corr"] - 0.03 and v8["margin"]["corr"] >= v7["margin"]["corr"] - 0.05 and v8["total"]["RMSE"] <= v7["total"]["RMSE"] + 0.25,
         "M": abs(v8["team_sd"] / out["HISTORY"]["team_sd"] - 1) <= 0.12 and abs(v8["total_sd"] / out["HISTORY"]["total_sd"] - 1) <= 0.12,
         "N": out["BOOT"]["winner_Brier_V28_minus_V27"]["mean"] <= 0.02 and out["BOOT"]["winner_Brier_V28_minus_V27"]["ci95"][0] <= 0.0}
    dev = json.loads((R.out_dir(TAG) / "V28_ANALYSIS_v11.json").read_text())
    dev_struct = dev["GATES"]["structural"]["max_TD_per_possession"] == 1 and dev["GATES"]["structural"]["GT8_blocks"] == 0 and dev["GATES"]["coherence"]["pass_ge_0.45"] and all(v == 0 for k, v in dev["ACCOUNTING_TOTALS"].items() if k != "max_td_per_block")
    G["DEV_structural_and_coherence"] = bool(dev_struct)
    G["total_yellow_extra"] = {"RMSE_le_league_plus_0.35": v8["total"]["RMSE"] <= lm["total"]["RMSE"] + 0.35, "corr_ge_V27_minus_0.02": v8["total"]["corr"] >= v7["total"]["corr"] - 0.02,
                               "RMSE_le_V27_plus_0.10": v8["total"]["RMSE"] <= v7["total"]["RMSE"] + 0.10}
    G["CHAMPION"] = bool(all(G[k] for k in ("J", "K", "L", "M", "N")) and dev_struct)
    G["TOTAL_YELLOW"] = bool(G["CHAMPION"] and all(G["total_yellow_extra"].values()))
    out["GATES"] = G
    (OUT / "HOLDOUT_ANALYSIS.json").write_text(json.dumps(out, indent=1, default=float))
    show = {k: out[k] for k in ("N_GAMES", "V27", "V28", "LEAGUE_MEAN", "BOOT", "PROB_MOVEMENT", "GATES", "HISTORY")}
    for a in ("V27", "V28"):
        show[a] = {k: v for k, v in out[a].items() if k not in ("ladder", "total_shape")}
    show["HISTORY"] = {k: v for k, v in out["HISTORY"].items() if k != "ladder"}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "freeze":
        cmd_freeze()
    elif cmd == "run":
        cmd_run(int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--workers" else 4)
    elif cmd == "analyze":
        cmd_analyze()
    else:
        raise SystemExit(__doc__)
