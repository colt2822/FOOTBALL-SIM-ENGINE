"""Analysis of a V28 pilot run vs the sealed V27 baseline and history.   python scripts/sports_nova_m1_v28_game_state_scoring_analysis.py analyze VARIANT [TAG]

Reads only: the V28 pilot arrays, the sealed V27 (MULTI_TD ablation BASE) arrays, the frozen drive table.  Writes V28_ANALYSIS_v<VARIANT>.json.
Regulation quantities (reg_score) are compared with regulation actuals; final quantities (incl. overtime) with final outcomes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.sports_nova_m1_v28_game_state_scoring as R  # noqa: E402
from scripts.sports_nova_m1_fg_causal_ablation_v1_analysis import load_drives, observed_scores  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import config as cfg  # noqa: E402

V27_FIELDS = ["sim", "block", "off", "sec_before", "pre_h", "pre_a", "plays", "pass_att", "designed", "scrambles", "targets", "rec", "pass_yds", "rush_yds",
              "rec_sites", "carry_sites", "trial_plays", "raw_td", "raw_rec_td", "raw_rush_td", "raw_pts", "td", "rec_td", "rush_td", "supp_td", "fg_elig", "fg", "pts"]
TEAM_UP, TOT_UP, TEAM_LO, TOT_LO = [35, 40, 45, 50], [55, 60, 65, 70], [7, 10], [25, 30]
BOOT = 2000
RNG = np.random.default_rng(20260921)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def shape(v):
    v = np.asarray(v, float)
    s = v.std(ddof=1)
    z = (v - v.mean()) / s
    return {"skew": float((z ** 3).mean()), "excess_kurtosis": float((z ** 4).mean() - 3.0), "cv": float(s / v.mean())}


# ---------------------------------------------------------------------------------------------- history
def observed_final(d: pd.DataFrame, games: list[str]) -> pd.DataFrame:
    rows = []
    for g in games:
        x = d[d.game_id == g].sort_values("source_drive_number")
        parts = g.split("_")
        away, home = parts[2], parts[3]
        sc = {home: 0.0, away: 0.0}
        for r in x.itertuples():
            off = r.offense_team
            de = away if off == home else home
            pts_off = float(r.points_scored)
            delta = float(r.end_score_diff - r.start_score_diff)
            sc[off] += pts_off
            sc[de] += pts_off - delta
        rows.append({"game": g, "final_home": sc[home], "final_away": sc[away]})
    return pd.DataFrame(rows)


def history(games: list[str]) -> dict:
    cache = R.OUT / "hist_reference.json"
    d = load_drives()
    drv = pd.read_parquet(R.DATA / "validation_inputs" / "NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet",
                          columns=["game_id", "season", "week", "offense_team", "touchdowns", "plays", "pass_yards", "rush_yards", "points_scored", "possession_result",
                                   "field_goal_attempts"])
    out: dict = {}
    obs_p = observed_scores(d, games).set_index("game")
    fin_p = observed_final(d, games).set_index("game")
    wide_ids = d[d.season.between(2014, 2025)].game_id.unique().tolist()
    rows, skipped = [], 0
    for g in wide_ids:
        try:
            rows.append(observed_scores(d, [g]))
        except KeyError:
            skipped += 1
    wide = pd.concat(rows, ignore_index=True)
    n_nan = int((~np.isfinite(wide.obs_home + wide.obs_away)).sum())
    wide = wide[np.isfinite(wide.obs_home + wide.obs_away)].reset_index(drop=True)      # games whose first-drive score is NaN in the source cannot be reconstructed
    wide["season"] = wide.game.str.split("_").str[0].astype(int)
    wide["week"] = wide.game.str.split("_").str[1].astype(int)
    wide["key"] = wide.season * 100 + wide.week
    team = np.r_[wide.obs_home.to_numpy(), wide.obs_away.to_numpy()]
    tot = (wide.obs_home + wide.obs_away).to_numpy()
    lad = lambda arr, xs, up: {str(x): float((arr >= x).mean() if up else (arr <= x).mean()) for x in xs}
    out["scores_2014_2025"] = {"games": int(len(wide)), "skipped_franchise_code": skipped, "dropped_nan_games": n_nan, "team_mean": float(team.mean()), "team_sd": float(team.std(ddof=1)),
                               "total_mean": float(tot.mean()), "total_sd": float(tot.std(ddof=1)), "margin_sd": float((wide.obs_home - wide.obs_away).std(ddof=1)),
                               "team_up": lad(team, TEAM_UP, True), "total_up": lad(tot, TOT_UP, True), "team_lo": lad(team, TEAM_LO, False), "total_lo": lad(tot, TOT_LO, False),
                               "team_shape": shape(team), "total_shape": shape(tot), "reg_tie_rate": float((wide.obs_home == wide.obs_away).mean()),
                               "home_away_corr": corr(wide.obs_home, wide.obs_away)}
    pop = drv[drv.season.between(2020, 2025)]
    ngames = pop.game_id.nunique()
    yds = pop.pass_yards.astype(float) + pop.rush_yards.astype(float)
    res = pop.possession_result.astype(str).str.upper().str.replace(" ", "_", regex=False)
    out["drives_2020_2025"] = {"games": int(ngames), "drives_per_team_game": float(len(pop) / (2 * ngames)), "TD_per_team_game": float(pop.touchdowns.sum() / (2 * ngames)),
                               "TD_possession_rate": float((pop.touchdowns > 0).mean()), "TD_possessions_per_team_game": float((pop.touchdowns > 0).sum() / (2 * ngames)),
                               "max_TD_per_drive": int(pop.touchdowns.max()), "FG_made_per_team_game": float((res == "FIELD_GOAL").sum() / (2 * ngames)),
                               "GT8_share": float((pop.points_scored > 8).mean()), "spearman_yards_TD": float(pd.Series(yds.to_numpy()).corr(pd.Series((pop.touchdowns > 0).astype(int).to_numpy()), method="spearman"))}
    bins = [(-1e9, 10), (10, 30), (30, 50), (50, 70), (70, 1e9)]
    out["TD_by_yards_bin_2020_2025"] = {f"({lo},{hi}]": float((pop.touchdowns[(yds > lo) & (yds <= hi)] > 0).mean()) for lo, hi in bins}
    out["coherence_range_history"] = float((pop.touchdowns[yds >= 70] > 0).mean() - (pop.touchdowns[yds <= 10] > 0).mean())
    tg = pop.assign(yds=yds).groupby(["game_id", "offense_team"]).agg(y=("yds", "sum"), p=("points_scored", "sum"))
    out["team_game_corr_yards_points_2020_2025"] = corr(tg.y, tg.p)
    # league-mean baseline (causal trailing 5 seasons, strictly earlier games) for every pilot game
    base = {}
    for g in games:
        season, week = int(g.split("_")[0]), int(g.split("_")[1])
        prev = wide[(wide.key < season * 100 + week) & (wide.season >= season - 5)]
        base[g] = {"total": float((prev.obs_home + prev.obs_away).mean()), "team": float(np.r_[prev.obs_home, prev.obs_away].mean()), "margin": float((prev.obs_home - prev.obs_away).mean())}
    out["league_mean_baseline"] = base
    out["obs_pilot"] = {g: {"home": float(obs_p.loc[g].obs_home), "away": float(obs_p.loc[g].obs_away), "final_home": float(fin_p.loc[g].final_home),
                            "final_away": float(fin_p.loc[g].final_away), "ot": bool(obs_p.loc[g].ot)} for g in games}
    cache.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------------- arms
def load_arm(kind: str, games: list[str], variant: int = 1, tag: str = "V28_1") -> dict:
    A = {"score": [], "final": [], "winner": [], "yards": [], "blocks": [], "went_ot": []}
    for g in games:
        if kind == "v27":
            z = np.load(R.V27_ARR / f"{g}.npz")
            sc = z["base__team_score"]
            A["score"].append(sc); A["final"].append(sc); A["winner"].append(z["base__winner"]); A["went_ot"].append(np.zeros(len(sc), int))
            A["yards"].append(z["base__team_pass_yards"] + z["base__team_rush_yards"])
            b = pd.DataFrame(z["base__blocks"], columns=V27_FIELDS)
            b["yds"] = b.pass_yds + b.rush_yds
        else:
            z = np.load(R.out_dir(tag) / f"pilot_v{variant}" / "arrays" / f"{g}.npz")
            A["score"].append(z["reg_score"]); A["final"].append(z["team_score"]); A["winner"].append(z["winner"]); A["went_ot"].append(z["went_ot"])
            A["yards"].append(z["team_pass_yards"] + z["team_rush_yards"])
            b = pd.DataFrame(z["blocks"], columns=__import__("worker.sports_nova_v28_game_state_scoring.simulator", fromlist=["BLOCK_FIELDS"]).BLOCK_FIELDS)
        b["game"] = g
        A["blocks"].append(b)
    A["blocks"] = pd.concat(A["blocks"], ignore_index=True)
    return A


def per_game(A, games):
    tot = [s.sum(axis=1).mean() for s in A["score"]]
    mar = [(s[:, 0] - s[:, 1]).mean() for s in A["score"]]
    team = [s.mean(axis=0) for s in A["score"]]
    pw = []
    for f, w in zip(A["final"], A["winner"]):
        pw.append(((w == "HOME").mean() + 0.5 * (w == "TIE").mean(), (w == "TIE").mean()))
    return {"total": np.array(tot), "margin": np.array(mar), "team": np.array(team), "p_home": np.array([p[0] for p in pw]), "p_tie": np.array([p[1] for p in pw])}


def err_stats(pred, obs):
    e = np.asarray(pred) - np.asarray(obs)
    return {"bias": float(e.mean()), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "corr": corr(pred, obs), "n": int(len(e))}


def arm_metrics(A, games, n=R.N_SIMS):
    sc = np.concatenate(A["score"]); fin = np.concatenate(A["final"]); w = np.concatenate(A["winner"])
    tot = sc.sum(axis=1); team = sc.ravel(); mar = sc[:, 0] - sc[:, 1]
    b = A["blocks"]
    ntg = len(games) * n * 2
    up = lambda arr, xs: {str(x): float((arr >= x).mean()) for x in xs}
    lo = lambda arr, xs: {str(x): float((arr <= x).mean()) for x in xs}
    demean = lambda i: np.concatenate([s[:, i] - s[:, i].mean() for s in A["score"]])
    ydem = np.concatenate([(y - y.mean(axis=0)).ravel() for y in A["yards"]])
    pdem = np.concatenate([(s - s.mean(axis=0)).ravel() for s in A["score"]])
    yd = b.yds.to_numpy(); tdp = (b.td > 0).to_numpy()
    lo_m, hi_m = yd <= 10, yd >= 70
    out = {"pts_per_team_game": float(team.mean()), "TD_per_team_game": float(b.td.sum() / ntg), "FG_per_team_game": float(b.fg.sum() / ntg),
           "total_mean": float(tot.mean()), "margin_mean": float(mar.mean()), "team_sd": float(team.std(ddof=1)), "total_sd": float(tot.std(ddof=1)), "margin_sd": float(mar.std(ddof=1)),
           "team_up": up(team, TEAM_UP), "total_up": up(tot, TOT_UP), "team_lo": lo(team, TEAM_LO), "total_lo": lo(tot, TOT_LO), "team_shape": shape(team), "total_shape": shape(tot),
           "blocks_per_team_game": float(len(b) / ntg), "TD_possession_rate": float(tdp.mean()), "TD_possessions_per_team_game": float(tdp.sum() / ntg),
           "multi_TD_possession_share": float((b.td > 1).mean()), "GT8_share": float((b.pts > 8).mean()), "GT8_blocks": int((b.pts > 8).sum()),
           "reg_tie_rate": float((sc[:, 0] == sc[:, 1]).mean()), "final_tie_rate": float((w == "TIE").mean()), "overtime_rate": float(np.concatenate(A["went_ot"]).mean()),
           "home_away_score_corr": corr(demean(0), demean(1)), "team_game_corr_yards_points_within_game": corr(ydem, pdem), "block_corr_yards_points": corr(yd, b.pts),
           "block_spearman_yards_TD": float(pd.Series(yd).corr(pd.Series(tdp.astype(int)), method="spearman")),
           "coherence_range_TD_yards_ge70_minus_le10": float(tdp[hi_m].mean() - tdp[lo_m].mean()),
           "TD_by_yards_bin": {f"({lo_},{hi_}]": float(tdp[(yd > lo_) & (yd <= hi_)].mean()) for lo_, hi_ in [(-1e9, 10), (10, 30), (30, 50), (50, 70), (70, 1e9)]},
           "share_blocks_yards_gt100": float((yd > 100).mean()), "yards_per_block": float(yd.mean()), "turnovers": "NOT_REPRESENTED_IN_ENGINE",
           "team_game_yards_mean": float(np.concatenate([y.ravel() for y in A["yards"]]).mean()), "team_game_yards_sd": float(np.concatenate([y.ravel() for y in A["yards"]]).std()),
           "team_game_yards_cv": float(np.concatenate([y.ravel() for y in A["yards"]]).std() / np.concatenate([y.ravel() for y in A["yards"]]).mean())}
    if "td_qb_rush" in b:
        out["QB_rush_TD_per_team_game"] = float(b.td_qb_rush.sum() / ntg)
        out["pass_TD_share"] = float(b.td_pass.sum() / max(1, b.td.sum()))
        out["rush_TD_share"] = float(b.td_rush.sum() / max(1, b.td.sum()))
        out["TD_unattributable_blocks"] = int(b.td_unattr.sum())
    else:
        out["rush_TD_share"] = float(b.rush_td.sum() / max(1, b.td.sum()))
    return out


def ladder_mae(m, hist, keys):
    xs = []
    for k, sect in keys:
        for x, v in m[sect].items():
            xs.append(abs(v - hist[sect][x]))
    return float(np.mean(xs))


def boot_delta(fn, n_games, B=BOOT):
    idx = RNG.integers(0, n_games, size=(B, n_games))
    vals = np.array([fn(i) for i in idx])
    return {"mean": float(vals.mean()), "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


def cmd_analyze(variant: int, tag: str) -> None:
    pre = R.load_freeze(tag)
    games = pre["COHORT"]["game_ids"]
    hpath = R.OUT / "hist_reference.json"
    hist = json.loads(hpath.read_text()) if hpath.exists() else history(games)
    A27, A28 = load_arm("v27", games), load_arm("v28", games, variant, tag)
    m27, m28 = arm_metrics(A27, games), arm_metrics(A28, games)
    hs = hist["scores_2014_2025"]
    P27, P28 = per_game(A27, games), per_game(A28, games)
    obs_tot = np.array([hist["obs_pilot"][g]["home"] + hist["obs_pilot"][g]["away"] for g in games])
    obs_mar = np.array([hist["obs_pilot"][g]["home"] - hist["obs_pilot"][g]["away"] for g in games])
    obs_team = np.array([[hist["obs_pilot"][g]["home"], hist["obs_pilot"][g]["away"]] for g in games])
    base_tot = np.array([hist["league_mean_baseline"][g]["total"] for g in games])
    base_mar = np.array([hist["league_mean_baseline"][g]["margin"] for g in games])
    base_team = np.array([hist["league_mean_baseline"][g]["team"] for g in games])
    fin_home = np.array([hist["obs_pilot"][g]["final_home"] for g in games]); fin_away = np.array([hist["obs_pilot"][g]["final_away"] for g in games])
    decisive = fin_home != fin_away
    home_won = (fin_home > fin_away).astype(float)
    disc = {}
    for name, P in (("V27", P27), ("V28", P28)):
        d = {"total": err_stats(P["total"], obs_tot), "margin": {**err_stats(P["margin"], obs_mar)}, "team_score": err_stats(P["team"].ravel(), obs_team.ravel())}
        p = P["p_home"][decisive]; y = home_won[decisive]
        d["winner"] = {"decisive_games": int(decisive.sum()), "accuracy": float(((p > 0.5) == (y > 0.5)).mean()), "Brier": float(((p - y) ** 2).mean()), "logloss": float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean())}
        disc[name] = d
    disc["LEAGUE_MEAN"] = {"total": err_stats(base_tot, obs_tot), "margin": err_stats(base_mar, obs_mar), "team_score": err_stats(np.repeat(base_team, 2), obs_team.ravel())}
    p50 = np.full(int(decisive.sum()), 0.5)
    disc["LEAGUE_MEAN"]["winner"] = {"Brier_coinflip": float(((p50 - home_won[decisive]) ** 2).mean())}
    e27, e28 = P27["total"] - obs_tot, P28["total"] - obs_tot
    n = len(games)
    boots = {
        "total_RMSE_V28_minus_V27": boot_delta(lambda i: math.sqrt(np.mean(e28[i] ** 2)) - math.sqrt(np.mean(e27[i] ** 2)), n),
        "total_MAE_V28_minus_V27": boot_delta(lambda i: np.mean(np.abs(e28[i])) - np.mean(np.abs(e27[i])), n),
        "total_corr_V28_minus_V27": boot_delta(lambda i: corr(P28["total"][i], obs_tot[i]) - corr(P27["total"][i], obs_tot[i]), n),
        "total_RMSE_V28_minus_league_mean": boot_delta(lambda i: math.sqrt(np.mean(e28[i] ** 2)) - math.sqrt(np.mean((base_tot[i] - obs_tot[i]) ** 2)), n),
        "total_bias_removed_RMSE_V28_minus_V27": boot_delta(lambda i: np.std(e28[i]) - np.std(e27[i]), n),
        "margin_corr_V28_minus_V27": boot_delta(lambda i: corr(P28["margin"][i], obs_mar[i]) - corr(P27["margin"][i], obs_mar[i]), n),
    }
    dp = P28["p_home"] - P27["p_home"]
    prob_move = {"mean_abs_dP_home": float(np.abs(dp).mean()), "mean_dP_home": float(dp.mean()), "flips_of_favorite": int(((P28["p_home"] > .5) != (P27["p_home"] > .5)).sum()),
                 "expected_MC_noise_mean_abs_dP_unpaired": float(np.mean(np.sqrt(2 * P27["p_home"] * (1 - P27["p_home"]) / R.N_SIMS)) * math.sqrt(2 / math.pi)),
                 "Brier_delta_V28_minus_V27": boot_delta(lambda i: np.mean(((P28["p_home"] - home_won) ** 2)[decisive & np.isin(np.arange(n), i)]) if False else
                                                          float(np.mean(((P28["p_home"][i] - home_won[i]) ** 2)[decisive[i]]) - np.mean(((P27["p_home"][i] - home_won[i]) ** 2)[decisive[i]])), n)}
    tails = {"historical": {k: hs[k] for k in ("team_up", "total_up", "team_lo", "total_lo")}, "V27": {k: m27[k] for k in ("team_up", "total_up", "team_lo", "total_lo")},
             "V28": {k: m28[k] for k in ("team_up", "total_up", "team_lo", "total_lo")}}
    keys_up = [("x", "team_up"), ("x", "total_up")]
    keys_all = keys_up + [("x", "team_lo"), ("x", "total_lo")]
    tail_summary = {"upper_ladder_MAE_pp": {"V27": 100 * ladder_mae(m27, hs, keys_up), "V28": 100 * ladder_mae(m28, hs, keys_up)},
                    "all_ladder_MAE_pp": {"V27": 100 * ladder_mae(m27, hs, keys_all), "V28": 100 * ladder_mae(m28, hs, keys_all)},
                    "shape_gap_total": {"V27": {k: abs(m27["total_shape"][k] - hs["total_shape"][k]) for k in hs["total_shape"]}, "V28": {k: abs(m28["total_shape"][k] - hs["total_shape"][k]) for k in hs["total_shape"]}}}
    hd = hist["drives_2020_2025"]
    struct = {"TD_per_team_game": {"hist": hd["TD_per_team_game"], "V27": m27["TD_per_team_game"], "V28": m28["TD_per_team_game"]},
              "TD_possession_rate": {"hist": hd["TD_possession_rate"], "V27": m27["TD_possession_rate"], "V28": m28["TD_possession_rate"]},
              "TD_possessions_per_team_game": {"hist": hd["TD_possessions_per_team_game"], "V27": m27["TD_possessions_per_team_game"], "V28": m28["TD_possessions_per_team_game"]},
              "blocks_or_drives_per_team_game": {"hist": hd["drives_per_team_game"], "V27": m27["blocks_per_team_game"], "V28": m28["blocks_per_team_game"]},
              "FG_per_team_game": {"hist": hd["FG_made_per_team_game"], "V27": m27["FG_per_team_game"], "V28": m28["FG_per_team_game"]},
              "GT8_share": {"hist": hd["GT8_share"], "V27": m27["GT8_share"], "V28": m28["GT8_share"]},
              "multi_TD_possession_share": {"hist": 0.0, "V27": m27["multi_TD_possession_share"], "V28": m28["multi_TD_possession_share"]},
              "coherence_range": {"hist": hist["coherence_range_history"], "V27": m27["coherence_range_TD_yards_ge70_minus_le10"], "V28": m28["coherence_range_TD_yards_ge70_minus_le10"]},
              "TD_by_yards_bin": {"hist": hist["TD_by_yards_bin_2020_2025"], "V27": m27["TD_by_yards_bin"], "V28": m28["TD_by_yards_bin"]},
              "team_game_corr_yards_points": {"hist": hist["team_game_corr_yards_points_2020_2025"], "V27": m27["team_game_corr_yards_points_within_game"], "V28": m28["team_game_corr_yards_points_within_game"]},
              "share_blocks_yards_gt100": {"V27": m27["share_blocks_yards_gt100"], "V28": m28["share_blocks_yards_gt100"]}}
    # ---- gates (numbers frozen in the prereg)
    b28 = disc["V28"]["total"]["bias"]
    G = {"J_no_catastrophic_mean_regression": {"total_bias": b28, "abs_le_3": abs(b28) <= 3.0, "pts_per_team_vs_hist": m28["pts_per_team_game"] - hs["team_mean"],
                                               "abs_le_1.5": abs(m28["pts_per_team_game"] - hs["team_mean"]) <= 1.5},
         "K_tails": {"all_ladder_MAE_pp_V27": tail_summary["all_ladder_MAE_pp"]["V27"], "all_ladder_MAE_pp_V28": tail_summary["all_ladder_MAE_pp"]["V28"],
                     "pass": tail_summary["all_ladder_MAE_pp"]["V28"] <= tail_summary["all_ladder_MAE_pp"]["V27"] + 0.5},
         "L_discrimination": {"total_corr": [disc["V27"]["total"]["corr"], disc["V28"]["total"]["corr"]], "margin_corr": [disc["V27"]["margin"]["corr"], disc["V28"]["margin"]["corr"]],
                              "total_RMSE": [disc["V27"]["total"]["RMSE"], disc["V28"]["total"]["RMSE"]],
                              "pass": disc["V28"]["total"]["corr"] >= disc["V27"]["total"]["corr"] - 0.03 and disc["V28"]["margin"]["corr"] >= disc["V27"]["margin"]["corr"] - 0.05
                                      and disc["V28"]["total"]["RMSE"] <= disc["V27"]["total"]["RMSE"] + 0.25},
         "structural": {"max_TD_per_possession": int(A28["blocks"].td.max()), "GT8_blocks": m28["GT8_blocks"], "multi_TD_possession_share": m28["multi_TD_possession_share"]},
         "coherence": {"range": m28["coherence_range_TD_yards_ge70_minus_le10"], "pass_ge_0.45": m28["coherence_range_TD_yards_ge70_minus_le10"] >= 0.45},
         "total_vs_league_mean": {"V28_RMSE": disc["V28"]["total"]["RMSE"], "league_mean_RMSE": disc["LEAGUE_MEAN"]["total"]["RMSE"],
                                  "competitive_le_plus_0.35": disc["V28"]["total"]["RMSE"] <= disc["LEAGUE_MEAN"]["total"]["RMSE"] + 0.35}}
    kg = json.loads((R.OUT / "KAPPA_GRID_YARD_MARGINALS.json").read_text()) if (R.OUT / "KAPPA_GRID_YARD_MARGINALS.json").exists() else {"hist_cv": 0.225796938375857}
    G["M_dispersion"] = {"team_sd": [m28["team_sd"], hs["team_sd"]], "total_sd": [m28["total_sd"], hs["total_sd"]],
                         "pass": abs(m28["team_sd"] / hs["team_sd"] - 1) <= 0.12 and abs(m28["total_sd"] / hs["total_sd"] - 1) <= 0.12}
    G["intermediate_tiebreak"] = {"TD_possession_rate_gap": m28["TD_possession_rate"] - hd["TD_possession_rate"], "team_game_yards_cv": [m28["team_game_yards_cv"], kg["hist_cv"]],
                                  "sum_abs_gaps": abs(m28["TD_possession_rate"] - hd["TD_possession_rate"]) + abs(m28["team_game_yards_cv"] - kg["hist_cv"])}
    G["N_winner"] = {"brier_delta": prob_move["Brier_delta_V28_minus_V27"], "pass": prob_move["Brier_delta_V28_minus_V27"]["mean"] <= 0.02 and prob_move["Brier_delta_V28_minus_V27"]["ci95"][0] <= 0.0}
    G["intermediate_tiebreak"]["blocks_per_team_game"] = [m28["blocks_per_team_game"], hd["drives_per_team_game"]]
    G["intermediate_tiebreak"]["sum_abs_gaps_v2"] = abs(m28["TD_possession_rate"] - hd["TD_possession_rate"]) + abs(m28["blocks_per_team_game"] / hd["drives_per_team_game"] - 1.0)
    td_gap = m28["TD_possession_rate"] - hd["TD_possession_rate"]
    G["alignment_variant_trigger"] = {"TD_possession_rate_gap_vs_hist": td_gap, "abs_gt_0.010": abs(td_gap) > 0.010}
    run_meta = json.loads((R.out_dir(tag) / f"pilot_v{variant}" / "run_meta.json").read_text())
    acct = {}
    for r in run_meta["RESULTS"]:
        for k, v in r["accounting"].items():
            acct[k] = max(acct.get(k, 0), v) if k == "max_td_per_block" else acct.get(k, 0) + v
    out = {"SCHEMA": "SPORTS_NOVA_V28_ANALYSIS", "VARIANT": variant, "MODEL_VERSION": cfg.VERSIONS[variant], "TAG": tag, "V28_HASH": R.v28_hash(), "GAMES": len(games), "N_SIMS": R.N_SIMS,
           "GUARDED_UNCHANGED_DURING_RUN": run_meta["GUARDED_UNCHANGED"], "ACCOUNTING_TOTALS": acct,
           "V27_metrics": m27, "V28_metrics": m28, "STRUCTURE": struct, "DISCRIMINATION": disc, "PAIRED_BOOTSTRAP_over_games": boots, "PROBABILITY_MOVEMENT_VS_V27": prob_move,
           "TAILS": tails, "TAIL_SUMMARY": tail_summary, "HISTORY_SCORES_2014_2025": hs, "GATES": G}
    (R.out_dir(tag) / f"V28_ANALYSIS_v{variant}.json").write_text(json.dumps(out, indent=1, default=float))
    keyprint = {"accounting": acct, "structure": {k: v for k, v in struct.items() if k != "TD_by_yards_bin"}, "TD_by_yards_bin": struct["TD_by_yards_bin"],
                "discrimination": {k: {kk: vv for kk, vv in v.items()} for k, v in disc.items()}, "boots": boots, "prob_move": prob_move, "tail_summary": tail_summary, "gates": G}
    print(json.dumps(keyprint, indent=1, default=float))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "analyze":
        raise SystemExit(__doc__)
    cmd_analyze(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "V28_1")
